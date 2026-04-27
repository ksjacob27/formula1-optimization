import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from models.lstm import get_model

RESULTS_DIR = Path('results')
DATA_RAW    = Path('data/raw')
DATA_PROC   = Path('data/processed')

# Default pit loss (used when a circuit-specific value is not supplied).
PIT_LOSS_SECONDS       = 22.0

# Per-circuit pit loss (seconds): time penalty for diverting through the pit
# lane vs continuing at racing speed. Values are published F1 strategy estimates
# and reflect the 2024 calendar layout. Within-circuit variation across years
# is small (±1s) unless the pit lane is rebuilt.
# Source: F1 broadcast strategy graphics, Mercedes/Ferrari strategy briefings,
# and pit-lane length / pit-speed-limit calculations.
PIT_LOSS_BY_ROUND_2024 = {
    1:  22.0,  # Bahrain
    2:  19.0,  # Saudi Arabia (Jeddah, short pit lane)
    3:  21.0,  # Australia (Melbourne)
    4:  22.0,  # Japan (Suzuka)
    5:  22.0,  # China (Shanghai)
    6:  20.0,  # Miami
    7:  26.0,  # Imola (slow pit lane geometry)
    8:  24.0,  # Monaco
    9:  17.0,  # Canada (Montreal, fast pit lane)
    10: 22.0,  # Spain (Barcelona)
    11: 21.0,  # Austria (Red Bull Ring)
    12: 22.0,  # Britain (Silverstone)
    13: 20.0,  # Hungary
    14: 20.0,  # Belgium (Spa)
    15: 22.0,  # Netherlands (Zandvoort)
    16: 21.0,  # Italy (Monza)
    17: 18.0,  # Azerbaijan (Baku, fast pit lane)
    18: 28.0,  # Singapore (slow pit lane geometry)
    19: 22.0,  # USA (COTA)
    20: 22.0,  # Mexico
    21: 21.0,  # Brazil (São Paulo)
    22: 16.0,  # Las Vegas (very fast pit lane)
    23: 22.0,  # Qatar
    24: 22.0,  # Abu Dhabi
}

SEQUENCE_LENGTH        = 10
MAX_PREDICTION_HORIZON = 20
FEATURE_COLS           = [
    'LapTimeSeconds', 'StintLength', 'FuelLoad',
    'AirTemp', 'TrackTemp',
    'Compound_SOFT', 'Compound_MEDIUM', 'Compound_HARD',
    'TrackTemp_global',   # absolute temp, globally normalized across training races
]
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_COLS)}
FUEL_LOAD_KG   = 110.0
FUEL_BURN_RATE = 1.6


def load_model(
    model_type: str,
    device: torch.device,
    run_name: str = None,
    input_size: int = len(FEATURE_COLS)
) -> nn.Module:
    name  = run_name or model_type
    model = get_model(model_type, input_size=input_size).to(device)
    model.load_state_dict(torch.load(
        RESULTS_DIR / f'best_{name}.pt',
        map_location=device
    ))
    model.eval()
    return model


def _model_input_size(model: nn.Module) -> int:
    """Infer expected input feature count from model's RNN layer."""
    if hasattr(model, 'gru'):
        return model.gru.input_size
    if hasattr(model, 'lstm'):
        return model.lstm.input_size
    return len(FEATURE_COLS)


def _resolve_model(model_or_dict, compound: str) -> nn.Module:
    """Return the appropriate model for a given compound.

    Accepts either a single nn.Module (used for all compounds) or a dict
    mapping compound strings ('SOFT', 'MEDIUM', 'HARD') to nn.Module.
    Falls back to the single model if the compound key is missing from the dict.
    """
    if isinstance(model_or_dict, dict):
        return model_or_dict.get(compound, next(iter(model_or_dict.values())))
    return model_or_dict


def load_compound_models(
    device: torch.device,
    model_type: str = 'gru',
    run_prefix: str = 'gru_delta',
    input_size: int = 8,
) -> dict:
    """Load compound-specific models trained in Phase 2.

    Returns a dict {'SOFT': model, 'MEDIUM': model, 'HARD': model}.
    run_prefix+'_soft' / '_medium' / '_hard' must exist in RESULTS_DIR.
    """
    models = {}
    for compound in ('SOFT', 'MEDIUM', 'HARD'):
        run_name = f'{run_prefix}_{compound.lower()}'
        models[compound] = load_model(model_type, device, run_name=run_name, input_size=input_size)
    return models


def prepare_race(year: int, round_number: int, driver: str) -> pd.DataFrame:
    import fastf1
    fastf1.Cache.enable_cache(str(Path('cache')))

    session = fastf1.get_session(year, round_number, 'R')
    session.load(telemetry=False, weather=True, messages=False)

    # laps = session.laps.pick_quicklaps()
    # match training-time filtering: keep slow degraded end-of-stint laps,
    # drop only anomalous laps (SC/VSC/red flag, broken timing, pit-in laps)
    laps = session.laps
    laps = laps[laps['IsAccurate'] == True]
    laps = laps[laps['TrackStatus'] == '1']
    laps = laps[laps['PitInTime'].isna()]
    laps = laps[laps['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])]
    laps = laps[laps['Driver'] == driver].sort_values('LapNumber').reset_index(drop=True)

    if laps.empty:
        raise ValueError(f"No clean laps found for driver {driver}")

    laps['LapNumber'] = laps['LapNumber'].astype(int)

    weather = session.weather_data.sort_values('Time')
    laps = pd.merge_asof(
        laps.sort_values('LapStartTime'),
        weather[['Time', 'AirTemp', 'TrackTemp']],
        left_on='LapStartTime',
        right_on='Time',
        direction='nearest'
    ).sort_values('LapNumber').reset_index(drop=True)

    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    laps['FuelLoad']       = (FUEL_LOAD_KG - laps['LapNumber'] * FUEL_BURN_RATE).clip(lower=0)
    laps['StintLength']    = laps['TyreLife'].astype(int)

    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        laps[f'Compound_{compound}'] = (laps['Compound'] == compound).astype(int)

    return laps


def normalize_driver_race(laps: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    laps = laps.copy()

    # Apply global TrackTemp scaler (fitted on 2022+2023 training data) before
    # per-race normalization overwrites the raw values. This preserves absolute
    # temperature level — e.g. Las Vegas 18°C vs Bahrain 48°C look different.
    with open(DATA_PROC / 'global_tracktemp_scaler.pkl', 'rb') as f:
        global_tt_scaler = pickle.load(f)
    laps['TrackTemp_global'] = global_tt_scaler.transform(laps[['TrackTemp']]).flatten()

    scalers = {}
    for col in ['LapTimeSeconds', 'StintLength', 'FuelLoad', 'AirTemp', 'TrackTemp']:
        scaler = MinMaxScaler()
        laps[col] = scaler.fit_transform(laps[[col]])
        scalers[col] = scaler
    return laps, scalers


def predict_worn(
    model,
    laps_norm: pd.DataFrame,
    start_lap: int,
    n_future: int,
    compound: str,
    device: torch.device,
    delta: bool = False,
    mask_features: list[str] = None
) -> np.ndarray:
    """Predict lap times continuing on WORN tyres from current history.

    model may be a single nn.Module or a dict {'SOFT': ..., 'MEDIUM': ..., 'HARD': ...}.
    If a dict, the model for the current compound is selected automatically.
    """
    model    = _resolve_model(model, compound)
    n_feat   = _model_input_size(model)
    values   = laps_norm[FEATURE_COLS].values.copy()
    if mask_features:
        for fname in mask_features:
            values[:, FEATURE_INDEX[fname]] = 0.0
    seed_idx = max(0, start_lap - SEQUENCE_LENGTH)
    sequence = values[seed_idx:start_lap].tolist()

    if len(sequence) < SEQUENCE_LENGTH:
        pad      = [sequence[0]] * (SEQUENCE_LENGTH - len(sequence))
        sequence = pad + sequence

    compound_vec      = [
        1 if compound == 'SOFT'   else 0,
        1 if compound == 'MEDIUM' else 0,
        1 if compound == 'HARD'   else 0,
    ]
    current_tyre_life = sequence[-1][1]
    last_abs_lap_time = sequence[-1][0]  # needed for delta integration
    predictions       = []

    for i in range(n_future):
        window = sequence[-SEQUENCE_LENGTH:]
        seq_tensor = torch.tensor(
            [[row[:n_feat] for row in window]], dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            out = model(seq_tensor).item()

        if delta:
            abs_lap_time  = last_abs_lap_time + out
            last_abs_lap_time = abs_lap_time
        else:
            abs_lap_time = out

        predictions.append(abs_lap_time)

        last_row      = list(sequence[-1])
        next_row      = last_row.copy()
        next_row[0]   = abs_lap_time
        next_row[1]   = current_tyre_life + (i + 1)
        next_row[2]   = max(0, last_row[2] - FUEL_BURN_RATE / 110.0)
        next_row[5:8] = compound_vec
        sequence.append(next_row)

    return np.array(predictions)


def predict_fresh(
    model,
    laps_norm: pd.DataFrame,
    start_lap: int,
    n_future: int,
    compound: str,
    device: torch.device,
    delta: bool = False,
    mask_features: list[str] = None
) -> np.ndarray:
    """Predict lap times on FRESH tyres by seeding with low stint-length laps.

    model may be a single nn.Module or a dict {'SOFT': ..., 'MEDIUM': ..., 'HARD': ...}.
    If a dict, the model for the fresh compound is selected automatically.
    """
    model        = _resolve_model(model, compound)
    compound_col = f'Compound_{compound}'

    if compound_col in laps_norm.columns:
        fresh_laps = laps_norm[
            (laps_norm[compound_col] == 1) &
            (laps_norm['StintLength'] < 0.2)
        ]
    else:
        fresh_laps = laps_norm[laps_norm['StintLength'] < 0.2]

    n_feat = _model_input_size(model)

    if len(fresh_laps) >= SEQUENCE_LENGTH:
        seed = fresh_laps[FEATURE_COLS].values[:SEQUENCE_LENGTH].tolist()
    else:
        values   = laps_norm[FEATURE_COLS].values.copy()
        seed_idx = max(0, start_lap - SEQUENCE_LENGTH)
        seed     = values[seed_idx:start_lap].tolist()
        if len(seed) < SEQUENCE_LENGTH:
            pad  = [seed[0]] * (SEQUENCE_LENGTH - len(seed))
            seed = pad + seed
        compound_vec_local = [
            1 if compound == 'SOFT'   else 0,
            1 if compound == 'MEDIUM' else 0,
            1 if compound == 'HARD'   else 0,
        ]
        for row in seed:
            row[1]   = 0.0
            row[5:8] = compound_vec_local

    if mask_features:
        for fname in mask_features:
            idx = FEATURE_INDEX[fname]
            for row in seed:
                row[idx] = 0.0

    compound_vec = [
        1 if compound == 'SOFT'   else 0,
        1 if compound == 'MEDIUM' else 0,
        1 if compound == 'HARD'   else 0,
    ]

    sequence          = [list(r) for r in seed]
    last_abs_lap_time = sequence[-1][0]  # needed for delta integration
    predictions       = []

    for i in range(n_future):
        window = sequence[-SEQUENCE_LENGTH:]
        seq_tensor = torch.tensor(
            [[row[:n_feat] for row in window]], dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            out = model(seq_tensor).item()

        if delta:
            abs_lap_time      = last_abs_lap_time + out
            last_abs_lap_time = abs_lap_time
        else:
            abs_lap_time = out

        predictions.append(abs_lap_time)

        last_row      = list(sequence[-1])
        next_row      = last_row.copy()
        next_row[0]   = abs_lap_time
        next_row[1]   = (i + 1) / 40.0
        next_row[2]   = max(0, last_row[2] - FUEL_BURN_RATE / 110.0)
        next_row[5:8] = compound_vec
        sequence.append(next_row)

    return np.array(predictions)


def find_optimal_two_stop(
    model: nn.Module,
    laps_norm: pd.DataFrame,
    laps_raw: pd.DataFrame,
    pit_loss_norm: float,
    device: torch.device,
    start_lap: int,
    total_laps: int,
    fresh_compounds: tuple[str, str] = ('MEDIUM', 'HARD'),
    delta: bool = False,
    mask_features: list[str] = None
) -> tuple[int, int, float]:
    """
    Search over (pit_1, pit_2) pairs and return the best 2-stop strategy.
    Total time is computed from start_lap to total_laps.

    Restricts pit_1 to [start_lap+1, total_laps//2] and
    pit_2 to [pit_1+8, total_laps-SEQUENCE_LENGTH] to keep compute tractable.

    Returns (pit_1, pit_2, total_normalized_time).
    """
    compound_at_start = laps_raw['Compound'].iloc[
        min(start_lap - 1, len(laps_raw) - 1)
    ]

    best_total = float('inf')
    best_pit_1 = None
    best_pit_2 = None

    worn_sums = {}  # cache: n_worn → sum of worn predictions

    for pit_1 in range(start_lap + 1, total_laps // 2 + 1):
        n_worn = min(pit_1 - start_lap, MAX_PREDICTION_HORIZON)
        if n_worn not in worn_sums:
            worn_preds = predict_worn(
                model, laps_norm, start_lap, n_worn,
                compound=compound_at_start,
                device=device, delta=delta, mask_features=mask_features
            )
            worn_sums[n_worn] = worn_preds.sum()
        worn_total = worn_sums[n_worn]

        for pit_2 in range(pit_1 + 8, total_laps - SEQUENCE_LENGTH + 1):
            n_fresh_1 = min(pit_2 - pit_1, MAX_PREDICTION_HORIZON)
            fresh_1 = predict_fresh(
                model, laps_norm, pit_1, n_fresh_1,
                compound=fresh_compounds[0],
                device=device, delta=delta, mask_features=mask_features
            )

            n_fresh_2 = min(total_laps - pit_2, MAX_PREDICTION_HORIZON)
            fresh_2 = predict_fresh(
                model, laps_norm, pit_2, n_fresh_2,
                compound=fresh_compounds[1],
                device=device, delta=delta, mask_features=mask_features
            )

            total = (worn_total + pit_loss_norm
                     + fresh_1.sum() + pit_loss_norm + fresh_2.sum())

            if total < best_total:
                best_total = total
                best_pit_1 = pit_1
                best_pit_2 = pit_2

    return best_pit_1, best_pit_2, best_total


def find_optimal_pit_window(
    model: nn.Module,
    laps_norm: pd.DataFrame,
    laps_raw: pd.DataFrame,
    scalers: dict,
    device: torch.device,
    total_laps: int = 57,
    fresh_compound: str = 'MEDIUM',
    pit_loss_seconds: float = PIT_LOSS_SECONDS,
    delta: bool = False,
    mask_features: list[str] = None
) -> dict:
    pit_lap_range   = range(SEQUENCE_LENGTH + 1, total_laps - SEQUENCE_LENGTH)
    results         = []

    lap_time_scaler = scalers['LapTimeSeconds']
    lap_time_max    = lap_time_scaler.data_max_[0]
    lap_time_min    = lap_time_scaler.data_min_[0]
    lap_time_range  = lap_time_max - lap_time_min
    pit_loss_norm   = pit_loss_seconds / lap_time_max

    print(f"Lap time range (s): {lap_time_range:.2f}")
    print(f"Lap time max (s):   {lap_time_max:.2f}")
    print(f"Pit loss (s):        {pit_loss_seconds:.1f}")
    print(f"Pit loss normalized: {pit_loss_norm:.4f}")
    print(f"Avg normalized lap time: {laps_norm['LapTimeSeconds'].mean():.4f}")

    for pit_lap in pit_lap_range:
        remaining    = min(total_laps - pit_lap, MAX_PREDICTION_HORIZON)
        compound_idx = min(pit_lap - 1, len(laps_raw) - 1)

        stay_preds = predict_worn(
            model, laps_norm, pit_lap, remaining,
            compound=laps_raw['Compound'].iloc[compound_idx],
            device=device,
            delta=delta,
            mask_features=mask_features
        )
        pit_preds = predict_fresh(
            model, laps_norm, pit_lap, remaining,
            compound=fresh_compound,
            device=device,
            delta=delta,
            mask_features=mask_features
        )

        pit_total  = pit_loss_norm + pit_preds.sum()
        stay_total = stay_preds.sum()

        if pit_lap == 20:
            print(f"\nDebug pit_lap=20:")
            print(f"  Stay preds: {stay_preds[:5]}")
            print(f"  Pit preds:  {pit_preds[:5]}")
            print(f"  Difference: {(pit_preds[:5] - stay_preds[:5])}")
            print(f"  Stay total: {stay_total:.4f}")
            print(f"  Pit total:  {pit_total:.4f}")
            print(f"  Delta:      {pit_total - stay_total:.4f}")

        results.append({
            'pit_lap':    pit_lap,
            'stay_total': stay_total,
            'pit_total':  pit_total,
            'delta':      pit_total - stay_total
        })

    results_df = pd.DataFrame(results)

    # Find first crossover (first lap where pitting becomes faster)
    first_pit_lap = None
    for i in range(1, len(results_df)):
        if results_df['delta'].iloc[i] < 0 and results_df['delta'].iloc[i - 1] >= 0:
            first_pit_lap = int(results_df['pit_lap'].iloc[i])
            break

    # Detect all pit windows (contiguous regions where delta < 0)
    pit_windows = []
    in_window   = False
    window_start = None

    for _, row in results_df.iterrows():
        if row['delta'] < 0 and not in_window:
            in_window    = True
            window_start = int(row['pit_lap'])
        elif row['delta'] >= 0 and in_window:
            in_window   = False
            window_end  = int(row['pit_lap']) - 1
            window_data = results_df[
                (results_df['pit_lap'] >= window_start) &
                (results_df['pit_lap'] <= window_end)
            ]
            # best_lap = int(window_data.loc[window_data['delta'].idxmin(), 'pit_lap'])
            # use first crossover — pitting at local minimum means staying out longer than necessary
            best_lap = window_start
            pit_windows.append({'start': window_start, 'end': window_end, 'best': best_lap})

    # Close final window if still open at end of race
    if in_window:
        window_end  = int(results_df['pit_lap'].iloc[-1])
        # best_lap    = int(window_data.loc[window_data['delta'].idxmin(), 'pit_lap'])
        best_lap    = window_start
        pit_windows.append({'start': window_start, 'end': window_end, 'best': best_lap})

    # Primary recommendation = first crossover, fallback to global minimum
    best_pit_lap = first_pit_lap if first_pit_lap else (
        int(results_df.loc[results_df['delta'].idxmin(), 'pit_lap'])
    )

    print(f"\nPit windows detected: {len(pit_windows)}")
    for i, w in enumerate(pit_windows):
        print(f"  Window {i+1}: laps {w['start']}–{w['end']} (best: lap {w['best']})")
    print(f"Primary recommendation: lap {best_pit_lap}")

    # Compare 1-stop vs 2-stop total race time from lap SEQUENCE_LENGTH onwards.
    # Each strategy is evaluated from a common start point for a fair comparison.
    start_lap = SEQUENCE_LENGTH
    n_to_pit  = min(best_pit_lap - start_lap, MAX_PREDICTION_HORIZON)

    worn_to_pit = predict_worn(
        model, laps_norm, start_lap, n_to_pit,
        compound=laps_raw['Compound'].iloc[min(start_lap - 1, len(laps_raw) - 1)],
        device=device, delta=delta, mask_features=mask_features
    )
    n_after_pit = min(total_laps - best_pit_lap, MAX_PREDICTION_HORIZON)
    fresh_after = predict_fresh(
        model, laps_norm, best_pit_lap, n_after_pit,
        compound=fresh_compound,
        device=device, delta=delta, mask_features=mask_features
    )
    one_stop_total = worn_to_pit.sum() + pit_loss_norm + fresh_after.sum()

    pit_1, pit_2, two_stop_total = find_optimal_two_stop(
        model, laps_norm, laps_raw, pit_loss_norm, device,
        start_lap=start_lap,
        total_laps=total_laps,
        fresh_compounds=(fresh_compound, fresh_compound),
        delta=delta,
        mask_features=mask_features
    )

    if two_stop_total < one_stop_total and pit_1 is not None:
        recommended_pit_laps = [pit_1, pit_2]
        print(f"2-stop faster: pit laps {pit_1} + {pit_2} "
              f"(total {two_stop_total:.4f} vs 1-stop {one_stop_total:.4f})")
    else:
        recommended_pit_laps = [best_pit_lap]
        print(f"1-stop faster: pit lap {best_pit_lap} "
              f"(total {one_stop_total:.4f} vs 2-stop {two_stop_total:.4f})")

    return {
        'best_pit_lap':          best_pit_lap,
        'recommended_pit_laps':  recommended_pit_laps,
        'one_stop_total':        one_stop_total,
        'two_stop_total':        two_stop_total,
        'pit_windows':           pit_windows,
        'results':               results_df
    }


def plot_strategy(
    laps_raw: pd.DataFrame,
    laps_norm: pd.DataFrame,
    scalers: dict,
    model: nn.Module,
    device: torch.device,
    best_pit_lap: int,
    results_df: pd.DataFrame,
    pit_windows: list,
    driver: str,
    year: int,
    round_number: int
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    actual_laps  = laps_raw['LapNumber'].values
    actual_times = laps_raw['LapTimeSeconds'].values

    axes[0].plot(actual_laps, actual_times, 'o-', linewidth=1.5,
                 markersize=3, label='Actual lap times', color='steelblue')

    if len(laps_norm) > SEQUENCE_LENGTH:
        n_future   = len(laps_norm) - SEQUENCE_LENGTH
        preds_norm = predict_worn(
            model, laps_norm, SEQUENCE_LENGTH, n_future,
            compound=laps_raw['Compound'].iloc[SEQUENCE_LENGTH],
            device=device
        )
        preds_actual = scalers['LapTimeSeconds'].inverse_transform(
            preds_norm.reshape(-1, 1)
        ).flatten()
        pred_laps = actual_laps[SEQUENCE_LENGTH:SEQUENCE_LENGTH + len(preds_actual)]
        axes[0].plot(pred_laps, preds_actual, '--', linewidth=1.5,
                     label='Predicted lap times', color='orange', alpha=0.8)

    # Mark actual pit stops
    # pit_laps = laps_raw[
    #     (laps_raw['TyreLife'] <= 2) & (laps_raw['LapNumber'] > 3)
    # ]['LapNumber'].values
    # detect by TyreLife decrease — robust when pick_quicklaps filters the out-lap
    prev_tyre = laps_raw['TyreLife'].shift(1)
    pit_laps = laps_raw[
        (laps_raw['TyreLife'] < prev_tyre) & (laps_raw['LapNumber'] > 3)
    ]['LapNumber'].values
    for i, pl in enumerate(pit_laps):
        axes[0].axvline(pl, color='red', linestyle=':', alpha=0.7, linewidth=1.2,
                        label='Actual pit stop' if i == 0 else None)

    # Shade all pit windows and mark best lap in each
    for i, window in enumerate(pit_windows):
        axes[0].axvspan(window['start'], window['end'],
                        alpha=0.15, color='green',
                        label='Pit window' if i == 0 else None)
        axes[0].axvline(window['best'], color='green', linestyle='--', linewidth=1.5,
                        label=f"Best lap in window {i+1}: {window['best']}")

    axes[0].set_xlabel('Lap Number')
    axes[0].set_ylabel('Lap Time (s)')
    axes[0].set_title(f'{driver} — {year} Round {round_number}: Predicted Degradation & Pit Windows')
    axes[0].legend(fontsize=8)

    # Delta plot
    axes[1].plot(results_df['pit_lap'], results_df['delta'],
                 linewidth=1.5, color='purple')
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=1)

    for i, window in enumerate(pit_windows):
        axes[1].axvline(window['best'], color='green', linestyle='--', linewidth=1.5,
                        label=f"Best lap window {i+1}: {window['best']}" if i == 0 else f"lap {window['best']}")

    axes[1].fill_between(
        results_df['pit_lap'], results_df['delta'], 0,
        where=results_df['delta'] < 0,
        alpha=0.2, color='green', label='Pitting is faster'
    )
    axes[1].fill_between(
        results_df['pit_lap'], results_df['delta'], 0,
        where=results_df['delta'] >= 0,
        alpha=0.2, color='red', label='Staying out is faster'
    )
    axes[1].set_xlabel('Pit Lap')
    axes[1].set_ylabel('Delta (pit - stay, normalized)')
    axes[1].set_title('Pit Stop Decision: Cumulative Time Delta')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'strategy_{driver}_{year}_r{round_number}.png', dpi=150)
    plt.show()
    print(f"Saved strategy plot for {driver}")


def evaluate_strategy_across_races(
    model: nn.Module,
    device: torch.device,
    races: list[dict],
    plot: bool = False,
    delta: bool = False,
    mask_features: list[str] = None
) -> pd.DataFrame:
    results = []

    for race in races:
        year           = race['year']
        round_num      = race['round']
        driver         = race['driver']
        total_laps     = race['total_laps']
        fresh_compound = race['fresh_compound']
        pit_loss       = race.get('pit_loss', PIT_LOSS_SECONDS)

        print(f"\n{'='*50}")
        print(f"Processing {year} Round {round_num} — {driver} (pit loss {pit_loss}s)")
        print(f"{'='*50}")

        try:
            laps_raw           = prepare_race(year, round_num, driver)
            laps_norm, scalers = normalize_driver_race(laps_raw)

            print(f"Clean laps: {len(laps_raw)} | Compounds: {laps_raw['Compound'].unique()}")
            print(f"  TyreLife values: {sorted(laps_raw['TyreLife'].unique())}")
            print(f"  Min lap with TyreLife<=3: {laps_raw[laps_raw['TyreLife'] <= 3]['LapNumber'].values}")

            result       = find_optimal_pit_window(
                model, laps_norm, laps_raw, scalers, device,
                total_laps=total_laps,
                fresh_compound=fresh_compound,
                pit_loss_seconds=pit_loss,
                delta=delta,
                mask_features=mask_features
            )
            best_pit_lap         = result['best_pit_lap']
            recommended_pit_laps = result['recommended_pit_laps']
            results_df           = result['results']
            pit_windows          = result['pit_windows']

            # actual_pit_laps = laps_raw[
            #     (laps_raw['TyreLife'] <= 2) & (laps_raw['LapNumber'] > 3)
            # ]['LapNumber'].values
            # detect by TyreLife decrease — robust when pick_quicklaps filters the out-lap
            _prev_tyre      = laps_raw['TyreLife'].shift(1)
            actual_pit_laps = laps_raw[
                (laps_raw['TyreLife'] < _prev_tyre) & (laps_raw['LapNumber'] > 3)
            ]['LapNumber'].values

            if len(actual_pit_laps) == 0:
                print(f"  No actual pit laps found — skipping")
                continue

            # Score the first recommended pit lap against the nearest actual pit.
            # This is forgiving of strategy-count mismatches (which our TyreLife
            # detection doesn't reliably resolve) while still rewarding correct
            # first-stop timing — the decision with the most strategic leverage.
            predicted_sorted = sorted(recommended_pit_laps)
            first_predicted  = predicted_sorted[0]
            closest_actual   = actual_pit_laps[
                np.argmin(np.abs(actual_pit_laps - first_predicted))
            ]
            error    = abs(first_predicted - closest_actual)
            within_5 = error <= 5

            strategy_label = f"{len(predicted_sorted)}-stop"
            print(f"  Predicted pit lap(s): {predicted_sorted} ({strategy_label})")
            print(f"  Actual pit lap(s):    {list(actual_pit_laps)}")
            print(f"  Error: {error} laps | Within ±5: {'✓' if within_5 else '✗'}")

            if plot:
                plot_strategy(
                    laps_raw, laps_norm, scalers, model, device,
                    best_pit_lap, results_df, pit_windows,
                    driver, year, round_num
                )

            results.append({
                'year':                year,
                'round':               round_num,
                'driver':              driver,
                'pit_loss':            pit_loss,
                'predicted_strategy':  strategy_label,
                'predicted_pit':       predicted_sorted[0],
                'predicted_pits':      str(predicted_sorted),
                'actual_pits':         str(list(actual_pit_laps)),
                'closest_actual':      closest_actual,
                'error_laps':          error,
                'within_5_laps':       within_5
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    return pd.DataFrame(results)


if __name__ == '__main__':
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    USE_DELTA     = True
    MASK_FEATURES = None

    # Phase 2: use compound-specific models if available, else fall back to Run 4 baseline.
    _compound_weights = [
        RESULTS_DIR / f'best_gru_delta_{c.lower()}.pt'
        for c in ('SOFT', 'MEDIUM', 'HARD')
    ]
    if all(p.exists() for p in _compound_weights):
        print("Loading compound-specific models (Phase 2).")
        model = load_compound_models(device, model_type='gru', run_prefix='gru_delta', input_size=9)
    else:
        print("Compound models not found — using gru_baseline_delta (Phase 1 baseline).")
        model = load_model('gru', device, run_name='gru_baseline_delta', input_size=8)

    # 2024 calendar — all 24 rounds. total_laps is the official race distance.
    # Each entry is evaluated for VER as primary driver; we also add NOR as a
    # secondary driver for additional sample size. Races where the driver did
    # not finish (or had an unusual race) are skipped automatically by the
    # exception handler in evaluate_strategy_across_races.
    RACE_META_2024 = [
        # round, total_laps, fresh_compound
        (1,  57, 'HARD'),    # Bahrain
        (2,  50, 'HARD'),    # Saudi Arabia
        (3,  58, 'HARD'),    # Australia
        (4,  53, 'MEDIUM'),  # Japan
        (5,  56, 'HARD'),    # China
        (6,  57, 'HARD'),    # Miami
        (7,  63, 'HARD'),    # Imola
        (8,  78, 'HARD'),    # Monaco
        (9,  70, 'HARD'),    # Canada
        (10, 66, 'MEDIUM'),  # Spain
        (11, 71, 'HARD'),    # Austria
        (12, 52, 'HARD'),    # Britain
        (13, 70, 'HARD'),    # Hungary
        (14, 44, 'HARD'),    # Belgium
        (15, 72, 'HARD'),    # Netherlands
        (16, 53, 'HARD'),    # Italy
        (17, 51, 'HARD'),    # Azerbaijan
        (18, 62, 'HARD'),    # Singapore
        (19, 56, 'HARD'),    # USA
        (20, 71, 'HARD'),    # Mexico
        (21, 69, 'HARD'),    # Brazil
        (22, 50, 'HARD'),    # Las Vegas
        (23, 57, 'HARD'),    # Qatar
        (24, 58, 'HARD'),    # Abu Dhabi
    ]

    DRIVERS = ['VER', 'NOR', 'LEC', 'SAI', 'HAM', 'RUS']

    TEST_RACES = []
    for round_num, total_laps, fresh_compound in RACE_META_2024:
        for driver in DRIVERS:
            TEST_RACES.append({
                'year':           2024,
                'round':          round_num,
                'driver':         driver,
                'total_laps':     total_laps,
                'fresh_compound': fresh_compound,
                'pit_loss':       PIT_LOSS_BY_ROUND_2024.get(round_num, PIT_LOSS_SECONDS),
            })

    print(f"Evaluating {len(TEST_RACES)} race-driver combinations "
          f"({len(RACE_META_2024)} races × {len(DRIVERS)} drivers)")

    summary = evaluate_strategy_across_races(
        model, device, TEST_RACES,
        plot=False, delta=USE_DELTA,
        mask_features=MASK_FEATURES
    )

    print(f"\n{'='*50}")
    print(f"STRATEGY EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(summary[['year', 'round', 'driver', 'predicted_strategy', 'predicted_pits',
                    'actual_pits', 'error_laps', 'within_5_laps']].to_string(index=False))

    accuracy = summary['within_5_laps'].mean() * 100
    print(f"\nOverall accuracy (within ±5 laps): {accuracy:.1f}%")
    print(f"Races evaluated: {len(summary)}")
    print(f"Secondary success criterion (≥70%): {'✓ PASSED' if accuracy >= 70 else '✗ FAILED'}")

    summary.to_csv('results/strategy_summary.csv', index=False)
    print(f"\nSaved results/strategy_summary.csv")