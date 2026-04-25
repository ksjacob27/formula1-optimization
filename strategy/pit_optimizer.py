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

PIT_LOSS_SECONDS       = 22.0
SEQUENCE_LENGTH        = 10
MAX_PREDICTION_HORIZON = 20
FEATURE_COLS           = [
    'LapTimeSeconds', 'StintLength', 'FuelLoad',
    'AirTemp', 'TrackTemp',
    'Compound_SOFT', 'Compound_MEDIUM', 'Compound_HARD'
]
FUEL_LOAD_KG   = 110.0
FUEL_BURN_RATE = 1.6


def load_model(model_type: str, device: torch.device) -> nn.Module:
    model = get_model(model_type).to(device)
    model.load_state_dict(torch.load(
        RESULTS_DIR / f'best_{model_type}.pt',
        map_location=device
    ))
    model.eval()
    return model


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
    scalers = {}
    for col in ['LapTimeSeconds', 'StintLength', 'FuelLoad', 'AirTemp', 'TrackTemp']:
        scaler = MinMaxScaler()
        laps[col] = scaler.fit_transform(laps[[col]])
        scalers[col] = scaler
    return laps, scalers


def predict_worn(
    model: nn.Module,
    laps_norm: pd.DataFrame,
    start_lap: int,
    n_future: int,
    compound: str,
    device: torch.device
) -> np.ndarray:
    """Predict lap times continuing on WORN tyres from current history."""
    values   = laps_norm[FEATURE_COLS].values.copy()
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
    predictions       = []

    for i in range(n_future):
        seq_tensor = torch.tensor(
            [sequence[-SEQUENCE_LENGTH:]], dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            pred = model(seq_tensor).item()

        predictions.append(pred)

        last_row      = list(sequence[-1])
        next_row      = last_row.copy()
        next_row[0]   = pred
        next_row[1]   = current_tyre_life + (i + 1)
        next_row[2]   = max(0, last_row[2] - FUEL_BURN_RATE / 110.0)
        next_row[5:8] = compound_vec
        sequence.append(next_row)

    return np.array(predictions)


def predict_fresh(
    model: nn.Module,
    laps_norm: pd.DataFrame,
    start_lap: int,
    n_future: int,
    compound: str,
    device: torch.device
) -> np.ndarray:
    """Predict lap times on FRESH tyres by seeding with low stint-length laps."""
    compound_col = f'Compound_{compound}'

    if compound_col in laps_norm.columns:
        fresh_laps = laps_norm[
            (laps_norm[compound_col] == 1) &
            (laps_norm['StintLength'] < 0.2)
        ]
    else:
        fresh_laps = laps_norm[laps_norm['StintLength'] < 0.2]

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

    compound_vec = [
        1 if compound == 'SOFT'   else 0,
        1 if compound == 'MEDIUM' else 0,
        1 if compound == 'HARD'   else 0,
    ]

    sequence    = [list(r) for r in seed]
    predictions = []

    for i in range(n_future):
        seq_tensor = torch.tensor(
            [sequence[-SEQUENCE_LENGTH:]], dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            pred = model(seq_tensor).item()

        predictions.append(pred)

        last_row      = list(sequence[-1])
        next_row      = last_row.copy()
        next_row[0]   = pred
        next_row[1]   = (i + 1) / 40.0
        next_row[2]   = max(0, last_row[2] - FUEL_BURN_RATE / 110.0)
        next_row[5:8] = compound_vec
        sequence.append(next_row)

    return np.array(predictions)


def find_optimal_pit_window(
    model: nn.Module,
    laps_norm: pd.DataFrame,
    laps_raw: pd.DataFrame,
    scalers: dict,
    device: torch.device,
    total_laps: int = 57,
    fresh_compound: str = 'MEDIUM'
) -> dict:
    pit_lap_range   = range(SEQUENCE_LENGTH + 1, total_laps - SEQUENCE_LENGTH)
    results         = []

    lap_time_scaler = scalers['LapTimeSeconds']
    lap_time_max    = lap_time_scaler.data_max_[0]
    lap_time_min    = lap_time_scaler.data_min_[0]
    lap_time_range  = lap_time_max - lap_time_min
    pit_loss_norm   = PIT_LOSS_SECONDS / lap_time_max

    print(f"Lap time range (s): {lap_time_range:.2f}")
    print(f"Lap time max (s):   {lap_time_max:.2f}")
    print(f"Pit loss normalized: {pit_loss_norm:.4f}")
    print(f"Avg normalized lap time: {laps_norm['LapTimeSeconds'].mean():.4f}")

    for pit_lap in pit_lap_range:
        remaining    = min(total_laps - pit_lap, MAX_PREDICTION_HORIZON)
        compound_idx = min(pit_lap - 1, len(laps_raw) - 1)

        stay_preds = predict_worn(
            model, laps_norm, pit_lap, remaining,
            compound=laps_raw['Compound'].iloc[compound_idx],
            device=device
        )
        pit_preds = predict_fresh(
            model, laps_norm, pit_lap, remaining,
            compound=fresh_compound,
            device=device
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

    return {
        'best_pit_lap': best_pit_lap,
        'pit_windows':  pit_windows,
        'results':      results_df
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
    races: list[dict]
) -> pd.DataFrame:
    results = []

    for race in races:
        year           = race['year']
        round_num      = race['round']
        driver         = race['driver']
        total_laps     = race['total_laps']
        fresh_compound = race['fresh_compound']

        print(f"\n{'='*50}")
        print(f"Processing {year} Round {round_num} — {driver}")
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
                fresh_compound=fresh_compound
            )
            best_pit_lap = result['best_pit_lap']
            results_df   = result['results']
            pit_windows  = result['pit_windows']

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

            closest_actual = actual_pit_laps[
                np.argmin(np.abs(actual_pit_laps - best_pit_lap))
            ]
            error    = abs(best_pit_lap - closest_actual)
            within_2 = error <= 2

            print(f"  Predicted pit lap: {best_pit_lap}")
            print(f"  Actual pit lap(s): {actual_pit_laps}")
            print(f"  Error: {error} laps | Within ±2: {'✓' if within_2 else '✗'}")

            plot_strategy(
                laps_raw, laps_norm, scalers, model, device,
                best_pit_lap, results_df, pit_windows,
                driver, year, round_num
            )

            results.append({
                'year':           year,
                'round':          round_num,
                'driver':         driver,
                'predicted_pit':  best_pit_lap,
                'actual_pits':    list(actual_pit_laps),
                'closest_actual': closest_actual,
                'error_laps':     error,
                'within_2_laps':  within_2
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    return pd.DataFrame(results)


if __name__ == '__main__':
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_model('gru', device)

    TEST_RACES = [
        {'year': 2024, 'round': 1, 'driver': 'VER', 'total_laps': 57, 'fresh_compound': 'HARD'},
        {'year': 2024, 'round': 2, 'driver': 'VER', 'total_laps': 50, 'fresh_compound': 'HARD'},  # swap LEC -> VER
        {'year': 2024, 'round': 3, 'driver': 'NOR', 'total_laps': 58, 'fresh_compound': 'HARD'},
        {'year': 2024, 'round': 4, 'driver': 'VER', 'total_laps': 53, 'fresh_compound': 'MEDIUM'},
        {'year': 2024, 'round': 5, 'driver': 'VER', 'total_laps': 56, 'fresh_compound': 'HARD'},  # swap LEC -> VER
    ]

    summary = evaluate_strategy_across_races(model, device, TEST_RACES)

    print(f"\n{'='*50}")
    print(f"STRATEGY EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(summary[['year', 'round', 'driver', 'predicted_pit',
                    'closest_actual', 'error_laps', 'within_2_laps']].to_string(index=False))

    accuracy = summary['within_2_laps'].mean() * 100
    print(f"\nOverall accuracy (within ±2 laps): {accuracy:.1f}%")
    print(f"Races evaluated: {len(summary)}")
    print(f"Secondary success criterion (≥70%): {'✓ PASSED' if accuracy >= 70 else '✗ FAILED'}")

    summary.to_csv('results/strategy_summary.csv', index=False)
    print(f"\nSaved results/strategy_summary.csv")