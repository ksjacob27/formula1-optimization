import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pickle

DATA_RAW  = Path(__file__).resolve().parent.parent / 'data' / 'raw'
DATA_PROC = Path(__file__).resolve().parent.parent / 'data' / 'processed'
DATA_PROC.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 10
FEATURE_COLS = [
    'LapTimeSeconds', 'StintLength', 'FuelLoad',
    'AirTemp', 'TrackTemp',
    'Compound_SOFT', 'Compound_MEDIUM', 'Compound_HARD',
    'TrackTemp_global',   # absolute temp, globally normalized across all training races
]
TARGET_COL = 'LapTimeSeconds'


def build_sequences(
    group: pd.DataFrame,
    delta: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding window sequences from a single driver stint.
    Returns X of shape (n_samples, sequence_length, n_features)
    and y of shape (n_samples,).

    If delta=True, y is the lap-to-lap change in LapTimeSeconds
    (LapTime[t+1] - LapTime[t]) rather than the absolute value.
    X always contains absolute feature values so the model retains
    full context. Delta values are in the same normalized units as
    LapTimeSeconds (no additional scaling applied).
    """
    values = group[FEATURE_COLS].values
    target = group[TARGET_COL].values

    X, y = [], []
    for i in range(len(values) - SEQUENCE_LENGTH):
        X.append(values[i:i + SEQUENCE_LENGTH])
        if delta:
            # target = how much lap time changes on the next lap
            y.append(target[i + SEQUENCE_LENGTH] - target[i + SEQUENCE_LENGTH - 1])
        else:
            y.append(target[i + SEQUENCE_LENGTH])

    return np.array(X), np.array(y)


def normalize_race(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Normalize continuous features per race using MinMaxScaler.
    Returns normalized df and a dict of fitted scalers.
    """
    df = df.copy()
    scalers = {}
    continuous_cols = ['LapTimeSeconds', 'StintLength', 'FuelLoad',
                       'AirTemp', 'TrackTemp']

    for col in continuous_cols:
        scaler = MinMaxScaler()
        df[col] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler

    return df, scalers


def process_split(
    parquet_paths: Path | list[Path],
    split_name: str,
    delta: bool = False,
    global_tt_scaler=None
) -> None:
    """
    Load one or more season parquets, build sequences per driver per stint,
    normalize per race, and save tensors.

    global_tt_scaler: fitted MinMaxScaler for TrackTemp across all training
    races. Applied before per-race normalization so the absolute temperature
    level is preserved in the TrackTemp_global column (index 8). Must be
    pre-fitted on training data and passed for both train and test splits so
    they share the same scale.

    If delta=True, saves delta-target tensors with a '_delta' suffix so
    they coexist alongside the absolute-target tensors for comparison.
    """
    if isinstance(parquet_paths, Path):
        parquet_paths = [parquet_paths]
    df = pd.concat([pd.read_parquet(p) for p in parquet_paths], ignore_index=True)

    # Apply global scaler to raw TrackTemp before per-race normalization overwrites it
    df['TrackTemp_global'] = global_tt_scaler.transform(df[['TrackTemp']]).flatten()

    all_X, all_y = [], []

    for (year, round_num), race_df in df.groupby(['Year', 'Round']):
        race_df, _ = normalize_race(race_df)

        for driver, driver_df in race_df.groupby('Driver'):
            driver_df = driver_df.sort_values('LapNumber')

            # Split into individual stints (tyre life resets at pit stop)
            # A new stint starts when TyreLife would decrease
            # We detect this via StintLength going back to a low value
            stint_groups = detect_stints(driver_df)

            for stint_df in stint_groups:
                if len(stint_df) <= SEQUENCE_LENGTH:
                    continue  # stint too short to build any sequences
                X, y = build_sequences(stint_df, delta=delta)
                all_X.append(X)
                all_y.append(y)

    X_all = np.concatenate(all_X, axis=0).astype(np.float32)
    y_all = np.concatenate(all_y, axis=0).astype(np.float32)

    X_tensor = torch.tensor(X_all, dtype=torch.float32)
    y_tensor = torch.tensor(y_all, dtype=torch.float32)

    suffix = f'_{split_name}_delta' if delta else f'_{split_name}'
    torch.save(X_tensor, DATA_PROC / f'X{suffix}.pt')
    torch.save(y_tensor, DATA_PROC / f'y{suffix}.pt')

    mode_str = 'delta' if delta else 'absolute'
    print(f"{split_name} ({mode_str}): {X_tensor.shape[0]} sequences, "
          f"input shape {X_tensor.shape}")


def detect_stints(driver_df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Split a driver's race into individual stints based on StintLength.
    A new stint begins when StintLength resets (drops significantly).
    """
    stints = []
    current_stint = []

    prev_stint_len = 0
    for _, row in driver_df.iterrows():
        if row['StintLength'] < prev_stint_len and prev_stint_len > 3:
            # Tyre life reset — new stint
            if current_stint:
                stints.append(pd.DataFrame(current_stint))
            current_stint = [row]
        else:
            current_stint.append(row)
        prev_stint_len = row['StintLength']

    if current_stint:
        stints.append(pd.DataFrame(current_stint))

    return stints


if __name__ == '__main__':
    # Fit global TrackTemp scaler on training data (2022+2023) so absolute
    # temperature level is preserved across races after per-race normalization.
    # Test split uses the same scaler — no lookahead.
    train_frames = [
        pd.read_parquet(DATA_RAW / 'season_2022.parquet'),
        pd.read_parquet(DATA_RAW / 'season_2023.parquet'),
    ]
    train_all = pd.concat(train_frames, ignore_index=True)
    global_tt_scaler = MinMaxScaler()
    global_tt_scaler.fit(train_all[['TrackTemp']])
    with open(DATA_PROC / 'global_tracktemp_scaler.pkl', 'wb') as f:
        pickle.dump(global_tt_scaler, f)
    print(f"Global TrackTemp scaler saved.")
    print(f"  Range: {global_tt_scaler.data_min_[0]:.1f}°C – {global_tt_scaler.data_max_[0]:.1f}°C")

    # process_split(DATA_RAW / 'season_2023.parquet', 'train')
    # combined 2022 + 2023 for larger training set
    process_split([DATA_RAW / 'season_2022.parquet', DATA_RAW / 'season_2023.parquet'], 'train',
                  global_tt_scaler=global_tt_scaler)
    process_split(DATA_RAW / 'season_2024.parquet', 'test',
                  global_tt_scaler=global_tt_scaler)

    # Delta-target versions (saved with _delta suffix, coexist with absolute)
    process_split([DATA_RAW / 'season_2022.parquet', DATA_RAW / 'season_2023.parquet'], 'train',
                  delta=True, global_tt_scaler=global_tt_scaler)
    process_split(DATA_RAW / 'season_2024.parquet', 'test',
                  delta=True, global_tt_scaler=global_tt_scaler)

    # Split delta tensors by compound for compound-specific model training
    # Compound one-hot indices: SOFT=5, MEDIUM=6, HARD=7 (in FEATURE_COLS)
    COMPOUND_IDX = {'SOFT': 5, 'MEDIUM': 6, 'HARD': 7}
    for split in ('train', 'test'):
        X = torch.load(DATA_PROC / f'X_{split}_delta.pt')
        y = torch.load(DATA_PROC / f'y_{split}_delta.pt')
        for compound, idx in COMPOUND_IDX.items():
            mask = X[:, -1, idx] == 1.0
            torch.save(X[mask], DATA_PROC / f'X_{split}_delta_{compound.lower()}.pt')
            torch.save(y[mask], DATA_PROC / f'y_{split}_delta_{compound.lower()}.pt')
            print(f"  {split} {compound}: {mask.sum().item()} sequences")

    print("\nPreprocessing complete.")
    print(f"Files saved to {DATA_PROC}")