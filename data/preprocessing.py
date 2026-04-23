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
    'Compound_SOFT', 'Compound_MEDIUM', 'Compound_HARD'
]
TARGET_COL = 'LapTimeSeconds'


def build_sequences(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding window sequences from a single driver stint.
    Returns X of shape (n_samples, sequence_length, n_features)
    and y of shape (n_samples,)
    """
    values = group[FEATURE_COLS].values
    target = group[TARGET_COL].values

    X, y = [], []
    for i in range(len(values) - SEQUENCE_LENGTH):
        X.append(values[i:i + SEQUENCE_LENGTH])
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


def process_split(parquet_path: Path, split_name: str) -> None:
    """
    Load a season parquet, build sequences per driver per stint,
    normalize per race, and save tensors.
    """
    df = pd.read_parquet(parquet_path)
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
                X, y = build_sequences(stint_df)
                all_X.append(X)
                all_y.append(y)

    X_all = np.concatenate(all_X, axis=0).astype(np.float32)
    y_all = np.concatenate(all_y, axis=0).astype(np.float32)

    X_tensor = torch.tensor(X_all, dtype=torch.float32)
    y_tensor = torch.tensor(y_all, dtype=torch.float32)

    torch.save(X_tensor, DATA_PROC / f'X_{split_name}.pt')
    torch.save(y_tensor, DATA_PROC / f'y_{split_name}.pt')

    print(f"{split_name}: {X_tensor.shape[0]} sequences, "
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
    process_split(DATA_RAW / 'season_2023.parquet', 'train')
    process_split(DATA_RAW / 'season_2024.parquet', 'test')
    print("\nPreprocessing complete.")
    print(f"Files saved to {DATA_PROC}")