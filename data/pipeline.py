import fastf1
import pandas as pd
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / 'cache'
DATA_DIR  = Path(__file__).resolve().parent.parent / 'data' / 'raw'

CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))

FUEL_LOAD_KG   = 110.0
FUEL_BURN_RATE = 1.6


def load_race(year: int, round_number: int) -> pd.DataFrame:
    """Load and return cleaned lap data for a single race."""
    session = fastf1.get_session(year, round_number, 'R')
    session.load(telemetry=False, weather=True, messages=False)

    laps = session.laps.pick_quicklaps()

    # Filter to dry compounds only (scope of project)
    laps = laps[laps['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])]

    # Merge nearest weather reading onto each lap
    weather = session.weather_data.sort_values('Time')
    laps = laps.sort_values('LapStartTime')
    laps = pd.merge_asof(
        laps,
        weather[['Time', 'AirTemp', 'TrackTemp']],
        left_on='LapStartTime',
        right_on='Time',
        direction='nearest'
    )

    # Feature engineering
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    laps['FuelLoad']       = (FUEL_LOAD_KG - laps['LapNumber'] * FUEL_BURN_RATE).clip(lower=0)
    laps['StintLength']    = laps['TyreLife']
    laps['Year']           = year
    laps['Round']          = round_number

    # One-hot encode compound
    compound_dummies = pd.get_dummies(laps['Compound'], prefix='Compound')
    laps = pd.concat([laps, compound_dummies], axis=1)

    # Guard: ensure all compound columns exist even if not used in this race
    for col in ['Compound_SOFT', 'Compound_MEDIUM', 'Compound_HARD']:
        if col not in laps.columns:
            laps[col] = 0
        else:
            laps[col] = laps[col].astype(int)

    feature_cols = [
        'Year', 'Round', 'Driver', 'LapNumber',
        'LapTimeSeconds', 'StintLength', 'FuelLoad',
        'AirTemp', 'TrackTemp',
        'Compound_SOFT', 'Compound_MEDIUM', 'Compound_HARD'
    ]

    return laps[feature_cols].dropna()


def build_season(year: int, max_rounds: int = 22) -> pd.DataFrame:
    """Load all races for a season and return as a single DataFrame."""
    all_races = []

    for rnd in range(1, max_rounds + 1):
        try:
            df = load_race(year, rnd)
            all_races.append(df)
            print(f"  ✓ {year} round {rnd:02d} — {len(df)} laps")
        except Exception as e:
            print(f"  ✗ {year} round {rnd:02d} — skipped ({e})")

    return pd.concat(all_races, ignore_index=True)


def save_season(year: int, max_rounds: int = 22) -> None:
    """Build and save a season dataset to parquet."""
    print(f"\nBuilding {year} season...")
    df = build_season(year, max_rounds)
    out_path = DATA_DIR / f'season_{year}.parquet'
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} laps → {out_path}")


if __name__ == '__main__':
    save_season(2022)
    save_season(2023)
    save_season(2024, max_rounds=5)


