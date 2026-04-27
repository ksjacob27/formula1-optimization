import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
import json
import time

from models.lstm import get_model

DATA_PROC   = Path('data/processed')
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

# Training config
CONFIG = {
    'batch_size':   32,
    'epochs':       50,
    'learning_rate': 1e-3,
    'hidden_size':  64,
    'num_layers':   2,
    'dropout':      0.2,
    'val_split':    0.1,
    'patience':     7,       # early stopping
}


# Feature index map (must match FEATURE_COLS in data/preprocessing.py)
FEATURE_INDEX = {
    'LapTimeSeconds': 0, 'StintLength': 1, 'FuelLoad': 2,
    'AirTemp': 3, 'TrackTemp': 4,
    'Compound_SOFT': 5, 'Compound_MEDIUM': 6, 'Compound_HARD': 7,
    'TrackTemp_global': 8,
}
INPUT_SIZE = len(FEATURE_INDEX)  # 9


def load_data(
    device: torch.device,
    delta: bool = False,
    mask_features: list[str] = None,
    compound: str = None,          # 'SOFT', 'MEDIUM', or 'HARD' for compound-specific files
) -> tuple:
    if compound:
        # Compound-specific delta tensors (always delta; created by splitting X_train_delta.pt)
        suffix = f'_delta_{compound.lower()}'
    else:
        suffix = '_delta' if delta else ''
    X_train = torch.load(DATA_PROC / f'X_train{suffix}.pt').to(device)
    y_train = torch.load(DATA_PROC / f'y_train{suffix}.pt').to(device)
    X_test  = torch.load(DATA_PROC / f'X_test{suffix}.pt').to(device)
    y_test  = torch.load(DATA_PROC / f'y_test{suffix}.pt').to(device)

    # Ablation: zero out specified feature columns at both train and test time
    if mask_features:
        for fname in mask_features:
            idx = FEATURE_INDEX[fname]
            X_train[:, :, idx] = 0.0
            X_test[:,  :, idx] = 0.0

    return X_train, y_train, X_test, y_test


def train_model(
    model_type: str,
    config: dict,
    device: torch.device,
    run_name: str = None,
    delta: bool = False,
    mask_features: list[str] = None,
    compound: str = None,          # 'SOFT', 'MEDIUM', or 'HARD'
) -> dict:

    name = run_name or model_type

    compound_str = f', compound: {compound}' if compound else ''
    print(f"\n{'='*50}")
    print(f"Training {name.upper()} ({'delta' if delta else 'absolute'} target"
          f"{', masked: ' + ','.join(mask_features) if mask_features else ''}{compound_str})")
    print(f"{'='*50}")

    # Load data
    X_train_full, y_train_full, X_test, y_test = load_data(
        device, delta=delta, mask_features=mask_features, compound=compound
    )

    # Train/val split
    dataset    = TensorDataset(X_train_full, y_train_full)
    val_size   = int(len(dataset) * config['val_split'])
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'])

    # Model, optimizer, loss
    model = get_model(
        model_type,
        input_size=config.get('input_size', INPUT_SIZE),
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    # Training loop
    best_val_loss  = float('inf')
    patience_count = 0
    history        = {'train_loss': [], 'val_loss': []}

    for epoch in range(config['epochs']):
        start = time.time()

        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= train_size

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item() * len(X_batch)
        val_loss /= val_size

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        elapsed = time.time() - start
        print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"{elapsed:.1f}s")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(),
                       RESULTS_DIR / f'best_{name}.pt')
        else:
            patience_count += 1
            if patience_count >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Evaluate on test set
    model.load_state_dict(torch.load(RESULTS_DIR / f'best_{name}.pt'))
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test)
        test_mae   = torch.mean(torch.abs(test_preds - y_test)).item()
        test_mse   = torch.mean((test_preds - y_test) ** 2).item()

    print(f"\n{name.upper()} Test MAE: {test_mae:.6f}")
    print(f"{name.upper()} Test MSE: {test_mse:.6f}")

    results = {
        'model_type':   model_type,
        'run_name':     name,
        'test_mae':     test_mae,
        'test_mse':     test_mse,
        'best_val_loss': best_val_loss,
        'history':      history,
        'config':       config
    }

    # Save results
    with open(RESULTS_DIR / f'results_{name}.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def moving_average_baseline(device: torch.device, delta: bool = False) -> float:
    """
    Naive baseline for absolute mode: predict next lap time = mean of last
    SEQUENCE_LENGTH lap times in the input window.

    For delta mode: predict the mean of the lap-to-lap deltas observed in
    the input window (recent trend extrapolation). This is equivalent to
    predicting that degradation continues at the same rate as the window average.
    Predicting zero (no change) would be the simplest delta baseline, but mean
    delta is a fairer comparison since it uses the same input information.
    """
    _, _, X_test, y_test = load_data(device, delta=delta)

    # LapTimeSeconds is feature index 0 in all modes
    if delta:
        # Mean delta across the 9 consecutive pairs in the 10-lap window
        window_laps = X_test[:, :, 0]  # (n, 10)
        window_deltas = window_laps[:, 1:] - window_laps[:, :-1]  # (n, 9)
        preds = window_deltas.mean(dim=1)
        label = "Mean-Delta Baseline"
    else:
        preds = X_test[:, :, 0].mean(dim=1)
        label = "Moving Average Baseline"

    mae = torch.mean(torch.abs(preds - y_test)).item()
    print(f"\n{label} MAE: {mae:.6f}")
    return mae


# if __name__ == '__main__':
#     device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     # Baseline first
#     baseline_mae = moving_average_baseline(device)

#     # Train both models
#     lstm_results = train_model('lstm', CONFIG, device)
#     gru_results  = train_model('gru',  CONFIG, device)

#     # Summary
#     print(f"\n{'='*50}")
#     print(f"SUMMARY")
#     print(f"{'='*50}")
#     print(f"Baseline MAE:  {baseline_mae:.6f}")
#     print(f"LSTM MAE:      {lstm_results['test_mae']:.6f}")
#     print(f"GRU MAE:       {gru_results['test_mae']:.6f}")

#     lstm_improvement = (baseline_mae - lstm_results['test_mae']) / baseline_mae * 100
#     gru_improvement  = (baseline_mae - gru_results['test_mae'])  / baseline_mae * 100
#     print(f"LSTM improvement over baseline: {lstm_improvement:.1f}%")
#     print(f"GRU improvement over baseline:  {gru_improvement:.1f}%")

#     CONFIG_LSTM_HIGH_DROPOUT = {**CONFIG, 'num_layers': 2, 'dropout': 0.4}
#     CONFIG_LSTM_SMALL = {**CONFIG, 'num_layers': 2, 'hidden_size': 32}

# if __name__ == '__main__':  # Run 1–3 (absolute target)
#     device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#     print(f"Using device: {device}")
#     baseline_mae = moving_average_baseline(device)
#     EXPERIMENTS = {
#         'lstm_baseline':     ('lstm',           {**CONFIG, 'hidden_size': 64,  'num_layers': 2, 'dropout': 0.2}),
#         'lstm_large':        ('lstm',           {**CONFIG, 'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2}),
#         'lstm_deep':         ('lstm',           {**CONFIG, 'hidden_size': 64,  'num_layers': 3, 'dropout': 0.3}),
#         'lstm_high_dropout': ('lstm',           {**CONFIG, 'hidden_size': 64,  'num_layers': 2, 'dropout': 0.4}),
#         'lstm_attention':    ('lstm_attention', {**CONFIG, 'hidden_size': 64,  'num_layers': 2, 'dropout': 0.2}),
#         'gru_baseline':      ('gru',            {**CONFIG, 'hidden_size': 64,  'num_layers': 2, 'dropout': 0.2}),
#     }
#     all_results = {}
#     for run_name, (model_type, cfg) in EXPERIMENTS.items():
#         all_results[run_name] = train_model(model_type, cfg, device, run_name=run_name)
#     print(f"\n{'='*50}")
#     print(f"SUMMARY")
#     print(f"{'='*50}")
#     print(f"{'Run':<22} {'MAE':>10} {'Improvement':>14}")
#     print(f"{'-'*48}")
#     print(f"{'Baseline (moving avg)':<22} {baseline_mae:>10.6f} {'—':>14}")
#     for run_name, results in all_results.items():
#         mae = results['test_mae']
#         improvement = (baseline_mae - mae) / baseline_mae * 100
#         print(f"{run_name:<22} {mae:>10.6f} {improvement:>+13.1f}%")

# if __name__ == '__main__':  # Run 4 — delta target
#     device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#     print(f"Using device: {device}")
#     baseline_mae = moving_average_baseline(device, delta=True)
#     EXPERIMENTS = {
#         'lstm_baseline_delta':     ('lstm',           {**CONFIG, 'hidden_size': 64,  'num_layers': 2, 'dropout': 0.2}),
#         'lstm_high_dropout_delta': ('lstm',           {**CONFIG, 'hidden_size': 64,  'num_layers': 2, 'dropout': 0.4}),
#         'gru_baseline_delta':      ('gru',            {**CONFIG, 'hidden_size': 64,  'num_layers': 2, 'dropout': 0.2}),
#     }
#     all_results = {}
#     for run_name, (model_type, cfg) in EXPERIMENTS.items():
#         all_results[run_name] = train_model(
#             model_type, cfg, device, run_name=run_name, delta=True
#         )
#     print(f"\n{'='*50}")
#     print(f"SUMMARY — Run 4 (delta target)")
#     print(f"{'='*50}")
#     print(f"{'Run':<28} {'MAE':>10} {'Improvement':>14}")
#     print(f"{'-'*54}")
#     print(f"{'Baseline (mean delta)':<28} {baseline_mae:>10.6f} {'—':>14}")
#     for run_name, results in all_results.items():
#         mae = results['test_mae']
#         improvement = (baseline_mae - mae) / baseline_mae * 100
#         print(f"{run_name:<28} {mae:>10.6f} {improvement:>+13.1f}%")

# if __name__ == '__main__':  # Run 5 — TrackTemp ablation (delta target)
#     device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#     print(f"Using device: {device}")
#     baseline_mae = moving_average_baseline(device, delta=True)
#     cfg = {**CONFIG, 'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2}
#     results = train_model(
#         'gru', cfg, device,
#         run_name='gru_baseline_delta_no_tracktemp',
#         delta=True,
#         mask_features=['TrackTemp']
#     )
#     print(f"\n{'='*50}")
#     print(f"SUMMARY — Run 5 (TrackTemp ablation)")
#     print(f"{'='*50}")
#     print(f"Baseline (mean delta):                      {baseline_mae:.6f}")
#     print(f"gru_baseline_delta (with TrackTemp):        0.036866   [from Run 4]")
#     print(f"gru_baseline_delta_no_tracktemp:            {results['test_mae']:.6f}")
#     diff = results['test_mae'] - 0.036866
#     print(f"Δ MAE (no_tracktemp − with_tracktemp):      {diff:+.6f}")

if False:  # Run 6 — dual TrackTemp normalization (delta target, 9 features)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    baseline_mae = moving_average_baseline(device, delta=True)

    # Same GRU delta architecture, now with 9-feature input: existing per-race
    # normalized TrackTemp (index 4) + new globally normalized TrackTemp_global
    # (index 8). The global feature preserves absolute temperature level that
    # per-race MinMaxScaler otherwise destroys. Hypothesis: improved strategy
    # accuracy at temperature-extreme circuits (Las Vegas, Singapore, Bahrain).
    cfg = {**CONFIG, 'input_size': INPUT_SIZE, 'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2}
    results = train_model(
        'gru', cfg, device,
        run_name='gru_delta_global_tracktemp',
        delta=True,
    )

    print(f"\n{'='*50}")
    print(f"SUMMARY — Run 6 (dual TrackTemp normalization)")
    print(f"{'='*50}")
    print(f"Baseline (mean delta):                         {baseline_mae:.6f}")
    print(f"gru_baseline_delta (9 feat, global TrackTemp): {results['test_mae']:.6f}")
    diff = results['test_mae'] - 0.036866
    print(f"Δ MAE vs gru_baseline_delta (Run 4):           {diff:+.6f}")

if __name__ == '__main__':  # Phase 2 — compound-specific GRU models (delta target)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    baseline_mae = moving_average_baseline(device, delta=True)

    COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']
    cfg = {**CONFIG, 'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2}

    all_results = {}
    for compound in COMPOUNDS:
        run_name = f'gru_delta_{compound.lower()}'
        all_results[compound] = train_model(
            'gru', cfg, device,
            run_name=run_name,
            compound=compound,
        )

    print(f"\n{'='*50}")
    print(f"SUMMARY — Phase 2 (compound-specific GRU)")
    print(f"{'='*50}")
    print(f"{'Model':<24} {'MAE':>10} {'vs baseline':>14}")
    print(f"{'-'*50}")
    for compound, res in all_results.items():
        mae = res['test_mae']
        diff = mae - baseline_mae
        print(f"gru_delta_{compound.lower():<12} {mae:>10.6f} {diff:>+13.6f}")