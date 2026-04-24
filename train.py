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


def load_data(device: torch.device) -> tuple:
    X_train = torch.load(DATA_PROC / 'X_train.pt').to(device)
    y_train = torch.load(DATA_PROC / 'y_train.pt').to(device)
    X_test  = torch.load(DATA_PROC / 'X_test.pt').to(device)
    y_test  = torch.load(DATA_PROC / 'y_test.pt').to(device)
    return X_train, y_train, X_test, y_test


def train_model(
    model_type: str,
    config: dict,
    device: torch.device
) -> dict:

    print(f"\n{'='*50}")
    print(f"Training {model_type.upper()}")
    print(f"{'='*50}")

    # Load data
    X_train_full, y_train_full, X_test, y_test = load_data(device)

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
                       RESULTS_DIR / f'best_{model_type}.pt')
        else:
            patience_count += 1
            if patience_count >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Evaluate on test set
    model.load_state_dict(torch.load(RESULTS_DIR / f'best_{model_type}.pt'))
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test)
        test_mae   = torch.mean(torch.abs(test_preds - y_test)).item()
        test_mse   = torch.mean((test_preds - y_test) ** 2).item()

    print(f"\n{model_type.upper()} Test MAE: {test_mae:.6f}")
    print(f"{model_type.upper()} Test MSE: {test_mse:.6f}")

    results = {
        'model_type':   model_type,
        'test_mae':     test_mae,
        'test_mse':     test_mse,
        'best_val_loss': best_val_loss,
        'history':      history,
        'config':       config
    }

    # Save results
    with open(RESULTS_DIR / f'results_{model_type}.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def moving_average_baseline(device: torch.device) -> float:
    """
    Naive baseline: predict next lap time as the mean of the
    last SEQUENCE_LENGTH lap times in the input window.
    """
    _, _, X_test, y_test = load_data(device)

    # LapTimeSeconds is the first feature (index 0)
    preds = X_test[:, :, 0].mean(dim=1)
    mae   = torch.mean(torch.abs(preds - y_test)).item()
    print(f"\nMoving Average Baseline MAE: {mae:.6f}")
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

if __name__ == '__main__':
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Baseline first
    baseline_mae = moving_average_baseline(device)

    # Train original models
    lstm_results = train_model('lstm', CONFIG, device)
    gru_results  = train_model('gru',  CONFIG, device)

    # LSTM experiments
    CONFIG_LSTM_HIGH_DROPOUT = {**CONFIG, 'num_layers': 2, 'dropout': 0.4}
    CONFIG_LSTM_SMALL        = {**CONFIG, 'num_layers': 2, 'hidden_size': 32}

    lstm_dropout_results = train_model('lstm', CONFIG_LSTM_HIGH_DROPOUT, device)
    lstm_small_results   = train_model('lstm', CONFIG_LSTM_SMALL, device)

    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Baseline MAE:             {baseline_mae:.6f}")
    print(f"LSTM MAE:                 {lstm_results['test_mae']:.6f}")
    print(f"GRU MAE:                  {gru_results['test_mae']:.6f}")
    print(f"LSTM high dropout MAE:    {lstm_dropout_results['test_mae']:.6f}")
    print(f"LSTM small hidden MAE:    {lstm_small_results['test_mae']:.6f}")

    for name, mae in [
        ('LSTM', lstm_results['test_mae']),
        ('GRU', gru_results['test_mae']),
        ('LSTM high dropout', lstm_dropout_results['test_mae']),
        ('LSTM small hidden', lstm_small_results['test_mae']),
    ]:
        improvement = (baseline_mae - mae) / baseline_mae * 100
        print(f"{name} improvement over baseline: {improvement:.1f}%")