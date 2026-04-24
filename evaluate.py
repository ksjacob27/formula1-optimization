import torch
import torch.nn as nn
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from models.lstm import get_model

DATA_PROC   = Path('data/processed')
RESULTS_DIR = Path('results')

def load_results(model_type: str) -> dict:
    with open(RESULTS_DIR / f'results_{model_type}.json') as f:
        return json.load(f)


def load_best_model(model_type: str, device: torch.device) -> nn.Module:
    model = get_model(model_type).to(device)
    model.load_state_dict(torch.load(
        RESULTS_DIR / f'best_{model_type}.pt',
        map_location=device
    ))
    model.eval()
    return model


def plot_training_curves(lstm_results: dict, gru_results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, results in zip(axes, [lstm_results, gru_results]):
        model_type = results['model_type'].upper()
        ax.plot(results['history']['train_loss'], label='Train Loss', linewidth=1.5)
        ax.plot(results['history']['val_loss'],   label='Val Loss',   linewidth=1.5)
        ax.set_title(f'{model_type} Training Curves')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.legend()

    plt.suptitle('Training and Validation Loss')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'training_curves.png', dpi=150)
    plt.show()
    print("Saved training_curves.png")


def plot_predictions(model_type: str, device: torch.device, n_samples: int = 200) -> None:
    model    = load_best_model(model_type, device)
    X_test   = torch.load(DATA_PROC / 'X_test.pt').to(device)
    y_test   = torch.load(DATA_PROC / 'y_test.pt').to(device)

    with torch.no_grad():
        preds = model(X_test[:n_samples])
        actual = y_test[:n_samples]

    preds  = preds.cpu().numpy()
    actual = actual.cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Predicted vs actual over samples
    axes[0].plot(actual, label='Actual',    linewidth=1.2, alpha=0.8)
    axes[0].plot(preds,  label='Predicted', linewidth=1.2, alpha=0.8)
    axes[0].set_title(f'{model_type.upper()} — Predicted vs Actual Lap Times (first {n_samples} samples)')
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Normalized Lap Time')
    axes[0].legend()

    # Scatter plot
    axes[1].scatter(actual, preds, alpha=0.3, s=10)
    min_val = min(actual.min(), preds.min())
    max_val = max(actual.max(), preds.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Perfect prediction')
    axes[1].set_title(f'{model_type.upper()} — Scatter: Predicted vs Actual')
    axes[1].set_xlabel('Actual Lap Time (normalized)')
    axes[1].set_ylabel('Predicted Lap Time (normalized)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'predictions_{model_type}.png', dpi=150)
    plt.show()
    print(f"Saved predictions_{model_type}.png")


def plot_mae_comparison(baseline_mae: float, lstm_results: dict, gru_results: dict) -> None:
    models = ['Baseline', 'LSTM', 'GRU']
    maes   = [
        baseline_mae,
        lstm_results['test_mae'],
        gru_results['test_mae']
    ]
    colors = ['#888888', '#4C72B0', '#DD8452']
    improvements = [
        0,
        (baseline_mae - lstm_results['test_mae']) / baseline_mae * 100,
        (baseline_mae - gru_results['test_mae'])  / baseline_mae * 100,
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, maes, color=colors, edgecolor='white', width=0.5)

    # Annotate bars with MAE and improvement
    for bar, mae, imp in zip(bars, maes, improvements):
        label = f'{mae:.4f}'
        if imp != 0:
            label += f'\n({imp:+.1f}%)'
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                label, ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('MAE (normalized)')
    ax.set_title('Model Comparison — Test MAE')
    ax.set_ylim(0, max(maes) * 1.2)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'mae_comparison.png', dpi=150)
    plt.show()
    print("Saved mae_comparison.png")


def print_summary(baseline_mae: float, lstm_results: dict, gru_results: dict) -> None:
    print(f"\n{'='*50}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"{'Model':<20} {'MAE':>10} {'Improvement':>15}")
    print(f"{'-'*45}")
    print(f"{'Baseline':<20} {baseline_mae:>10.6f} {'—':>15}")
    for results in [lstm_results, gru_results]:
        name = results['model_type'].upper()
        mae  = results['test_mae']
        imp  = (baseline_mae - mae) / baseline_mae * 100
        print(f"{name:<20} {mae:>10.6f} {imp:>+14.1f}%")
    print(f"{'='*50}")

    # Check against success criteria
    best_improvement = max(
        (baseline_mae - lstm_results['test_mae']) / baseline_mae * 100,
        (baseline_mae - gru_results['test_mae'])  / baseline_mae * 100
    )
    print(f"\nPrimary success criterion (>10% MAE improvement):")
    if best_improvement >= 10:
        print(f"  ✓ PASSED — best model achieves {best_improvement:.1f}% improvement")
    else:
        print(f"  ✗ FAILED — best improvement is only {best_improvement:.1f}%")


if __name__ == '__main__':
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    lstm_results = load_results('lstm')
    gru_results  = load_results('gru')
    baseline_mae = 0.095744  # from train.py output

    print_summary(baseline_mae, lstm_results, gru_results)
    plot_training_curves(lstm_results, gru_results)
    plot_predictions('gru', device)
    plot_mae_comparison(baseline_mae, lstm_results, gru_results)