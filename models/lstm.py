import torch
import torch.nn as nn


class TireLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(TireLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, sequence_length, input_size)
        out, _ = self.lstm(x)

        # Take only the last timestep output
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)

        return out.squeeze(-1)


class TireGRU(nn.Module):
    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(TireGRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, sequence_length, input_size)
        out, _ = self.gru(x)

        # Take only the last timestep output
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)

        return out.squeeze(-1)


class TireLSTMAttention(nn.Module):
    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(TireLSTMAttention, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        # scores each timestep — softmax gives attention weights
        self.attention = nn.Linear(hidden_size, 1)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)                          # (batch, seq_len, hidden)

        scores  = self.attention(out)                  # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)         # (batch, seq_len, 1)
        context = (weights * out).sum(dim=1)           # (batch, hidden)

        context = self.dropout(context)
        return self.fc(context).squeeze(-1)


def get_model(model_type: str, **kwargs) -> nn.Module:
    """
    Factory function to instantiate a model by name.
    Usage: model = get_model('lstm') or get_model('gru')
    """
    models = {
        'lstm':          TireLSTM,
        'gru':           TireGRU,
        'lstm_attention': TireLSTMAttention,
    }
    if model_type not in models:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from {list(models.keys())}")

    return models[model_type](**kwargs)


if __name__ == '__main__':
    # Sanity check both models
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    dummy_input = torch.randn(32, 10, 8).to(device)  # batch=32, seq=10, features=8

    for model_type in ['lstm', 'gru']:
        model = get_model(model_type).to(device)
        output = model(dummy_input)
        print(f"{model_type.upper()} output shape: {output.shape}")  # should be (32,)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"{model_type.upper()} parameters: {total_params:,}")