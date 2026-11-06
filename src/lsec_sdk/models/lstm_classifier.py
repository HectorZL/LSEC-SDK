"""Recurrent neural network models for sign language sequence classification."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Computes a learned attention weight for each time step in a sequence."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (batch, time_steps, hidden_dim)
            mask: Optional binary mask (batch, time_steps)
        Returns:
            context: Aggregated tensor of shape (batch, hidden_dim)
            weights: Attention weights of shape (batch, time_steps, 1)
        """
        scores = self.attn(x)  # (batch, time_steps, 1)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        weights = F.softmax(scores, dim=1)
        context = torch.sum(x * weights, dim=1)  # (batch, hidden_dim)
        return context, weights


class SignLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM with input projection and temporal attention pooling
    for sign language sequence recognition.
    """

    def __init__(
        self,
        input_dim: int = 411,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # Input feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        # Recurrent backbone
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Temporal pooling
        if use_attention:
            self.pool = TemporalAttention(lstm_out_dim)
        else:
            self.pool = None

        # Output classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Tensor of shape (batch, time_steps, input_dim)
               or flattened from (batch, time_steps, 137, 3)
            mask: Optional boolean mask (batch, time_steps)
        Returns:
            logits: (batch, num_classes)
            attention_weights: Optional (batch, time_steps, 1)
        """
        if x.dim() == 4:
            # Flatten (batch, time, 137, 3) -> (batch, time, 411)
            b, t, k, c = x.shape
            x = x.view(b, t, k * c)

        proj = self.input_proj(x)
        lstm_out, _ = self.lstm(proj)  # (batch, time, lstm_out_dim)

        weights = None
        if self.use_attention and self.pool is not None:
            pooled, weights = self.pool(lstm_out, mask)
        else:
            # Mean pooling over time dimension
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                pooled = (lstm_out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)
            else:
                pooled = lstm_out.mean(dim=1)

        logits = self.classifier(pooled)
        return logits, weights


class SignGRUClassifier(nn.Module):
    """Lightweight Bidirectional GRU for ultra-fast edge inference."""

    def __init__(
        self,
        input_dim: int = 411,
        hidden_dim: int = 96,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.pool = TemporalAttention(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if x.dim() == 4:
            b, t, k, c = x.shape
            x = x.view(b, t, k * c)
        proj = self.input_proj(x)
        gru_out, _ = self.gru(proj)
        pooled, weights = self.pool(gru_out, mask)
        logits = self.classifier(pooled)
        return logits, weights

