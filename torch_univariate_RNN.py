import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch_models import BaseRnn

class UnivariateModel(BaseRnn):
    """PyTorch implementation of the univariate RNN model that only looks at GDP series."""
    
    def __init__(
        self,
        seq_len: int,
        n_monthly: int,
        n_quarterly: int,
        hidden: int,
        dropout: float,
        Rnn: type[nn.RNNBase] = nn.LSTM,
    ) -> None:
        super().__init__()
        self.rnn = Rnn(1, hidden, batch_first=True)  # Only input is GDP (1 feature)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(
        self,
        x_monthly: torch.Tensor,   # (B, Q, n_monthly) - not used
        x_quarterly: torch.Tensor, # (B, Q, 1) - only GDP
        x_shift: torch.Tensor,     # (B, Q, 1) - not used
    ) -> torch.Tensor:             # (B, 1)
        # Only use quarterly data (GDP)
        x = x_quarterly
        
        # Process through RNN layer
        x, _ = self.rnn(x)
        
        # Get the last output from RNN sequence
        x = x[:, -1, :]
        
        # Apply dropout and fully connected layer
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


def create_datapoints_univariate(monthly, quarterly, sequence_length):
    sequence_length_quarterly = sequence_length // 3

    x_monthly = []
    x_quarterly = []
    x_shift = []
    y = []
    y_dates = []

    for q in range(len(quarterly) - sequence_length_quarterly): # -1 to account for the next quarter
        q_end = q + sequence_length_quarterly

        x_monthly.append(
            np.ones((sequence_length // 3, 1))
        )
        x_quarterly.append(
            quarterly.iloc[q:q_end]["GDPC1"].values.reshape(-1, 1)
        )
        x_shift.append(
            np.arange(0, sequence_length // 3).reshape(-1, 1)
        )
        y.append(
            quarterly.iloc[q_end][["GDPC1"]]
        )
        y_dates.append(
            quarterly.index[q_end]
        )

    return [np.asarray(d, dtype=np.float32) for d in [x_monthly, x_quarterly, x_shift, y]] + [y_dates]
