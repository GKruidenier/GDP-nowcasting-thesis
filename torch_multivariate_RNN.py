import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch_models import BaseRnn
from feature_selection import map_month_to_quarter
from functools import partial

class MultivariateModel(BaseRnn):
    """PyTorch implementation of the multivariate RNN model."""
    
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
        self.rnn = Rnn(n_monthly + n_quarterly, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(
        self,
        x_monthly: torch.Tensor,   # (B, Q, n_monthly)
        x_quarterly: torch.Tensor, # (B, Q, n_quarterly)
        x_shift: torch.Tensor,     # (B, Q, 1)
    ) -> torch.Tensor:             # (B, 1)
        B = x_monthly.shape[0]
        Q = x_monthly.shape[1]
        n_monthly = x_monthly.shape[2]
        n_quarterly = x_quarterly.shape[2]
        
        # Shape assertions for input tensors
        assert x_monthly.shape == (B, Q, n_monthly), f"Expected x_monthly shape (B, Q, n_monthly) = ({B}, {Q}, {n_monthly}), got {x_monthly.shape}"
        assert x_quarterly.shape == (B, Q, n_quarterly), f"Expected x_quarterly shape (B, Q, n_quarterly) = ({B}, {Q}, {n_quarterly}), got {x_quarterly.shape}"
        assert x_shift.shape == (B, Q, 1), f"Expected x_shift shape (B, Q, 1) = ({B}, {Q}, 1), got {x_shift.shape}"

        # Concatenate monthly and quarterly features
        x = torch.cat((x_monthly, x_quarterly), dim=-1)
        
        # Process through RNN layer
        x, _ = self.rnn(x)
        
        # Get the last output from RNN sequence
        x = x[:, -1, :]
        
        # Apply dropout and fully connected layer
        x = self.dropout(x)
        x = self.fc(x)

        assert x.shape == (B, 1), f"Expected output shape (B, 1) = ({B}, 1), got {x.shape}"
        
        return x


def create_datapoints_multivariate(monthly, quarterly, sequence_length):
    sequence_length_quarterly = sequence_length // 3

    x_monthly = []
    x_quarterly = []
    x_shift = []
    y = []
    y_dates = []

    for q in range(len(quarterly) - sequence_length_quarterly): # -1 to account for the next quarter
        m = q * 3
        m_end = m + sequence_length
        q_end = q + sequence_length_quarterly

        all_monthly = monthly.iloc[m:m_end]
        quarter = all_monthly.index.map(partial(map_month_to_quarter, lookahead_months=0))

        aggregated_monthly = all_monthly.groupby(quarter).mean()

        x_monthly.append(
            aggregated_monthly.values
        )
        x_quarterly.append(
            quarterly.iloc[q:q_end].values
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
