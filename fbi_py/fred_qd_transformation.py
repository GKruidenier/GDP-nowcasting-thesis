import pandas as pd
import numpy as np
from typing import Union, Optional

def transform_series(x: pd.Series, tcode: int) -> pd.Series:
    """Apply transformation based on FRED-QD tcode."""
    n = len(x)
    small = 1e-6
    y = pd.Series(index=x.index, dtype=float)

    if tcode == 1:
        y = x
    elif tcode == 2:
        y.iloc[1:] = x.values[1:] - x.values[:-1]
    elif tcode == 3:
        y.iloc[2:] = x.values[2:] - 2 * x.values[1:-1] + x.values[:-2]
    elif tcode == 4 and x.min(skipna=True) > small:
        y = np.log(x)
    elif tcode == 5 and x.min(skipna=True) > small:
        logx = np.log(x)
        y.iloc[1:] = logx.values[1:] - logx.values[:-1]
    elif tcode == 6 and x.min(skipna=True) > small:
        logx = np.log(x)
        y.iloc[2:] = logx.values[2:] - 2 * logx.values[1:-1] + logx.values[:-2]
    elif tcode == 7:
        y1 = pd.Series(index=x.index, dtype=float)
        y1.iloc[1:] = (x.values[1:] - x.values[:-1]) / x.values[:-1]
        y.iloc[2:] = y1.values[2:] - y1.values[1:-1]

    return y

def fredqd(file: str, date_start: Optional[str] = None, date_end: Optional[str] = None, transform: bool = True) -> pd.DataFrame:
    """Load and optionally transform FRED-QD dataset."""

    # Load full dataset and header info
    rawdata = pd.read_csv(file, skiprows=3, header=None, parse_dates=[0])
    attrdata = pd.read_csv(file, nrows=3, header=None)

    # Remove last NA rows if all non-finite
    for idx in range(len(rawdata) - 20, len(rawdata)):
        if not np.isfinite(rawdata.iloc[idx, 1:]).any():
            rawdata.drop(index=idx, inplace=True)

    # Set column headers
    headers = ['date'] + attrdata.iloc[0, 1:].tolist()
    tcodes = list(map(int, attrdata.iloc[2, 1:].tolist()))
    rawdata.columns = headers

    # Optionally transform data
    if transform:
        for i, col in enumerate(headers[1:]):
            rawdata[col] = transform_series(rawdata[col], tcodes[i])

    # Filter by date if needed
    rawdata['date'] = pd.to_datetime(rawdata['date'])
    if date_start:
        rawdata = rawdata[rawdata['date'] >= pd.to_datetime(date_start)]
    if date_end:
        rawdata = rawdata[rawdata['date'] <= pd.to_datetime(date_end)]

    rawdata.set_index('date', inplace=True)
    return rawdata
