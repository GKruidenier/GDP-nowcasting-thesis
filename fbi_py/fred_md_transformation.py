import pandas as pd
import numpy as np

def transxf(x, tcode):
    n = len(x)
    small = 1e-6
    y = pd.Series(index=x.index, dtype=float)

    if tcode == 1:
        y = x
    elif tcode == 2:
        y.iloc[1:] = x.values[1:] - x.values[:-1]
    elif tcode == 3:
        y.iloc[2:] = x.values[2:] - 2 * x.values[1:-1] + x.values[:-2]
    elif tcode == 4:
        if (x > small).all():
            y = np.log(x)
    elif tcode == 5:
        if (x > small).all():
            x_log = np.log(x)
            y.iloc[1:] = x_log.values[1:] - x_log.values[:-1]
    elif tcode == 6:
        if (x > small).all():
            x_log = np.log(x)
            y.iloc[2:] = x_log.values[2:] - 2 * x_log.values[1:-1] + x_log.values[:-2]
    elif tcode == 7:
        y1 = pd.Series(index=x.index, dtype=float)
        y1.iloc[1:] = (x.values[1:] - x.values[:-1]) / x.values[:-1]
        y.iloc[2:] = y1.values[2:] - y1.values[1:-1]
    return y

def fredmd(file, date_start=None, date_end=None, transform=True):
    rawdata = pd.read_csv(file, skiprows=2, header=None, parse_dates=[0])
    attrdata = pd.read_csv(file, nrows=2, header=None)

    headers = ['date'] + attrdata.iloc[0, 1:].astype(str).tolist()
    tcodes = list(map(int, attrdata.iloc[1, 1:].tolist()))
    rawdata.columns = headers

    rawdata.replace([np.inf, -np.inf], np.nan, inplace=True)
    rawdata.dropna(how='all', inplace=True)

    if transform:
        data = rawdata.copy()
        for i, col in enumerate(data.columns[1:], start=0):
            data[col] = transxf(rawdata[col], tcodes[i])
    else:
        data = rawdata.copy()

    data['date'] = pd.to_datetime(data['date'])
    if date_start:
        data = data[data['date'] >= pd.to_datetime(date_start)]
    if date_end:
        data = data[data['date'] <= pd.to_datetime(date_end)]

    data.set_index('date', inplace=True)
    return data
