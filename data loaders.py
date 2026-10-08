"""
Real-data loaders for the three pairs: NASDAQ100, XAUUSD, EURUSD.
Harmonizes each source's format into a standard OHLCV DataFrame with a
'datetime' column and lowercase open/high/low/close/volume columns, sorted
ascending. EURUSD daily is derived by resampling the M15 file to daily bars
(the uploaded EURUSD_D1.csv had no price data).
"""

import pandas as pd


def load_ndx_daily():
    df = pd.read_csv('/mnt/user-data/uploads/1d_data.csv', sep='\t')
    df.columns = [c.strip() for c in df.columns]
    df['datetime'] = pd.to_datetime(df['DateTime'], format='%Y.%m.%d %H:%M:%S')
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].sort_values('datetime').reset_index(drop=True)
    df = df.drop_duplicates(subset='datetime')
    return df


def load_ndx_m15():
    df = pd.read_csv('/mnt/user-data/uploads/15m_data.csv', sep='\t')
    df.columns = [c.strip() for c in df.columns]
    df['datetime'] = pd.to_datetime(df['DateTime'], format='%Y.%m.%d %H:%M:%S')
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].sort_values('datetime').reset_index(drop=True)
    df = df.drop_duplicates(subset='datetime')
    return df


def load_xau_daily():
    df = pd.read_csv('/mnt/user-data/uploads/XAU_1d_data.csv', sep=';')
    df.columns = [c.strip() for c in df.columns]
    df['datetime'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].sort_values('datetime').reset_index(drop=True)
    df = df.drop_duplicates(subset='datetime')
    return df


def load_xau_m15():
    df = pd.read_csv('/mnt/user-data/uploads/XAU_15m_data.csv', sep=';')
    df.columns = [c.strip() for c in df.columns]
    df['datetime'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].sort_values('datetime').reset_index(drop=True)
    df = df.drop_duplicates(subset='datetime')
    return df


def load_eur_m15():
    df = pd.read_csv('/mnt/user-data/uploads/EURUSD_M15_200001030000_201912312245.csv', sep='\t')
    df.columns = [c.strip().strip('<>') for c in df.columns]
    df['datetime'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], format='%Y.%m.%d %H:%M:%S')
    df = df.rename(columns={'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 'CLOSE': 'close', 'TICKVOL': 'volume'})
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].sort_values('datetime').reset_index(drop=True)
    df = df.drop_duplicates(subset='datetime')
    return df


def load_eur_daily_from_m15():
    """EURUSD_D1.csv had no price data -- derive daily OHLC by resampling
    the M15 series instead."""
    m15 = load_eur_m15()
    m15 = m15.set_index('datetime')
    daily = m15.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    daily = daily.dropna(subset=['open', 'close']).reset_index()
    return daily


if __name__ == "__main__":
    for name, loader in [
        ("NASDAQ100 Daily", load_ndx_daily),
        ("NASDAQ100 M15", load_ndx_m15),
        ("XAUUSD Daily", load_xau_daily),
        ("XAUUSD M15", load_xau_m15),
        ("EURUSD M15", load_eur_m15),
        ("EURUSD Daily (derived)", load_eur_daily_from_m15),
    ]:
        df = loader()
        print(f"{name}: {len(df)} rows, {df['datetime'].min()} to {df['datetime'].max()}, "
              f"price range {df['close'].min():.2f}-{df['close'].max():.2f}")
