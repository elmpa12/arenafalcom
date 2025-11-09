# regime_gates.py
import numpy as np
import pandas as pd

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n=60) -> pd.Series:
  prev = close.shift(1)
  tr = np.maximum(high-low, np.maximum((high-prev).abs(), (low-prev).abs()))
  return pd.Series(tr).rolling(n, min_periods=n).mean()

def vol_gate(df_1m: pd.DataFrame, min_pct=0.0006) -> bool:
  a60 = atr(df_1m['high'], df_1m['low'], df_1m['close'], n=60).iloc[-1]
  price = df_1m['close'].iloc[-1]
  return (a60/price) >= min_pct

def impulse_ok(last_candle: dict, atr_val: float, k=0.6) -> bool:
  body = abs(last_candle['close'] - last_candle['open'])
  return body >= k * atr_val
