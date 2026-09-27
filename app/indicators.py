"""日线 / 周线指标。只比较方向和相对位置，不依赖行情软件的柱状缩放。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_daily(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "amount" in out.columns:
        out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[out["close"] > 0]
    return out.reset_index(drop=True)


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.set_index("date")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if "amount" in d.columns:
        agg["amount"] = "sum"
    weekly = d.resample("W-FRI").agg(agg).dropna(subset=["close"])
    weekly = weekly[weekly["close"] > 0]
    return weekly.reset_index()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def bias(close: pd.Series, window: int) -> pd.Series:
    ma = sma(close, window)
    return (close - ma) / ma * 100.0


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = dif - dea
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": hist})


def volume_ratio(volume: pd.Series, window: int = 5) -> pd.Series:
    base = volume.shift(1).rolling(window, min_periods=window).mean()
    return volume / base.replace(0, np.nan)


def add_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    d = prepare_daily(daily)
    d["ma10"] = sma(d["close"], 10)
    d["ma20"] = sma(d["close"], 20)
    d["ma60"] = sma(d["close"], 60)
    d["ma120"] = sma(d["close"], 120)
    d["ma250"] = sma(d["close"], 250)
    d["bias6"] = bias(d["close"], 6)
    d["bias20"] = bias(d["close"], 20)
    macd_df = macd(d["close"])
    d["dif"] = macd_df["dif"]
    d["dea"] = macd_df["dea"]
    d["hist"] = macd_df["hist"]
    d["vol_ratio"] = volume_ratio(d["volume"].fillna(0))
    return d


def add_weekly_features(weekly: pd.DataFrame) -> pd.DataFrame:
    w = weekly.copy()
    w["ma20"] = sma(w["close"], 20)
    w["ma60"] = sma(w["close"], 60)
    w["ma120"] = sma(w["close"], 120)
    w["ma250"] = sma(w["close"], 250)
    w["bias6"] = bias(w["close"], 6)
    macd_df = macd(w["close"])
    w["dif"] = macd_df["dif"]
    w["dea"] = macd_df["dea"]
    w["hist"] = macd_df["hist"]
    w["vol_ratio"] = volume_ratio(w["volume"].fillna(0))
    return w


def last_number(series: pd.Series, offset: int = 1):
    if series is None or len(series) < offset:
        return None
    value = series.iloc[-offset]
    if pd.isna(value):
        return None
    return float(value)


def near_level(price: float, level: float | None, pct: float = 0.035) -> bool:
    if level is None or level <= 0 or price <= 0:
        return False
    # 站在支撑附近：没有跌破 3%，也没有远离到支撑上方 3.5% 以外。
    return level * (1 - pct) <= price <= level * (1 + pct)


def volume_node(daily: pd.DataFrame, bins: int = 36) -> float | None:
    """用近 250 日成交密集区近似筹码峰。"""
    window = daily.tail(250)
    if len(window) < 40:
        return None
    typical = (window["high"] + window["low"] + window["close"]) / 3.0
    weights = window["volume"].fillna(0).clip(lower=0)
    if float(weights.sum()) <= 0:
        return None
    low = float(typical.min())
    high = float(typical.max())
    if high <= low:
        return None
    cuts = np.linspace(low, high, bins + 1)
    idx = np.digitize(typical.to_numpy(), cuts[1:-1])
    volumes = np.zeros(bins)
    for i, vol in zip(idx, weights.to_numpy()):
        volumes[min(int(i), bins - 1)] += float(vol)
    peak = int(np.argmax(volumes))
    return float((cuts[peak] + cuts[peak + 1]) / 2.0)


def swing_lows(weekly: pd.DataFrame, lookback: int = 6, count: int = 3) -> list[float]:
    lows = weekly["low"].to_numpy()
    found: list[float] = []
    # 只确认已经走完右侧的摆动低点，避免用未来数据。
    for i in range(lookback, len(lows) - lookback):
        window = lows[i - lookback : i + lookback + 1]
        if lows[i] <= np.nanmin(window):
            found.append(float(lows[i]))
    return found[-count:]
