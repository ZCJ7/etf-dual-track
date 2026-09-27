import numpy as np
import pandas as pd

from app.strategy import analyze_etf, market_regime


def _frame(close, volume=None):
    close = np.asarray(close, dtype=float)
    n = len(close)
    dates = pd.bdate_range("2022-01-03", periods=n)
    vol = np.full(n, 1_000_000.0) if volume is None else np.asarray(volume, dtype=float)
    return pd.DataFrame({
        "date": dates,
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": vol,
    })


def test_uptrend_keeps_right_side_available():
    close = 3000 + np.arange(520) * 1.2
    regime = market_regime(_frame(close))
    assert regime["above_ma60"] is True
    assert regime["above_ma20"] is True
    assert regime["right_mode"] == "按ETF"
    assert regime["left_mode"] == "按ETF"


def test_breakdown_does_not_use_ma60_as_a_ban():
    close = np.concatenate([
        4000 + np.arange(400) * 0.4,
        4160 - np.arange(160) * 8,
    ])
    regime = market_regime(_frame(close))
    assert regime["above_ma60"] is False
    assert regime["right_mode"] == "按ETF"


def test_same_underlying_keeps_one_name():
    from app.data import product_key

    assert product_key("中证1000ETF南方") == product_key("中证1000ETF华夏")
    assert product_key("半导体ETF国联安") != product_key("科创半导体ETF华夏")


def test_analyze_returns_both_tracks():
    bench = _frame(3000 + np.arange(420) * 1.0)
    etf_close = 2 + np.arange(420) * 0.004
    regime = market_regime(bench)
    row = analyze_etf("510300", "沪深300ETF", "宽基", _frame(etf_close), bench, regime)
    assert row["code"] == "510300"
    assert len(row["checks_right"]) == 4
    assert len(row["checks_left"]) == 6
    assert row["action_right"]
    assert row["action_left"]
