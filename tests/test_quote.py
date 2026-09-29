import numpy as np
import pandas as pd

from app.data import apply_quote
from app.strategy import analyze_etf, market_regime


def _bars(close, start="2026-09-28"):
    close = np.atleast_1d(close).astype(float)
    dates = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.full(len(close), 1_000_000.0),
    })


def test_apply_quote_appends_the_update_price_on_a_weekday():
    out = apply_quote(_bars([1.0]), {
        "quoted_on": "2026-09-29",
        "price": 1.08,
        "open": 1.01,
        "high": 1.09,
        "low": 1.0,
        "volume": 12_000,
    })
    assert list(out["date"]) == ["2026-09-28", "2026-09-29"]
    assert out.iloc[-1]["close"] == 1.08
    assert out.iloc[-1]["volume"] == 1_200_000


def test_apply_quote_replaces_the_same_day_close():
    out = apply_quote(_bars([1.0], start="2026-09-29"), {
        "quoted_on": "2026-09-29",
        "price": 1.2,
        "high": 1.25,
        "low": 0.95,
    })
    assert len(out) == 1
    assert out.iloc[-1]["close"] == 1.2
    assert out.iloc[-1]["high"] >= 1.25


def test_apply_quote_does_not_invent_a_weekend_bar():
    out = apply_quote(_bars([1.0], start="2026-09-25"), {
        "quoted_on": "2026-09-26",
        "price": 9,
    })
    assert len(out) == 1
    assert out.iloc[-1]["close"] == 1.0


def test_signal_close_follows_the_update_price():
    close = 2 + np.arange(420) * 0.001
    bars = _bars(close, start="2024-01-02")
    nxt = (pd.Timestamp(bars["date"].iloc[-1]) + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    price = float(close[-1]) * 1.04
    overlaid = apply_quote(bars, {"quoted_on": nxt, "price": price, "volume": 1_000_000})
    bench = _bars(3000 + np.arange(420), start="2024-01-02")
    regime = market_regime(bench)
    row = analyze_etf("510300", "沪深300ETF", "宽基", overlaid, bench, regime)
    assert row["as_of"] == nxt
    assert row["close"] == round(price, 3)
