#!/usr/bin/env python3
"""右侧规则的单因子历史回测。

对照当前看板的右侧执行顺序，每次只改一个因子：
比价开关、MACD 开关、日线 BIAS20 阈值、周线均线周期。

信号用当日收盘计算，下一交易日开盘调仓。卖出只作用于已经持有的仓位。
减仓按条件从无到有触发一次，避免同一条件每天重复砍仓。
未出现新动作时保持原仓位。单名目标仓位为试错 15%、确认 45%、重仓 75%，
多名合计超过 100% 时按比例缩到满仓，不使用杠杆。

行情来自 AKShare 新浪日线，不做前复权，与看板一致。
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from curl_cffi import requests as creq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.data import categorize, dedupe_products  # noqa: E402
from app.strategy import analyze_etf, market_regime  # noqa: E402

CACHE = Path(os.environ.get("ETF_BT_CACHE", "/tmp/etf_bt"))
START = "2019-01-01"
WARMUP = "2016-01-01"
COST = 0.0002  # 单边万一，含点差
END_MARK = "2026-09-30"

ACTION_NONE = 0
ACTION_CLEAR = 1
ACTION_TRIM2 = 2
ACTION_TRIM1 = 3
ACTION_HEAVY = 4
ACTION_CONFIRM = 5
ACTION_PROBE = 6
ACTION_WATCH = 7
BUY_ACTIONS = {ACTION_HEAVY, ACTION_CONFIRM, ACTION_PROBE}
TARGETS = {ACTION_HEAVY: 0.75, ACTION_CONFIRM: 0.45, ACTION_PROBE: 0.15}


def _disable_proxy():
    if getattr(requests.Session, "_etf_direct", False):
        return

    def _request(self, method, url, **kwargs):
        return creq.request(
            method=method,
            url=url,
            params=kwargs.get("params"),
            data=kwargs.get("data"),
            headers=kwargs.get("headers"),
            timeout=kwargs.get("timeout") or 30,
            proxies={"http": None, "https": None},
            impersonate="chrome",
        )

    requests.Session.request = _request
    requests.Session._etf_direct = True


def _sina_symbol(code: str) -> str:
    return f"sh{code}" if code.startswith(("5", "6", "9")) else f"sz{code}"


def fetch_bars(code: str) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{code}.csv"
    if path.exists() and path.stat().st_size > 100:
        df = pd.read_csv(path, parse_dates=["date"])
        return df

    _disable_proxy()
    import akshare as ak

    last_error = None
    for attempt in range(4):
        try:
            if code == "000300":
                raw = ak.stock_zh_index_daily(symbol="sh000300")
            else:
                raw = ak.fund_etf_hist_sina(symbol=_sina_symbol(code))
            if raw is None or raw.empty:
                return pd.DataFrame()
            df = raw.rename(columns={"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount"})
            keep = [col for col in ("date", "open", "high", "low", "close", "volume", "amount") if col in df.columns]
            df = df[keep].copy()
            df["date"] = pd.to_datetime(df["date"])
            for col in ("open", "high", "low", "close", "volume"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["date", "open", "high", "low", "close"])
            df = df[df["close"] > 0].sort_values("date").drop_duplicates("date")
            df.to_csv(path, index=False)
            return df
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"{code} 拉取失败: {last_error}")


def load_universe(limit_scan: int = 80, keep: int = 36) -> list[dict]:
    _disable_proxy()
    import akshare as ak

    spot = ak.fund_etf_category_sina(symbol="ETF基金")
    frame = spot.copy()
    frame["代码"] = frame["代码"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    frame["名称"] = frame["名称"].astype(str)
    frame["成交额"] = pd.to_numeric(frame["成交额"], errors="coerce").fillna(0)
    frame["分类"] = frame["名称"].map(categorize)
    picked = frame[(frame["成交额"] >= 20_000_000) & (frame["分类"].isin(["宽基", "行业主题", "商品", "跨境"]))]
    picked = dedupe_products(picked, "名称", "成交额").head(limit_scan)
    meta = []
    for row in picked.itertuples(index=False):
        meta.append({"code": row.代码, "name": row.名称, "category": row.分类, "amount": float(row.成交额)})

    bars = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch_bars, item["code"]): item for item in meta}
        for future in as_completed(futures):
            item = futures[future]
            try:
                bars[item["code"]] = future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"skip {item['code']} {exc}", file=sys.stderr)
                bars[item["code"]] = pd.DataFrame()
    bench = fetch_bars("000300")
    eligible = []
    for item in meta:
        df = bars.get(item["code"], pd.DataFrame())
        if df.empty:
            continue
        if df["date"].iloc[0] <= pd.Timestamp("2018-01-01") and df["date"].iloc[-1] >= pd.Timestamp("2026-09-01"):
            eligible.append(item)
        if len(eligible) >= keep:
            break
    return eligible, bars, bench


def _ema_step(prev, value, alpha):
    if prev is None or (isinstance(prev, float) and np.isnan(prev)):
        return value
    return alpha * value + (1 - alpha) * prev


def weekly_asof(dates: pd.Series, close: np.ndarray, ma_window: int) -> dict[str, np.ndarray]:
    """未完成周用截至当日的收盘充当本周收盘，与看板 resample('W-FRI') 的最后一根一致。"""
    week = pd.to_datetime(dates).dt.to_period("W-FRI").astype(str).to_numpy()
    n = len(close)
    out = {key: np.full(n, np.nan) for key in ("w_close", "w_ma", "w_hist", "w_hist_prev", "w_dif", "w_dea", "w_dif_prev", "w_dea_prev", "w_close_prev", "w_ma_prev")}
    div = np.zeros(n, dtype=bool)
    alpha_f, alpha_s, alpha_d = 2 / 13, 2 / 27, 2 / 10
    ema_f = ema_s = dea = None
    committed_dif = committed_dea = None
    completed_closes: list[float] = []
    completed_hists: list[float] = []
    completed_mas: list[float] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and week[j] == week[i]:
            j += 1
        prev_close = completed_closes[-1] if completed_closes else np.nan
        prev_ma = completed_mas[-1] if completed_mas else np.nan
        prev_hist = completed_hists[-1] if completed_hists else np.nan
        prev_dif = committed_dif if committed_dif is not None else np.nan
        prev_dea = committed_dea if committed_dea is not None else np.nan
        last_c = last_hist = last_dif = last_dea = last_ma = np.nan
        for k in range(i, j):
            c = float(close[k])
            tmp_f = _ema_step(ema_f, c, alpha_f)
            tmp_s = _ema_step(ema_s, c, alpha_s)
            tmp_dif = tmp_f - tmp_s
            tmp_dea = _ema_step(dea, tmp_dif, alpha_d)
            tmp_hist = tmp_dif - tmp_dea
            if len(completed_closes) >= ma_window - 1:
                window = completed_closes[-(ma_window - 1) :] + [c]
                ma = float(np.mean(window))
            else:
                ma = np.nan
            out["w_close"][k] = c
            out["w_ma"][k] = ma
            out["w_hist"][k] = tmp_hist
            out["w_hist_prev"][k] = prev_hist
            out["w_dif"][k] = tmp_dif
            out["w_dea"][k] = tmp_dea
            out["w_dif_prev"][k] = prev_dif
            out["w_dea_prev"][k] = prev_dea
            out["w_close_prev"][k] = prev_close
            out["w_ma_prev"][k] = prev_ma
            series_c = completed_closes + [c]
            series_h = completed_hists + [tmp_hist]
            if len(series_c) >= 16:
                c_now, c_prev = series_c[-8:], series_c[-16:-8]
                h_now, h_prev = series_h[-8:], series_h[-16:-8]
                h_now_max = max(h_now)
                div[k] = max(c_now) > max(c_prev) and h_now_max < max(h_prev) and h_now_max > 0
            last_c, last_hist, last_dif, last_dea, last_ma = c, tmp_hist, tmp_dif, tmp_dea, ma
        ema_f = _ema_step(ema_f, last_c, alpha_f)
        ema_s = _ema_step(ema_s, last_c, alpha_s)
        committed_dif = last_dif
        dea = _ema_step(dea, committed_dif, alpha_d)
        committed_dea = dea
        completed_closes.append(last_c)
        completed_hists.append(last_hist)
        completed_mas.append(last_ma)
        i = j
    out["div_w"] = div
    return out


def _rolling_div(close: np.ndarray, hist: np.ndarray, split: int) -> np.ndarray:
    c = pd.Series(close)
    h = pd.Series(hist)
    now_c = c.rolling(split).max()
    prev_c = now_c.shift(split)
    now_h = h.rolling(split).max()
    prev_h = now_h.shift(split)
    return ((now_c > prev_c) & (now_h < prev_h) & (now_h > 0)).fillna(False).to_numpy()


def build_signals(daily: pd.DataFrame, bench: pd.DataFrame, ma: int, bias_max: float, use_macd: bool, use_rs: bool, use_bias: bool = True) -> pd.DataFrame:
    d = daily.sort_values("date").copy()
    b = bench.sort_values("date")[["date", "close"]].rename(columns={"close": "bench"})
    merged = d.merge(b, on="date", how="left")
    close = merged["close"].to_numpy(float)
    high = merged["high"].to_numpy(float)
    low = merged["low"].to_numpy(float)
    open_ = merged["open"].to_numpy(float)
    volume = merged["volume"].fillna(0).to_numpy(float)
    bench_close = merged["bench"].to_numpy(float)
    dates = pd.to_datetime(merged["date"])
    weekly = weekly_asof(dates, close, ma)

    ma10 = pd.Series(close).rolling(10).mean().to_numpy()
    ma20 = pd.Series(close).rolling(20).mean().to_numpy()
    bias = (close / ma20 - 1.0) * 100.0
    ema_f = pd.Series(close).ewm(span=12, adjust=False).mean()
    ema_s = pd.Series(close).ewm(span=26, adjust=False).mean()
    dif = ema_f - ema_s
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = (dif - dea).to_numpy()
    hist_prev = pd.Series(hist).shift(1).to_numpy()
    vol_base = pd.Series(volume).shift(1).rolling(5).mean().to_numpy()
    vol_ratio = volume / vol_base
    high_20 = pd.Series(high).shift(1).rolling(20).max().to_numpy()
    rs = close / bench_close
    rs_prev = pd.Series(rs).shift(20).to_numpy()
    rs_slope = rs - pd.Series(rs).shift(10).to_numpy()
    rs_ok = (rs > rs_prev) & (rs_slope > 0)
    if not use_rs:
        rs_ok = np.ones(len(close), dtype=bool)

    w_close = weekly["w_close"]
    w_ma = weekly["w_ma"]
    w_hist = weekly["w_hist"]
    w_hist_prev = weekly["w_hist_prev"]
    weekly_above = w_close > w_ma
    lost_weekly = w_close < w_ma
    red_longer = (w_hist > 0) & (w_hist > w_hist_prev)
    red_shorter = (w_hist > 0) & (w_hist < w_hist_prev)
    golden = (weekly["w_dif_prev"] <= weekly["w_dea_prev"]) & (weekly["w_dif"] > weekly["w_dea"])
    just_crossed = (weekly["w_close_prev"] <= weekly["w_ma_prev"]) & weekly_above
    just_lost = (weekly["w_close_prev"] > weekly["w_ma_prev"]) & lost_weekly
    if use_macd:
        trend_ok = weekly_above & (red_longer | golden)
        daily_macd_ok = ((hist > 0) & (hist > hist_prev)) | ((hist < 0) & (hist > hist_prev))
        divergence = _rolling_div(close, hist, 20) | weekly["div_w"]
    else:
        trend_ok = weekly_above
        daily_macd_ok = np.ones(len(close), dtype=bool)
        divergence = np.zeros(len(close), dtype=bool)
        red_shorter = np.zeros(len(close), dtype=bool)

    bias_pullback = (bias <= bias_max) if use_bias else np.ones(len(close), dtype=bool)
    pullback = (low <= ma20 * 1.015) & (close >= ma20 * 0.99)
    breakout = close >= high_20
    volume_ok = np.where(pullback, vol_ratio < 0.8, np.where(breakout, vol_ratio > 1.2, False))
    timing_ok = (bias_pullback & daily_macd_ok) if use_bias else daily_macd_ok
    valid = np.isfinite(w_ma) & np.isfinite(bias) & np.isfinite(vol_ratio) & np.isfinite(bench_close)
    rs_ok = rs_ok & valid
    trend_ok = trend_ok & valid
    timing_ok = timing_ok & valid
    volume_ok = volume_ok & valid
    score = rs_ok.astype(int) + trend_ok.astype(int) + timing_ok.astype(int) + volume_ok.astype(int)
    buy_ready = (score == 4) & valid
    if use_bias:
        heavy = buy_ready & pullback & bias_pullback & (bias >= -3) & ~just_crossed
    else:
        heavy = buy_ready & pullback & ~just_crossed
    weekday = dates.dt.dayofweek.to_numpy()
    broke = (close < ma10) | (close < ma20)
    daily_red_shorter = (hist > 0) & (hist < hist_prev) if use_macd else np.zeros(len(close), dtype=bool)

    clear = just_lost | (divergence & (weekly_above | just_lost))
    trim2 = weekly_above & broke & np.isfinite(bias) & (bias <= 2)
    trim1 = weekly_above & (red_shorter | daily_red_shorter)
    confirm = buy_ready & (weekday == 4)
    probe = buy_ready & np.isin(weekday, [2, 3]) & just_crossed
    watch = score >= 3
    action = np.select(
        [clear, trim2, trim1, heavy, confirm, probe, watch],
        [ACTION_CLEAR, ACTION_TRIM2, ACTION_TRIM1, ACTION_HEAVY, ACTION_CONFIRM, ACTION_PROBE, ACTION_WATCH],
        default=ACTION_NONE,
    )
    action = np.where(valid, action, ACTION_NONE)
    return pd.DataFrame({
        "date": dates.to_numpy(),
        "open": open_,
        "close": close,
        "action": action.astype(np.int8),
    })


def run_portfolio(panels: dict[str, pd.DataFrame], master: pd.DatetimeIndex) -> dict:
    codes = list(panels)
    aligned = {}
    for code, frame in panels.items():
        f = frame.set_index("date").reindex(master)
        f["action"] = f["action"].ffill().fillna(ACTION_NONE)
        f["open"] = f["open"].ffill()
        aligned[code] = f

    units = {code: 0.0 for code in codes}
    prev_action = {code: ACTION_NONE for code in codes}
    prev_w = {code: 0.0 for code in codes}
    prev_full = {code: 0.0 for code in codes}
    turnover_full = 0.0
    sleeves = {code: {"shares": 0.0, "invested": 0.0, "realized": 0.0, "entry": None} for code in codes}
    equity = 1.0
    equity_full = 1.0
    curve = []
    curve_full = []
    trades = []
    entries = 0
    buy_days = 0
    turnover = 0.0
    dates = list(master)
    opens = {code: aligned[code]["open"].to_numpy(float) for code in codes}
    actions = {code: aligned[code]["action"].to_numpy(int) for code in codes}

    def mark_trade(code, day, price, new_units):
        nonlocal entries
        sleeve = sleeves[code]
        old = units[code]
        if new_units > old + 1e-12:
            if old <= 1e-12:
                entries += 1
                sleeve["entry"] = day
            buy = new_units - old
            sleeve["shares"] += buy
            sleeve["invested"] += buy * price
        elif new_units < old - 1e-12:
            sell = old - new_units
            sleeve["realized"] += sell * price
            sleeve["shares"] = max(0.0, sleeve["shares"] - sell)
            if new_units <= 1e-12:
                pnl = sleeve["realized"] - sleeve["invested"]
                trades.append({
                    "code": code,
                    "pnl": pnl,
                    "ret": pnl / sleeve["invested"] if sleeve["invested"] else 0.0,
                    "entry": sleeve["entry"],
                    "exit": day,
                })
                sleeve.update(shares=0.0, invested=0.0, realized=0.0, entry=None)
        units[code] = new_units

    for i in range(len(dates) - 2):
        if dates[i] < pd.Timestamp(START):
            continue
        raw = {}
        for code in codes:
            action = int(actions[code][i])
            price = opens[code][i + 1]
            if not np.isfinite(price) or price <= 0:
                raw[code] = 0.0
                prev_action[code] = action
                continue
            current = units[code]
            if action in BUY_ACTIONS:
                buy_days += 1
            if action == ACTION_CLEAR:
                target = 0.0
            elif action == ACTION_TRIM2 and prev_action[code] != ACTION_TRIM2 and current > 0:
                target = current * 0.65
            elif action == ACTION_TRIM1 and prev_action[code] != ACTION_TRIM1 and current > 0:
                target = current * 0.75
            elif action in BUY_ACTIONS:
                target = max(current, TARGETS[action])
            else:
                target = current
            mark_trade(code, dates[i + 1], price, target)
            raw[code] = units[code]
            prev_action[code] = action
        total = sum(raw.values())
        scale = 1.0 if total <= 1 else 1.0 / total
        weights = {code: raw[code] * scale for code in codes}
        full_scale = (0.75 / total) if total > 1e-12 else 0.0
        weights_full = {code: raw[code] * full_scale for code in codes}
        traded_value = sum(weights.values())
        step_ret = 0.0
        step_full = 0.0
        turn = 0.0
        turn_full = 0.0
        for code in codes:
            p0 = opens[code][i + 1]
            p1 = opens[code][i + 2]
            if not np.isfinite(p0) or not np.isfinite(p1) or p0 <= 0:
                ret = 0.0
            else:
                ret = p1 / p0 - 1.0
            step_ret += weights[code] * ret
            step_full += weights_full[code] * ret
            turn += abs(weights[code] - prev_w[code])
            turn_full += abs(weights_full[code] - prev_full[code])
        equity *= 1 + step_ret - turn * COST
        equity_full *= 1 + step_full - turn_full * COST
        turnover += turn
        turnover_full += turn_full
        prev_w = weights
        prev_full = weights_full
        curve.append((dates[i + 1], equity, traded_value))
        curve_full.append(equity_full)

    # 期末仍持有的回合按最后开盘价结算，计入胜率
    if len(dates) >= 2:
        last_day = dates[-1]
        for code in codes:
            price = opens[code][-1]
            if units[code] > 1e-12 and np.isfinite(price):
                mark_trade(code, last_day, price, 0.0)

    eq = np.array([row[1] for row in curve])
    if len(eq) < 2:
        return {}
    years = (curve[-1][0] - curve[0][0]).days / 365.25
    peak = np.maximum.accumulate(eq)
    maxdd = float((eq / peak - 1).min())
    closed = trades
    wins = [t for t in closed if t["pnl"] > 0]
    rets = [t["ret"] for t in closed]
    hold_days = []
    for t in closed:
        if t["entry"] is not None and t["exit"] is not None:
            hold_days.append((pd.Timestamp(t["exit"]) - pd.Timestamp(t["entry"])).days)
    eq_full = np.array(curve_full)
    peak_full = np.maximum.accumulate(eq_full)
    return {
        "cagr": float(eq[-1] ** (1 / years) - 1) if years > 0 else None,
        "maxdd": maxdd,
        "cagr_full": float(eq_full[-1] ** (1 / years) - 1) if years > 0 else None,
        "maxdd_full": float((eq_full / peak_full - 1).min()),
        "turnover_full": (turnover_full / 2) / years if years else None,
        "win_rate": float(len(wins) / len(closed)) if closed else None,
        "avg_trade": float(np.mean(rets)) if rets else None,
        "median_trade": float(np.median(rets)) if rets else None,
        "trades": len(closed),
        "entries": entries,
        "buy_days": buy_days,
        "entries_per_year": entries / years if years else None,
        "buy_days_per_year": buy_days / years if years else None,
        "turnover": (turnover / 2) / years if years else None,
        "avg_hold_days": float(np.mean(hold_days)) if hold_days else None,
        "avg_exposure": float(np.mean([row[2] for row in curve])),
        "end": float(eq[-1]),
        "years": years,
        "start": str(curve[0][0].date()),
        "finish": str(curve[-1][0].date()),
    }


def buy_hold(bench: pd.DataFrame, master: pd.DatetimeIndex) -> dict:
    f = bench.sort_values("date").set_index("date").reindex(master)
    opens = f["open"].ffill().to_numpy(float)
    dates = list(master)
    equity = 1.0
    curve = []
    for i in range(len(dates) - 2):
        if dates[i] < pd.Timestamp(START):
            continue
        p0, p1 = opens[i + 1], opens[i + 2]
        if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
            equity *= p1 / p0
        curve.append(equity)
    eq = np.array(curve)
    years = (dates[-2] - pd.Timestamp(START)).days / 365.25
    peak = np.maximum.accumulate(eq)
    return {"cagr": float(eq[-1] ** (1 / years) - 1), "maxdd": float((eq / peak - 1).min()), "end": float(eq[-1])}


def verify_last_day(code: str, name: str, category: str, daily: pd.DataFrame, bench: pd.DataFrame):
    sig = build_signals(daily, bench, 20, 1.0, True, True)
    raw = daily.rename(columns={}).copy()
    raw["date"] = pd.to_datetime(raw["date"])
    b = bench.copy()
    b["date"] = pd.to_datetime(b["date"])
    regime = market_regime(b)
    row = analyze_etf(code, name, category, raw, b, regime)
    mapping = {
        "清仓": ACTION_CLEAR,
        "减仓二档": ACTION_TRIM2,
        "减仓一档": ACTION_TRIM1,
        "回踩重仓": ACTION_HEAVY,
        "确认加仓": ACTION_CONFIRM,
        "试错买入": ACTION_PROBE,
        "观察": ACTION_WATCH,
        "无信号": ACTION_NONE,
    }
    got = int(sig["action"].iloc[-1])
    expect = mapping[row["action_right"]]
    return {"code": code, "name": name, "as_of": row["as_of"], "expect": row["action_right"], "match": got == expect, "got": got}


def pct(value):
    if value is None:
        return None
    return round(value * 100, 2)


def main():
    universe, bars, bench = load_universe()
    bench = bench[bench["date"] >= WARMUP].copy()
    master = pd.DatetimeIndex(sorted(bench.loc[bench["date"] >= "2018-06-01", "date"]))
    variants = [
        ("基准：比价+MACD+BIAS1%+周MA20", dict(ma=20, bias_max=1.0, use_macd=True, use_rs=True)),
        ("去掉比价", dict(ma=20, bias_max=1.0, use_macd=True, use_rs=False)),
        ("去掉MACD", dict(ma=20, bias_max=1.0, use_macd=False, use_rs=True)),
        ("BIAS上限3%", dict(ma=20, bias_max=3.0, use_macd=True, use_rs=True)),
        ("BIAS上限5%", dict(ma=20, bias_max=5.0, use_macd=True, use_rs=True)),
        ("周线MA10", dict(ma=10, bias_max=1.0, use_macd=True, use_rs=True)),
        ("周线MA30", dict(ma=30, bias_max=1.0, use_macd=True, use_rs=True)),
    ]
    groups = {
        "全池": universe,
        "仅行业主题": [item for item in universe if item["category"] == "行业主题"],
    }
    checks = []
    for item in universe[:3]:
        checks.append(verify_last_day(item["code"], item["name"], item["category"], bars[item["code"]], bench))

    report = {"universe": [{k: item[k] for k in ("code", "name", "category")} for item in universe], "checks": checks, "benchmark": buy_hold(bench, master), "groups": {}}
    for group_name, members in groups.items():
        rows = []
        for label, params in variants:
            panels = {}
            for item in members:
                daily = bars[item["code"]]
                daily = daily[daily["date"] >= WARMUP]
                panels[item["code"]] = build_signals(daily, bench, **params)
            stats = run_portfolio(panels, master)
            stats["label"] = label
            stats["names"] = len(members)
            rows.append(stats)
            print(group_name, label, {k: stats.get(k) for k in ("cagr", "maxdd", "win_rate", "entries_per_year", "buy_days_per_year", "turnover")}, flush=True)
        report["groups"][group_name] = rows
    out = CACHE / "factor_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, default=str, indent=2))
    print(f"wrote {out}")
    print("checks", checks)
    print("benchmark", report["benchmark"])


if __name__ == "__main__":
    main()
