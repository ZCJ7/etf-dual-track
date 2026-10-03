#!/usr/bin/env python3
"""左右侧一起回测。

主方案去掉两类入场限制：
- 右侧不再要求相对沪深300的比价向上；
- 右侧不再要求日线 BIAS20 ≤ 1%，回踩重仓也不再卡 BIAS 区间；
- 左侧不再要求周线 BIAS6 < -4%。

减仓二档仍保留「跌破日线均线且 BIAS20 ≤ 2%」。其余 MACD、周线 MA20、量能、支撑、
左侧 MA60 拐头否决都保留。右侧合计仓位不超过 75%，左侧不超过 25%。
同一只 ETF 同时只在一侧持仓；左侧「卖半转右侧」把剩余仓位交给右侧规则管理。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.strategy import analyze_etf, market_regime  # noqa: E402
from backtest.factor_compare import (  # noqa: E402
    ACTION_CLEAR,
    ACTION_CONFIRM,
    ACTION_HEAVY,
    ACTION_NONE,
    ACTION_PROBE,
    ACTION_TRIM1,
    ACTION_TRIM2,
    ACTION_WATCH,
    BUY_ACTIONS,
    CACHE,
    COST,
    START,
    TARGETS,
    WARMUP,
    _ema_step,
    _rolling_div,
    build_signals,
    buy_hold,
    fetch_bars,
    load_universe,
)

L_NONE = 0
L_CLEAR = 1
L_HALF_RIGHT = 2
L_HALF_FRI = 3
L_WAIT = 4
L_PAUSE = 5
L_SECOND = 6
L_FIRST = 7
L_WATCH = 8
LEFT_BUYS = {L_SECOND, L_FIRST}
LEFT_TARGETS = {L_SECOND: 0.25, L_FIRST: 0.125}
RIGHT_CAP = 0.75
LEFT_CAP = 0.25


def _ma(completed: list[float], current: float, window: int) -> float:
    if len(completed) < window - 1:
        return np.nan
    return float((sum(completed[-(window - 1) :]) + current) / window)


def weekly_pack(dates: pd.Series, ohlcv: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """周线截至当日。MA20 供右侧和左侧位置判断，另算 MA6/60/120/250、周量比和支撑。"""
    close = ohlcv["close"]
    low = ohlcv["low"]
    volume = ohlcv["volume"]
    week = pd.to_datetime(dates).dt.to_period("W-FRI").astype(str).to_numpy()
    n = len(close)
    keys = (
        "w_close", "w_ma20", "w_ma6", "w_ma60", "w_ma120", "w_ma250",
        "w_hist", "w_hist_prev", "w_dif", "w_dea", "w_dif_prev", "w_dea_prev",
        "w_close_prev", "w_ma20_prev", "w_ma60_lag", "w_vol", "w_major",
        "swing1", "swing2", "swing3",
    )
    out = {key: np.full(n, np.nan) for key in keys}
    div = np.zeros(n, dtype=bool)
    alpha_f, alpha_s, alpha_d = 2 / 13, 2 / 27, 2 / 10
    ema_f = ema_s = dea = None
    committed_dif = committed_dea = None
    completed_closes: list[float] = []
    completed_hists: list[float] = []
    completed_ma20: list[float] = []
    completed_ma60: list[float] = []
    completed_lows: list[float] = []
    completed_vols: list[float] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and week[j] == week[i]:
            j += 1
        prev_close = completed_closes[-1] if completed_closes else np.nan
        prev_ma20 = completed_ma20[-1] if completed_ma20 else np.nan
        prev_hist = completed_hists[-1] if completed_hists else np.nan
        prev_dif = committed_dif if committed_dif is not None else np.nan
        prev_dea = committed_dea if committed_dea is not None else np.nan
        ma60_lag = completed_ma60[-4] if len(completed_ma60) >= 4 else np.nan
        hist_low = min(completed_lows) if completed_lows else np.inf
        swings = _confirmed_swings(completed_lows)
        week_low = np.inf
        week_vol = 0.0
        last_c = last_hist = last_dif = last_ma20 = last_ma60 = np.nan
        for k in range(i, j):
            c = float(close[k])
            week_low = min(week_low, float(low[k]))
            week_vol += float(volume[k])
            tmp_f = _ema_step(ema_f, c, alpha_f)
            tmp_s = _ema_step(ema_s, c, alpha_s)
            tmp_dif = tmp_f - tmp_s
            tmp_dea = _ema_step(dea, tmp_dif, alpha_d)
            tmp_hist = tmp_dif - tmp_dea
            ma20 = _ma(completed_closes, c, 20)
            ma60 = _ma(completed_closes, c, 60)
            out["w_close"][k] = c
            out["w_ma20"][k] = ma20
            out["w_ma6"][k] = _ma(completed_closes, c, 6)
            out["w_ma60"][k] = ma60
            out["w_ma120"][k] = _ma(completed_closes, c, 120)
            out["w_ma250"][k] = _ma(completed_closes, c, 250)
            out["w_hist"][k] = tmp_hist
            out["w_hist_prev"][k] = prev_hist
            out["w_dif"][k] = tmp_dif
            out["w_dea"][k] = tmp_dea
            out["w_dif_prev"][k] = prev_dif
            out["w_dea_prev"][k] = prev_dea
            out["w_close_prev"][k] = prev_close
            out["w_ma20_prev"][k] = prev_ma20
            out["w_ma60_lag"][k] = ma60_lag
            if len(completed_vols) >= 5 and week_vol > 0:
                out["w_vol"][k] = week_vol / (sum(completed_vols[-5:]) / 5.0)
            if len(completed_lows) + 1 > 30:
                out["w_major"][k] = min(hist_low, week_low)
            for slot, level in zip(("swing1", "swing2", "swing3"), swings):
                out[slot][k] = level
            series_c = completed_closes + [c]
            series_h = completed_hists + [tmp_hist]
            if len(series_c) >= 16:
                h_now = series_h[-8:]
                h_now_max = max(h_now)
                div[k] = max(series_c[-8:]) > max(series_c[-16:-8]) and h_now_max < max(series_h[-16:-8]) and h_now_max > 0
            last_c, last_hist, last_dif, last_ma20, last_ma60 = c, tmp_hist, tmp_dif, ma20, ma60
        ema_f = _ema_step(ema_f, last_c, alpha_f)
        ema_s = _ema_step(ema_s, last_c, alpha_s)
        committed_dif = last_dif
        dea = _ema_step(dea, committed_dif, alpha_d)
        committed_dea = dea
        completed_closes.append(last_c)
        completed_hists.append(last_hist)
        completed_ma20.append(last_ma20)
        completed_ma60.append(last_ma60)
        completed_lows.append(week_low if np.isfinite(week_low) else last_c)
        completed_vols.append(week_vol)
        i = j
    out["div_w"] = div
    return out


def _confirmed_swings(lows: list[float], lookback: int = 6, count: int = 3) -> list[float]:
    found: list[float] = []
    for i in range(lookback, len(lows) - lookback):
        window = lows[i - lookback : i + lookback + 1]
        if lows[i] <= min(window):
            found.append(lows[i])
    return found[-count:]


def _volume_nodes(high, low, close, volume, window: int = 250, bins: int = 36) -> np.ndarray:
    typical = (high + low + close) / 3.0
    volume = np.nan_to_num(volume, nan=0.0)
    n = len(typical)
    out = np.full(n, np.nan)
    for i in range(40, n):
        sl = slice(i - window + 1, i + 1) if i + 1 >= window else slice(0, i + 1)
        t = typical[sl]
        w = volume[sl]
        if len(t) < 40 or w.sum() <= 0:
            continue
        lo = float(np.min(t))
        hi = float(np.max(t))
        if hi <= lo:
            continue
        cuts = np.linspace(lo, hi, bins + 1)
        idx = np.digitize(t, cuts[1:-1])
        volumes = np.zeros(bins)
        np.add.at(volumes, np.clip(idx, 0, bins - 1), w)
        peak = int(np.argmax(volumes))
        out[i] = (cuts[peak] + cuts[peak + 1]) / 2.0
    return out


def _near(price: np.ndarray, level: np.ndarray) -> np.ndarray:
    ok = np.isfinite(level) & (level > 0) & (price > 0)
    return ok & (price >= level * 0.965) & (price <= level * 1.035)


def build_left_signals(daily: pd.DataFrame, pack: dict[str, np.ndarray], use_bias: bool) -> np.ndarray:
    close = daily["close"].to_numpy(float)
    high = daily["high"].to_numpy(float)
    low = daily["low"].to_numpy(float)
    volume = daily["volume"].fillna(0).to_numpy(float)
    dates = pd.to_datetime(daily["date"])
    ma120 = pd.Series(close).rolling(120).mean().to_numpy()
    ma250 = pd.Series(close).rolling(250).mean().to_numpy()
    ema_f = pd.Series(close).ewm(span=12, adjust=False).mean()
    ema_s = pd.Series(close).ewm(span=26, adjust=False).mean()
    hist = (ema_f - ema_s)
    dea = hist.ewm(span=9, adjust=False).mean()
    hist = (hist - dea).to_numpy()
    hist_prev = pd.Series(hist).shift(1).to_numpy()
    hist_prev2 = pd.Series(hist).shift(2).to_numpy()
    vol_ratio = volume / pd.Series(volume).shift(1).rolling(5).mean().to_numpy()
    nodes = _volume_nodes(high, low, close, volume)
    price = close
    support = (
        _near(price, pack["w_ma60"])
        | _near(price, pack["w_ma120"])
        | _near(price, pack["w_ma250"])
        | _near(price, ma120)
        | _near(price, ma250)
        | _near(price, pack["w_major"])
        | _near(price, pack["swing1"])
        | _near(price, pack["swing2"])
        | _near(price, pack["swing3"])
        | _near(price, nodes)
    )
    w_close = pack["w_close"]
    w_ma20 = pack["w_ma20"]
    w_ma60 = pack["w_ma60"]
    own_below = np.isfinite(w_ma20) & (w_close < w_ma20)
    structure = np.isfinite(w_ma60) & (~np.isfinite(pack["w_ma60_lag"]) | (w_ma60 >= pack["w_ma60_lag"]))
    if use_bias:
        space = np.isfinite(pack["w_ma6"]) & ((w_close / pack["w_ma6"] - 1.0) * 100.0 < -4)
    else:
        space = np.ones(len(close), dtype=bool)
    green_shorter = (hist < 0) & (hist > hist_prev)
    shrink = np.isfinite(vol_ratio) & (vol_ratio < 0.8)
    volume_ok = ~shrink
    prev_low = pd.Series(low).shift(1).to_numpy()
    prev_green = (hist_prev < 0) & (hist_prev > hist_prev2)
    second = green_shorter & prev_green & np.isfinite(prev_low) & (low >= prev_low)
    stood = np.isfinite(w_ma20) & (price >= w_ma20)
    touched = np.isfinite(w_ma20) & (high >= w_ma20 * 0.995)
    w_vol = pack["w_vol"]
    weekday = dates.dt.dayofweek.to_numpy()
    core = own_below & support & space & green_shorter & structure
    left_buy = core & volume_ok
    score = own_below.astype(int) + support.astype(int) + space.astype(int) + green_shorter.astype(int) + volume_ok.astype(int) + structure.astype(int)
    action = np.select(
        [
            stood & np.isfinite(w_vol) & (w_vol > 1.5),
            stood & np.isfinite(w_vol) & (w_vol >= 1.0) & (w_vol <= 1.5),
            touched & np.isfinite(w_vol) & (w_vol < 0.8) & ~stood,
            touched & (weekday < 4) & ~stood,
            touched & (weekday == 4) & ~stood,
            core & shrink,
            left_buy & second,
            left_buy,
            score >= 4,
        ],
        [L_HALF_RIGHT, L_HALF_FRI, L_CLEAR, L_WAIT, L_CLEAR, L_PAUSE, L_SECOND, L_FIRST, L_WATCH],
        default=L_NONE,
    )
    ready = np.isfinite(w_ma20) & np.isfinite(w_ma60)
    return np.where(ready, action, L_NONE).astype(np.int8)


def build_pair(daily: pd.DataFrame, bench: pd.DataFrame, use_rs: bool, use_bias: bool) -> pd.DataFrame:
    ordered = daily.sort_values("date").reset_index(drop=True)
    right = build_signals(ordered, bench, 20, 1.0, True, use_rs, use_bias)
    ohlcv = {
        "close": ordered["close"].to_numpy(float),
        "low": ordered["low"].to_numpy(float),
        "volume": ordered["volume"].fillna(0).to_numpy(float),
    }
    pack = weekly_pack(pd.to_datetime(ordered["date"]), ohlcv)
    # 右侧信号沿用 factor_compare 的周线。这里只取左侧。日期与 right 对齐。
    left = build_left_signals(ordered, pack, use_bias)
    if len(left) != len(right):
        raise RuntimeError("左右信号长度不一致")
    out = right.copy()
    out["left"] = left
    return out


def run_dual(panels: dict[str, pd.DataFrame], master: pd.DatetimeIndex) -> dict:
    codes = list(panels)
    aligned = {}
    for code, frame in panels.items():
        f = frame.set_index("date").reindex(master)
        f["action"] = f["action"].ffill().fillna(ACTION_NONE)
        f["left"] = f["left"].ffill().fillna(L_NONE)
        f["open"] = f["open"].ffill()
        aligned[code] = f
    units = {code: 0.0 for code in codes}
    track = {code: None for code in codes}
    prev_right = {code: ACTION_NONE for code in codes}
    prev_left = {code: L_NONE for code in codes}
    prev_w = {code: 0.0 for code in codes}
    sleeves = {
        code: {"shares": 0.0, "invested": 0.0, "realized": 0.0, "entry": None, "side": None}
        for code in codes
    }
    equity = 1.0
    curve = []
    trades = []
    entries = {"right": 0, "left": 0}
    turnover = 0.0
    dates = list(master)
    opens = {code: aligned[code]["open"].to_numpy(float) for code in codes}
    rights = {code: aligned[code]["action"].to_numpy(int) for code in codes}
    lefts = {code: aligned[code]["left"].to_numpy(int) for code in codes}

    def book(code, day, price, new_units, side):
        sleeve = sleeves[code]
        old = units[code]
        if new_units > old + 1e-12:
            if old <= 1e-12:
                entries[side] += 1
                sleeve["entry"] = day
                sleeve["side"] = side
            buy = new_units - old
            sleeve["shares"] += buy
            sleeve["invested"] += buy * price
        elif new_units < old - 1e-12:
            sell = min(old - new_units, sleeve["shares"])
            sleeve["realized"] += sell * price
            sleeve["shares"] = max(0.0, sleeve["shares"] - sell)
            if new_units <= 1e-12:
                trades.append({
                    "code": code,
                    "side": sleeve["side"],
                    "pnl": sleeve["realized"] - sleeve["invested"],
                    "ret": (sleeve["realized"] - sleeve["invested"]) / sleeve["invested"] if sleeve["invested"] else 0.0,
                    "entry": sleeve["entry"],
                    "exit": day,
                })
                sleeve.update(shares=0.0, invested=0.0, realized=0.0, entry=None, side=None)
        units[code] = new_units
        track[code] = side if new_units > 1e-12 else None

    for i in range(len(dates) - 2):
        if dates[i] < pd.Timestamp(START):
            continue
        raw_right = {}
        raw_left = {}
        for code in codes:
            price = opens[code][i + 1]
            right_action = int(rights[code][i])
            left_action = int(lefts[code][i])
            if not np.isfinite(price) or price <= 0:
                prev_right[code] = right_action
                prev_left[code] = left_action
                continue
            side = track[code]
            current = units[code]
            if side == "right":
                if right_action == ACTION_CLEAR:
                    target = 0.0
                elif right_action == ACTION_TRIM2 and prev_right[code] != ACTION_TRIM2 and current > 0:
                    target = current * 0.65
                elif right_action == ACTION_TRIM1 and prev_right[code] != ACTION_TRIM1 and current > 0:
                    target = current * 0.75
                elif right_action in BUY_ACTIONS:
                    target = max(current, TARGETS[right_action])
                else:
                    target = current
                book(code, dates[i + 1], price, target, "right" if target > 1e-12 else None)
            elif side == "left":
                if left_action == L_CLEAR:
                    target, new_side = 0.0, None
                elif left_action == L_HALF_RIGHT and prev_left[code] != L_HALF_RIGHT and current > 0:
                    target, new_side = current * 0.5, "right"
                elif left_action == L_HALF_FRI and prev_left[code] != L_HALF_FRI and current > 0:
                    target, new_side = current * 0.5, "left"
                elif left_action in LEFT_BUYS:
                    target, new_side = max(current, LEFT_TARGETS[left_action]), "left"
                else:
                    target, new_side = current, "left"
                book(code, dates[i + 1], price, target, new_side)
            else:
                if right_action in BUY_ACTIONS:
                    book(code, dates[i + 1], price, TARGETS[right_action], "right")
                elif left_action in LEFT_BUYS:
                    book(code, dates[i + 1], price, LEFT_TARGETS[left_action], "left")
            if track[code] == "right":
                raw_right[code] = units[code]
            elif track[code] == "left":
                raw_left[code] = units[code]
            prev_right[code] = right_action
            prev_left[code] = left_action
        sum_r = sum(raw_right.values())
        sum_l = sum(raw_left.values())
        scale_r = min(1.0, RIGHT_CAP / sum_r) if sum_r > 0 else 1.0
        scale_l = min(1.0, LEFT_CAP / sum_l) if sum_l > 0 else 1.0
        weights = {code: 0.0 for code in codes}
        for code, value in raw_right.items():
            weights[code] += value * scale_r
        for code, value in raw_left.items():
            weights[code] += value * scale_l
        step = 0.0
        turn = 0.0
        for code in codes:
            p0, p1 = opens[code][i + 1], opens[code][i + 2]
            ret = p1 / p0 - 1.0 if np.isfinite(p0) and np.isfinite(p1) and p0 > 0 else 0.0
            step += weights[code] * ret
            turn += abs(weights[code] - prev_w[code])
        equity *= 1 + step - turn * COST
        turnover += turn
        prev_w = weights
        curve.append((dates[i + 1], equity, sum(weights.values()), sum(value * scale_r for value in raw_right.values()), sum(value * scale_l for value in raw_left.values())))

    if dates:
        last_day = dates[-1]
        for code in codes:
            price = opens[code][-1]
            if units[code] > 1e-12 and np.isfinite(price):
                book(code, last_day, price, 0.0, None)

    eq = np.array([row[1] for row in curve])
    years = (curve[-1][0] - curve[0][0]).days / 365.25
    peak = np.maximum.accumulate(eq)
    def side_stats(side):
        picked = [t for t in trades if t["side"] == side]
        rets = [t["ret"] for t in picked]
        wins = [t for t in picked if t["pnl"] > 0]
        return {
            "trades": len(picked),
            "win_rate": (len(wins) / len(picked)) if picked else None,
            "avg_trade": float(np.mean(rets)) if rets else None,
            "entries_per_year": entries[side] / years,
        }
    return {
        "cagr": float(eq[-1] ** (1 / years) - 1),
        "maxdd": float((eq / peak - 1).min()),
        "end": float(eq[-1]),
        "win_rate": (sum(t["pnl"] > 0 for t in trades) / len(trades)) if trades else None,
        "avg_trade": float(np.mean([t["ret"] for t in trades])) if trades else None,
        "trades": len(trades),
        "turnover": (turnover / 2) / years,
        "avg_exposure": float(np.mean([row[2] for row in curve])),
        "avg_right": float(np.mean([row[3] for row in curve])),
        "avg_left": float(np.mean([row[4] for row in curve])),
        "right": side_stats("right"),
        "left": side_stats("left"),
        "years": years,
        "start": str(curve[0][0].date()),
        "finish": str(curve[-1][0].date()),
    }


def verify(code, name, category, daily, bench) -> dict:
    pair = build_pair(daily, bench, True, True)
    mapping_r = {
        "清仓": ACTION_CLEAR, "减仓二档": ACTION_TRIM2, "减仓一档": ACTION_TRIM1,
        "回踩重仓": ACTION_HEAVY, "确认加仓": ACTION_CONFIRM, "试错买入": ACTION_PROBE,
        "观察": ACTION_WATCH, "无信号": ACTION_NONE, "样本不足": ACTION_NONE,
    }
    mapping_l = {
        "全清": L_CLEAR, "卖半转右侧": L_HALF_RIGHT, "卖半等周五": L_HALF_FRI,
        "周五再定": L_WAIT, "暂缓": L_PAUSE, "第二批": L_SECOND, "第一批": L_FIRST,
        "观察": L_WATCH, "无信号": L_NONE, "样本不足": L_NONE,
    }
    raw = daily.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    b = bench.copy()
    b["date"] = pd.to_datetime(b["date"])
    sig = pair[pair["date"] >= START]
    step = max(1, len(sig) // 24)
    checked = mismatch_r = mismatch_l = 0
    samples = []
    for pos in range(0, len(sig), step):
        day = pd.Timestamp(sig["date"].iloc[pos])
        row = analyze_etf(code, name, category, raw[raw["date"] <= day], b[b["date"] <= day], market_regime(b[b["date"] <= day]))
        got_r = int(sig["action"].iloc[pos])
        got_l = int(sig["left"].iloc[pos])
        exp_r = mapping_r[row["action_right"]]
        exp_l = mapping_l[row["action_left"]]
        checked += 1
        if got_r != exp_r:
            mismatch_r += 1
            if len(samples) < 6:
                samples.append((str(day.date()), "右", row["action_right"], got_r))
        if got_l != exp_l:
            mismatch_l += 1
            if len(samples) < 8:
                samples.append((str(day.date()), "左", row["action_left"], got_l, row["detail_left"][:48]))
    return {"code": code, "checked": checked, "mismatch_right": mismatch_r, "mismatch_left": mismatch_l, "samples": samples}


def main():
    universe, bars, bench = load_universe()
    bench = bench[bench["date"] >= WARMUP].copy()
    master = pd.DatetimeIndex(sorted(bench.loc[bench["date"] >= "2018-06-01", "date"]))
    checks = [verify(item["code"], item["name"], item["category"], bars[item["code"]], bench) for item in universe[:2]]
    print("checks", checks, flush=True)
    variants = [
        ("原规则", True, True),
        ("去掉比价和入场BIAS", False, False),
    ]
    report = {"checks": checks, "benchmark": buy_hold(bench, master), "names": [item["code"] + " " + item["name"] for item in universe], "variants": []}
    for label, use_rs, use_bias in variants:
        panels = {}
        for item in universe:
            daily = bars[item["code"]]
            daily = daily[daily["date"] >= WARMUP]
            panels[item["code"]] = build_pair(daily, bench, use_rs, use_bias)
        stats = run_dual(panels, master)
        stats["label"] = label
        report["variants"].append(stats)
        print(label, {k: stats[k] for k in ("cagr", "maxdd", "win_rate", "avg_trade", "trades", "avg_exposure", "avg_right", "avg_left")}, flush=True)
    path = CACHE / "dual_track_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, default=str, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
