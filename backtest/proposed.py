#!/usr/bin/env python3
"""建议方案回测：三档补仓，右侧只在周线止盈/止损。

相对「去掉比价和入场 BIAS」的上一版，改动是：
- 右侧同时最多 3 只。买点当天先上 15%；之后的第一个周五若周收盘仍在
  周线 MA20 上，补到 45%；已到确认档且日线回踩 MA20 未破，补到 75%。
  多只按目标仓位分摊，右侧合计不超过 75%。
- 右侧出场只留两条：周五收盘跌破自己的周线 MA20 则清仓；持仓期间周线
  顶背离第一次出现时减半，同一轮只减这一次。日线红柱缩短、跌破日线均线、
  日线顶背离都不减仓。
- 左侧同时只留 1 只，第一笔 12.5%，之后出现「连续两天绿柱缩短且不创新低」
  再补到 25%。碰到周线 MA20 仍按原来的卖半、全清处理。
入场仍不要求比价，也不要求 BIAS 门槛。MACD、量能、支撑、周线 MA60 拐头否决保留。

capped=False 时不再限制同时持有几只，也不再把右侧压进 75%、左侧压进 25%。
每只仍按自己的档位计目标仓位。fit_account=True 时，当天目标加总超过 100% 就同比例缩小到一份资金；
fit_account=False 时按目标仓位直接加总，总仓位可以超过一份本金。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.dual_track import (  # noqa: E402
    L_CLEAR,
    L_HALF_FRI,
    L_HALF_RIGHT,
    L_NONE,
    L_SECOND,
    LEFT_BUYS,
    build_left_signals,
    build_pair,
    run_dual,
    weekly_pack,
)
from backtest.factor_compare import (  # noqa: E402
    CACHE,
    COST,
    START,
    WARMUP,
    buy_hold,
    load_universe,
)

TIER_UNITS = {1: 0.15, 2: 0.45, 3: 0.75}
RIGHT_CAP = 0.75
LEFT_CAP = 0.25
MAX_RIGHT = 3


def build_frame(daily: pd.DataFrame) -> pd.DataFrame:
    ordered = daily.sort_values("date").reset_index(drop=True)
    close = ordered["close"].to_numpy(float)
    high = ordered["high"].to_numpy(float)
    low = ordered["low"].to_numpy(float)
    volume = ordered["volume"].fillna(0).to_numpy(float)
    dates = pd.to_datetime(ordered["date"])
    pack = weekly_pack(dates, {"close": close, "low": low, "volume": volume})
    ma20 = pd.Series(close).rolling(20).mean().to_numpy()
    ema_f = pd.Series(close).ewm(span=12, adjust=False).mean()
    ema_s = pd.Series(close).ewm(span=26, adjust=False).mean()
    dif = ema_f - ema_s
    hist = (dif - dif.ewm(span=9, adjust=False).mean()).to_numpy()
    hist_prev = pd.Series(hist).shift(1).to_numpy()
    vol_ratio = volume / pd.Series(volume).shift(1).rolling(5).mean().to_numpy()
    high_20 = pd.Series(high).shift(1).rolling(20).max().to_numpy()
    w_ma = pack["w_ma20"]
    weekly_above = np.isfinite(w_ma) & (close > w_ma)
    red_longer = (pack["w_hist"] > 0) & (pack["w_hist"] > pack["w_hist_prev"])
    golden = (pack["w_dif_prev"] <= pack["w_dea_prev"]) & (pack["w_dif"] > pack["w_dea"])
    trend = weekly_above & (red_longer | golden)
    timing = ((hist > 0) & (hist > hist_prev)) | ((hist < 0) & (hist > hist_prev))
    pullback = np.isfinite(ma20) & (low <= ma20 * 1.015) & (close >= ma20 * 0.99)
    breakout = np.isfinite(high_20) & (close >= high_20)
    volume_ok = np.where(pullback, vol_ratio < 0.8, np.where(breakout, vol_ratio > 1.2, False))
    buy_ready = trend & timing & volume_ok & np.isfinite(vol_ratio)
    bias6 = np.where(np.isfinite(pack["w_ma6"]) & (pack["w_ma6"] > 0), (close / pack["w_ma6"] - 1.0) * 100.0, np.nan)
    strength = np.where(weekly_above, close / w_ma - 1.0, -np.inf)
    left = build_left_signals(ordered, pack, use_bias=False)
    return pd.DataFrame({
        "date": dates.to_numpy(),
        "open": ordered["open"].to_numpy(float),
        "close": close,
        "buy_ready": buy_ready,
        "pullback": pullback & weekly_above,
        "weekly_above": weekly_above,
        "weekly_below": np.isfinite(w_ma) & (close < w_ma),
        "div_w": pack["div_w"].astype(bool),
        "left": left,
        "bias6": bias6,
        "strength": strength,
    })


def run_proposed(
    panels: dict[str, pd.DataFrame],
    master: pd.DatetimeIndex,
    capped: bool = True,
    fit_account: bool = True,
) -> dict:
    codes = list(panels)
    aligned = {}
    for code, frame in panels.items():
        f = frame.set_index("date").reindex(master)
        for col in ("buy_ready", "pullback", "weekly_above", "weekly_below", "div_w"):
            f[col] = f[col].ffill().fillna(False).astype(bool)
        f["left"] = f["left"].ffill().fillna(L_NONE)
        f["open"] = f["open"].ffill()
        f["bias6"] = f["bias6"].ffill()
        f["strength"] = f["strength"].ffill()
        aligned[code] = f

    units = {c: 0.0 for c in codes}
    track = {c: None for c in codes}
    tier = {c: 0 for c in codes}
    took = {c: False for c in codes}
    prev_div = {c: False for c in codes}
    prev_left = {c: L_NONE for c in codes}
    prev_w = {c: 0.0 for c in codes}
    sleeves = {c: {"shares": 0.0, "invested": 0.0, "realized": 0.0, "entry": None, "side": None, "max_tier": 0} for c in codes}
    trades = []
    half_sells = 0
    entries = {"right": 0, "left": 0}
    equity = 1.0
    curve = []
    turnover = 0.0
    grosses = []
    name_counts = []
    dates = list(master)
    opens = {c: aligned[c]["open"].to_numpy(float) for c in codes}

    def book(code, day, price, new_units, side, new_tier):
        nonlocal half_sells
        sleeve = sleeves[code]
        old = units[code]
        if new_units > old + 1e-12:
            if old <= 1e-12:
                entries[side] += 1
                sleeve["entry"] = day
                sleeve["side"] = side
                sleeve["max_tier"] = new_tier
            sleeve["max_tier"] = max(sleeve["max_tier"], new_tier)
            buy = new_units - old
            sleeve["shares"] += buy
            sleeve["invested"] += buy * price
        elif new_units < old - 1e-12:
            sell = min(old - new_units, sleeve["shares"])
            sleeve["realized"] += sell * price
            sleeve["shares"] = max(0.0, sleeve["shares"] - sell)
            if new_units <= 1e-12 and sleeve["invested"]:
                trades.append({
                    "code": code,
                    "side": sleeve["side"],
                    "ret": (sleeve["realized"] - sleeve["invested"]) / sleeve["invested"],
                    "pnl": sleeve["realized"] - sleeve["invested"],
                    "hold": (pd.Timestamp(day) - pd.Timestamp(sleeve["entry"])).days,
                    "max_tier": sleeve["max_tier"],
                })
                sleeve.update(shares=0.0, invested=0.0, realized=0.0, entry=None, side=None, max_tier=0)
        units[code] = new_units
        if new_units <= 1e-12:
            track[code] = None
            tier[code] = 0
            took[code] = False
        else:
            track[code] = side
            tier[code] = new_tier

    for i in range(len(dates) - 2):
        if dates[i] < pd.Timestamp(START):
            continue
        day = dates[i]
        weekday = int(day.dayofweek)
        friday = weekday == 4
        raw_right, raw_left = {}, {}

        for code in codes:
            price = opens[code][i + 1]
            row = aligned[code].iloc[i]
            if not np.isfinite(price) or price <= 0 or track[code] is None:
                prev_div[code] = bool(row["div_w"])
                prev_left[code] = int(row["left"])
                continue
            if track[code] == "left":
                action = int(row["left"])
                current = units[code]
                if action == L_CLEAR:
                    book(code, dates[i + 1], price, 0.0, None, 0)
                elif action == L_HALF_RIGHT and prev_left[code] != L_HALF_RIGHT and current > 0:
                    half_sells += 1
                    book(code, dates[i + 1], price, current * 0.5, "right", 1)
                elif action == L_HALF_FRI and prev_left[code] != L_HALF_FRI and current > 0:
                    half_sells += 1
                    book(code, dates[i + 1], price, current * 0.5, "left", 1)
                elif action == L_SECOND and tier[code] < 2 and i > 0:
                    book(code, dates[i + 1], price, LEFT_CAP, "left", 2)
                prev_div[code] = bool(row["div_w"])
                prev_left[code] = action
                continue

            # 右侧：先止损，再一次性止盈，最后才补档。
            if friday and bool(row["weekly_below"]):
                book(code, dates[i + 1], price, 0.0, None, 0)
            else:
                div_edge = bool(row["div_w"]) and not prev_div[code]
                if div_edge and not took[code] and units[code] > 0:
                    half_sells += 1
                    took[code] = True
                    book(code, dates[i + 1], price, units[code] * 0.5, "right", tier[code])
                elif bool(row["weekly_above"]) and not took[code]:
                    new_tier = tier[code]
                    if new_tier == 1 and friday:
                        new_tier = 2
                    if new_tier >= 2 and bool(row["pullback"]):
                        new_tier = 3
                    if new_tier != tier[code]:
                        book(code, dates[i + 1], price, TIER_UNITS[new_tier], "right", new_tier)
            prev_div[code] = bool(row["div_w"])
            prev_left[code] = int(row["left"])

        held_right = [c for c in codes if track[c] == "right"]
        slots = len(codes) if not capped else MAX_RIGHT - len(held_right)
        if slots > 0:
            candidates = []
            for code in codes:
                if track[code] is not None:
                    continue
                row = aligned[code].iloc[i]
                price = opens[code][i + 1]
                if not np.isfinite(price) or price <= 0 or not bool(row["buy_ready"]):
                    continue
                candidates.append((float(row["strength"]), code))
            candidates.sort(reverse=True)
            for _, code in candidates[:slots]:
                row = aligned[code].iloc[i]
                price = opens[code][i + 1]
                if friday and bool(row["pullback"]):
                    new_tier = 3
                elif friday:
                    new_tier = 2
                else:
                    new_tier = 1
                book(code, dates[i + 1], price, TIER_UNITS[new_tier], "right", new_tier)
                prev_div[code] = bool(row["div_w"])

        left_open = any(track[c] == "left" for c in codes)
        if capped and left_open:
            pass
        else:
            candidates = []
            for code in codes:
                if track[code] is not None:
                    continue
                row = aligned[code].iloc[i]
                price = opens[code][i + 1]
                if not np.isfinite(price) or price <= 0 or int(row["left"]) not in LEFT_BUYS:
                    continue
                bias = row["bias6"]
                bias = float(bias) if np.isfinite(bias) else 0.0
                candidates.append((bias, code))
            candidates.sort()
            if capped:
                candidates = candidates[:1]
            for _, code in candidates:
                price = opens[code][i + 1]
                book(code, dates[i + 1], price, 0.125, "left", 1)

        for code in codes:
            if track[code] == "right":
                raw_right[code] = units[code]
            elif track[code] == "left":
                raw_left[code] = units[code]
        sum_r = sum(raw_right.values())
        sum_l = sum(raw_left.values())
        gross = sum_r + sum_l
        grosses.append(gross)
        name_counts.append(len(raw_right) + len(raw_left))
        if capped:
            scale_r = min(1.0, RIGHT_CAP / sum_r) if sum_r > 0 else 1.0
            scale_l = min(1.0, LEFT_CAP / sum_l) if sum_l > 0 else 1.0
        elif fit_account:
            # 单票仍按三档。加总超过一份资金时同比例缩小，否则一份本金覆盖不了。
            scale = min(1.0, 1.0 / gross) if gross > 0 else 1.0
            scale_r = scale_l = scale
        else:
            scale_r = scale_l = 1.0
        weights = {c: 0.0 for c in codes}
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
        curve.append((dates[i + 1], equity, sum(weights.values()), sum(v * scale_r for v in raw_right.values()), sum(v * scale_l for v in raw_left.values())))

    if dates:
        last_px_day = dates[-1]
        for code in codes:
            price = opens[code][-1]
            if units[code] > 1e-12 and np.isfinite(price):
                book(code, last_px_day, price, 0.0, None, 0)

    eq = np.array([row[1] for row in curve])
    years = (curve[-1][0] - curve[0][0]).days / 365.25
    peak = np.maximum.accumulate(eq)

    def side_stats(side):
        picked = [t for t in trades if t["side"] == side]
        rets = [t["ret"] for t in picked]
        holds = [t["hold"] for t in picked]
        wins = [t["ret"] for t in picked if t["pnl"] > 0]
        losses = [t["ret"] for t in picked if t["pnl"] <= 0]
        return {
            "trades": len(picked),
            "win_rate": (len(wins) / len(picked)) if picked else None,
            "avg_trade": float(np.mean(rets)) if rets else None,
            "avg_win": float(np.mean(wins)) if wins else None,
            "avg_loss": float(np.mean(losses)) if losses else None,
            "median_hold": float(np.median(holds)) if holds else None,
            "entries_per_year": entries[side] / years,
            "reached_heavy": sum(1 for t in picked if t["max_tier"] >= 3),
        }

    return {
        "cagr": float(eq[-1] ** (1 / years) - 1),
        "maxdd": float((eq / peak - 1).min()),
        "end": float(eq[-1]),
        "win_rate": (sum(t["pnl"] > 0 for t in trades) / len(trades)) if trades else None,
        "avg_trade": float(np.mean([t["ret"] for t in trades])) if trades else None,
        "median_hold": float(np.median([t["hold"] for t in trades])) if trades else None,
        "trades": len(trades),
        "half_sells": half_sells,
        "turnover": (turnover / 2) / years,
        "avg_exposure": float(np.mean([row[2] for row in curve])),
        "avg_right": float(np.mean([row[3] for row in curve])),
        "avg_left": float(np.mean([row[4] for row in curve])),
        "avg_gross": float(np.mean(grosses)) if grosses else None,
        "max_gross": float(np.max(grosses)) if grosses else None,
        "share_over_one": float(np.mean(np.array(grosses) > 1.0)) if grosses else None,
        "avg_names": float(np.mean(name_counts)) if name_counts else None,
        "max_names": int(np.max(name_counts)) if name_counts else 0,
        "right": side_stats("right"),
        "left": side_stats("left"),
        "years": years,
        "start": str(curve[0][0].date()),
        "finish": str(curve[-1][0].date()),
    }


def _pack_old(stats, label):
    stats = dict(stats)
    stats["label"] = label
    return stats


def main():
    universe, bars, bench = load_universe()
    bench = bench[bench["date"] >= WARMUP].copy()
    master = pd.DatetimeIndex(sorted(bench.loc[bench["date"] >= "2018-06-01", "date"]))
    old = {}
    proposed = {}
    for item in universe:
        daily = bars[item["code"]]
        daily = daily[daily["date"] >= WARMUP]
        old[item["code"]] = build_pair(daily, bench, False, False)
        proposed[item["code"]] = build_frame(daily)
    loose = run_dual(old, master)
    fresh = run_proposed(proposed, master, capped=True)
    opened = run_proposed(proposed, master, capped=False, fit_account=True)
    raw = run_proposed(proposed, master, capped=False, fit_account=False)
    report = {
        "benchmark": buy_hold(bench, master),
        "loose": _pack_old(loose, "去掉比价和入场BIAS"),
        "proposed": _pack_old(fresh, "三档补仓+周线出场"),
        "open_book": _pack_old(opened, "信号全做，超过100%时缩到一份资金"),
        "open_raw": _pack_old(raw, "信号全做，目标仓位直接加总"),
    }
    path = CACHE / "proposed_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, default=str, indent=2))
    print(json.dumps(report, ensure_ascii=False, default=str, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
