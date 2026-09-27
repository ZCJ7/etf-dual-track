"""左右侧双轨买卖点。规则来自既定执行手册，看板只做对照，不代替下单。"""

from __future__ import annotations

import pandas as pd

from app.indicators import (
    add_daily_features,
    add_weekly_features,
    last_number,
    near_level,
    swing_lows,
    to_weekly,
    volume_node,
)


def _check(key: str, ok: bool, text: str) -> dict:
    return {"key": key, "ok": bool(ok), "text": text}


def _fmt(value, digits=2, suffix=""):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{value:.{digits}f}{suffix}"


def _hist_state(hist, prev) -> str:
    if hist is None or prev is None:
        return "数据不足"
    if hist > 0 and hist > prev:
        return "红柱变长"
    if hist > 0 and hist <= prev:
        return "红柱缩短"
    if hist < 0 and hist > prev:
        return "绿柱缩短"
    if hist < 0 and hist <= prev:
        return "绿柱变长"
    return "柱体走平"


def market_regime(bench_daily: pd.DataFrame) -> dict:
    daily = add_daily_features(bench_daily)
    weekly = add_weekly_features(to_weekly(daily))
    close = last_number(weekly["close"])
    ma20 = last_number(weekly["ma20"])
    ma60 = last_number(weekly["ma60"])
    hist = last_number(weekly["hist"])
    prev = last_number(weekly["hist"], 2)
    above_ma20 = close is not None and ma20 is not None and close > ma20
    above_ma60 = close is not None and ma60 is not None and close > ma60
    state = _hist_state(hist, prev)
    right_mode, left_mode = "按ETF", "按ETF"
    headline = f"沪深300周线{'站上' if above_ma20 else '在'} MA20 {'之上' if above_ma20 else '之下'}，{state}。这只作对照，开仓看每只 ETF 自己的周线 MA20。"

    as_of = daily["date"].iloc[-1].strftime("%Y-%m-%d") if len(daily) else None
    return {
        "name": "沪深300",
        "as_of": as_of,
        "close": close,
        "weekly_ma20": ma20,
        "weekly_ma60": ma60,
        "above_ma20": above_ma20,
        "above_ma60": above_ma60,
        "macd_state": state,
        "right_mode": right_mode,
        "left_mode": left_mode,
        "headline": headline,
        "right_cap": "70%-80%",
        "left_cap": "20%-30%",
        "steps": [
            {"n": "1", "title": "大盘周线 MA60", "text": "在上方" if above_ma60 else "在下方，不限制开仓"},
            {"n": "2", "title": "大盘周线 MA20", "text": f"{'站上' if above_ma20 else '下方'} · {state}，不限制开仓"},
            {"n": "3", "title": "今日执行", "text": "看这只 ETF 自己的周线 MA20"},
            {"n": "4", "title": "持仓", "text": "右侧三档止盈 / 左侧看站上 MA20 的姿态"},
            {"n": "5", "title": "止损", "text": "右侧跌破周线 MA20；左侧跌破支撑 2%-3%"},
        ],
    }


def _relative_strength(daily: pd.DataFrame, bench: pd.DataFrame) -> dict:
    left = daily[["date", "close"]].rename(columns={"close": "etf"})
    right = bench[["date", "close"]].rename(columns={"close": "bench"})
    merged = left.merge(right, on="date", how="inner")
    if len(merged) < 25:
        return {"ok": False, "text": "比价样本不足"}
    rs = merged["etf"] / merged["bench"]
    now = float(rs.iloc[-1])
    prev = float(rs.iloc[-21])
    slope = float(rs.iloc[-1] - rs.iloc[-11])
    ok = now > prev and slope > 0
    change = (now / prev - 1) * 100
    text = f"20日比价 {change:+.1f}%，{'持续向上' if ok else '没有持续跑赢大盘'}"
    return {"ok": ok, "text": text, "change": change}


def _top_divergence(close: pd.Series, hist: pd.Series, split: int = 20) -> bool:
    if len(close) < split * 2 or len(hist) < split * 2:
        return False
    c_prev = close.iloc[-split * 2 : -split]
    c_now = close.iloc[-split:]
    h_prev = hist.iloc[-split * 2 : -split]
    h_now = hist.iloc[-split:]
    if c_prev.isna().all() or h_prev.isna().all() or h_now.isna().all():
        return False
    return (
        float(c_now.max()) > float(c_prev.max())
        and float(h_now.max()) < float(h_prev.max())
        and float(h_now.max()) > 0
    )


def _supports(daily: pd.DataFrame, weekly: pd.DataFrame, price: float) -> list[dict]:
    found = []

    def add(name: str, level, confidence: str):
        level_n = None if level is None or pd.isna(level) else float(level)
        if near_level(price, level_n):
            found.append({"name": name, "level": level_n, "confidence": confidence})

    add("周线MA60", last_number(weekly["ma60"]), "高")
    add("周线MA120", last_number(weekly["ma120"]), "高")
    add("周线MA250", last_number(weekly["ma250"]), "高")
    add("日线MA120", last_number(daily["ma120"]), "中")
    add("日线MA250", last_number(daily["ma250"]), "中")
    if len(weekly) > 30:
        major = float(weekly["low"].min())
        add("历史大底", major, "高")
    for level in swing_lows(weekly):
        add("前低", level, "中")
    add("筹码密集", volume_node(daily), "高")
    rank = {"高": 0, "中": 1}
    found.sort(key=lambda item: (rank[item["confidence"]], abs(price - item["level"])))
    # 同一类前低只留最近的一个
    unique = []
    seen = set()
    for item in found:
        if item["name"] in seen and item["name"] == "前低":
            continue
        seen.add(item["name"])
        unique.append(item)
    return unique


def analyze_etf(code: str, name: str, category: str, raw_daily: pd.DataFrame, bench_daily: pd.DataFrame, regime: dict) -> dict:
    daily = add_daily_features(raw_daily)
    weekly = add_weekly_features(to_weekly(daily))
    bench = add_daily_features(bench_daily)
    if len(daily) < 80 or len(weekly) < 30:
        return {
            "code": code,
            "name": name,
            "category": category,
            "close": last_number(daily["close"]) if len(daily) else None,
            "action_right": "样本不足",
            "action_left": "样本不足",
            "tone_right": "wait",
            "tone_left": "wait",
            "checks_right": [],
            "checks_left": [],
            "position_right": "—",
            "position_left": "—",
            "detail_right": "上市或缓存的K线还不够算周线结构。",
            "detail_left": "上市或缓存的K线还不够算周线结构。",
            "score_right": 0,
            "score_left": 0,
        }

    price = last_number(daily["close"])
    prev_close = last_number(daily["close"], 2)
    pct = None if prev_close in (None, 0) else (price / prev_close - 1) * 100
    w_close = last_number(weekly["close"])
    w_ma20 = last_number(weekly["ma20"])
    w_ma60 = last_number(weekly["ma60"])
    w_ma60_prev = last_number(weekly["ma60"], 5)
    w_hist = last_number(weekly["hist"])
    w_hist_prev = last_number(weekly["hist"], 2)
    w_dif = last_number(weekly["dif"])
    w_dea = last_number(weekly["dea"])
    w_dif_prev = last_number(weekly["dif"], 2)
    w_dea_prev = last_number(weekly["dea"], 2)
    w_bias6 = last_number(weekly["bias6"])
    w_vol = last_number(weekly["vol_ratio"])
    d_bias20 = last_number(daily["bias20"])
    d_hist = last_number(daily["hist"])
    d_hist_prev = last_number(daily["hist"], 2)
    d_ma10 = last_number(daily["ma10"])
    d_ma20 = last_number(daily["ma20"])
    d_low = last_number(daily["low"])
    d_high = last_number(daily["high"])
    d_vol = last_number(daily["vol_ratio"])
    weekday = int(daily["date"].iloc[-1].dayofweek)

    rs = _relative_strength(daily, bench)
    weekly_above = w_close is not None and w_ma20 is not None and w_close > w_ma20
    red_longer = w_hist is not None and w_hist_prev is not None and w_hist > 0 and w_hist > w_hist_prev
    golden = (
        w_dif is not None
        and w_dea is not None
        and w_dif_prev is not None
        and w_dea_prev is not None
        and w_dif_prev <= w_dea_prev
        and w_dif > w_dea
    )
    trend_ok = weekly_above and (red_longer or golden)
    daily_macd_ok = (d_hist is not None and d_hist_prev is not None) and (
        (d_hist > 0 and d_hist > d_hist_prev) or (d_hist < 0 and d_hist > d_hist_prev)
    )
    bias_pullback = d_bias20 is not None and d_bias20 <= 1.0
    high_20 = float(daily["high"].iloc[-21:-1].max()) if len(daily) > 22 else None
    breakout = high_20 is not None and price >= high_20
    pullback = d_ma20 is not None and d_low is not None and d_low <= d_ma20 * 1.015 and price >= d_ma20 * 0.99
    if pullback:
        volume_ok = d_vol is not None and d_vol < 0.8
        volume_text = f"回踩量比 {_fmt(d_vol)}，{'缩量' if volume_ok else '还没缩到 0.8 以下'}"
    elif breakout:
        volume_ok = d_vol is not None and d_vol > 1.2
        volume_text = f"突破量比 {_fmt(d_vol)}，{'放量' if volume_ok else '没放到 1.2 以上'}"
    else:
        volume_ok = False
        volume_text = f"量比 {_fmt(d_vol)}，既不是缩量回踩，也不是放量突破"

    checks_right = [
        _check("rs", rs["ok"], rs["text"]),
        _check(
            "trend",
            trend_ok,
            f"周线{'站上' if weekly_above else '未站上'} MA20，{_hist_state(w_hist, w_hist_prev)}"
            + ("，本周金叉" if golden else ""),
        ),
        _check(
            "timing",
            bias_pullback and daily_macd_ok,
            f"日线 BIAS20 {_fmt(d_bias20, 1, '%')}，日线{_hist_state(d_hist, d_hist_prev)}",
        ),
        _check("volume", volume_ok, volume_text),
    ]
    score_right = sum(1 for item in checks_right if item["ok"])

    divergence = _top_divergence(daily["close"], daily["hist"]) or _top_divergence(weekly["close"], weekly["hist"], 8)
    lost_weekly = w_close is not None and w_ma20 is not None and w_close < w_ma20
    red_shorter = w_hist is not None and w_hist_prev is not None and w_hist > 0 and w_hist < w_hist_prev
    daily_red_shorter = d_hist is not None and d_hist_prev is not None and d_hist > 0 and d_hist < d_hist_prev
    broke_daily_ma = (d_ma10 is not None and price < d_ma10) or (d_ma20 is not None and price < d_ma20)

    prev_week_close = last_number(weekly["close"], 2)
    prev_week_ma20 = last_number(weekly["ma20"], 2)
    just_crossed = (
        prev_week_close is not None
        and prev_week_ma20 is not None
        and prev_week_close <= prev_week_ma20
        and weekly_above
    )
    just_lost = (
        prev_week_close is not None
        and prev_week_ma20 is not None
        and prev_week_close > prev_week_ma20
        and lost_weekly
    )
    heavy = pullback and bias_pullback and d_bias20 is not None and d_bias20 >= -3

    buy_ready = all(item["ok"] for item in checks_right)
    if just_lost or (divergence and (weekly_above or just_lost)):
        action_r, tone_r, pos_r = "清仓", "sell", "0"
        detail_r = "本周收盘跌破周线 MA20。" if just_lost else "价格新高而 MACD 未新高，顶背离，全部清仓。"
    elif weekly_above and broke_daily_ma and d_bias20 is not None and d_bias20 <= 2:
        action_r, tone_r, pos_r = "减仓二档", "sell", "再减 30%-40%"
        detail_r = "跌破日线 MA10 或 MA20，且 BIAS 回到 0 轴附近。"
    elif weekly_above and (red_shorter or daily_red_shorter):
        action_r, tone_r, pos_r = "减仓一档", "sell", "减 20%-30%"
        detail_r = "MACD 红柱开始缩短，先降速。"
    elif buy_ready and heavy and not just_crossed:
        action_r, tone_r, pos_r = "回踩重仓", "buy", "加至 70%-80%"
        detail_r = "四维共振，日线回踩 MA20 未破，BIAS 回到 0 轴。"
    elif buy_ready and weekday == 4:
        action_r, tone_r, pos_r = "确认加仓", "buy", "加至 40%-50%"
        detail_r = "周五收盘仍站上周线 MA20，确认仓。"
    elif buy_ready and weekday in (2, 3) and just_crossed:
        action_r, tone_r, pos_r = "试错买入", "buy", "10%-20%"
        detail_r = "周中刚站上周线 MA20，先上试错仓。"
    elif score_right >= 3:
        missing = "、".join(item["key"] for item in checks_right if not item["ok"])
        action_r, tone_r, pos_r = "观察", "wait", "先不开"
        detail_r = f"还差 {missing}。"
    else:
        action_r, tone_r, pos_r = "无信号", "wait", "—"
        detail_r = "右侧四维没有形成共振。"

    # 左侧
    supports = _supports(daily, weekly, price)
    support = supports[0] if supports else None
    support_ok = support is not None and support["confidence"] in ("高", "中")
    own_below_ma20 = w_close is not None and w_ma20 is not None and w_close < w_ma20
    structure_ok = w_ma60 is not None and (w_ma60_prev is None or w_ma60 >= w_ma60_prev)
    space_ok = w_bias6 is not None and w_bias6 < -4
    green_shorter = d_hist is not None and d_hist_prev is not None and d_hist < 0 and d_hist > d_hist_prev
    green_longer = d_hist is not None and d_hist_prev is not None and d_hist < 0 and d_hist < d_hist_prev
    shrink_fall = d_vol is not None and d_vol < 0.8
    panic_volume = d_vol is not None and d_vol > 1.5
    volume_left_ok = not shrink_fall
    prev_low = last_number(daily["low"], 2)
    prev_hist = last_number(daily["hist"], 2)
    prev_hist2 = last_number(daily["hist"], 3)
    no_new_low = prev_low is not None and d_low is not None and d_low >= prev_low
    prev_green_shorter = (
        prev_hist is not None and prev_hist2 is not None and prev_hist < 0 and prev_hist > prev_hist2
    )
    second_batch = green_shorter and prev_green_shorter and no_new_low

    support_text = (
        f"{support['confidence']}置信 {support['name']} {_fmt(support['level'])}"
        if support
        else "附近没有高/中置信支撑，不买"
    )
    checks_left = [
        _check("ma20", own_below_ma20, "周线在自身 MA20 下方" if own_below_ma20 else "周线已站上自身 MA20，不走左侧"),
        _check("support", support_ok, support_text),
        _check("space", space_ok, f"周线 BIAS6 {_fmt(w_bias6, 1, '%')}，标准是低于 -4%"),
        _check(
            "momentum",
            green_shorter and not green_longer,
            "日线绿柱缩短" if green_shorter else ("绿柱仍在变长，禁止出手" if green_longer else f"日线{_hist_state(d_hist, d_hist_prev)}"),
        ),
        _check(
            "volume",
            volume_left_ok,
            "放量急跌，加分" if panic_volume else ("缩量阴跌，暂缓" if shrink_fall else f"量比 {_fmt(d_vol)}，量能中性"),
        ),
        _check("structure", structure_ok, "周线 MA60 没有拐头向下" if structure_ok else "周线 MA60 拐头向下，一票否决"),
    ]
    score_left = sum(1 for item in checks_left if item["ok"])
    core_left = own_below_ma20 and support_ok and space_ok and green_shorter and structure_ok
    left_buy = core_left and volume_left_ok
    stop = None if not support else round(support["level"] * 0.975, 3)

    touched_ma20 = w_ma20 is not None and d_high is not None and d_high >= w_ma20 * 0.995
    stood_on = w_ma20 is not None and price >= w_ma20
    if stood_on and w_vol is not None and w_vol > 1.5:
        action_l, tone_l, pos_l = "卖半转右侧", "sell", "卖 50%，剩余改按右侧"
        detail_l = "放量站上周线 MA20。"
    elif stood_on and w_vol is not None and 1.0 <= w_vol <= 1.5:
        action_l, tone_l, pos_l = "卖半等周五", "sell", "先卖 50%"
        detail_l = "温和放量站上，剩下的等周五收盘。"
    elif touched_ma20 and w_vol is not None and w_vol < 0.8 and not stood_on:
        action_l, tone_l, pos_l = "全清", "sell", "0"
        detail_l = "缩量触及周线 MA20，没有站上。"
    elif touched_ma20 and weekday < 4 and not stood_on:
        action_l, tone_l, pos_l = "周五再定", "wait", "先不动"
        detail_l = "周中摸到周线 MA20。若周五收盘跌破，尾盘全清。"
    elif touched_ma20 and weekday == 4 and not stood_on:
        action_l, tone_l, pos_l = "全清", "sell", "0"
        detail_l = "周五收盘没有站上周线 MA20。"
    elif core_left and shrink_fall:
        action_l, tone_l, pos_l = "暂缓", "wait", "先不开"
        detail_l = "支撑和动能都到了，但还是缩量阴跌，等放量出清。"
    elif left_buy and second_batch:
        action_l, tone_l, pos_l = "第二批", "buy", "加至 20%-30%"
        detail_l = "次日没有新低，绿柱继续缩短。"
    elif left_buy:
        action_l, tone_l, pos_l = "第一批", "buy", "10%-15%"
        detail_l = "周线在自身 MA20 下方，支撑和三低一缩同时满足。"
    elif score_left >= 4:
        missing = "、".join(item["key"] for item in checks_left if not item["ok"])
        action_l, tone_l, pos_l = "观察", "wait", "先不开"
        detail_l = f"还差 {missing}。"
    else:
        action_l, tone_l, pos_l = "无信号", "wait", "—"
        detail_l = "左侧还没同时踩在支撑和三低一缩上。"

    return {
        "code": code,
        "name": name,
        "category": category,
        "close": None if price is None else round(price, 3),
        "pct": None if pct is None else round(pct, 2),
        "as_of": daily["date"].iloc[-1].strftime("%Y-%m-%d"),
        "weekday": weekday,
        "action_right": action_r,
        "action_left": action_l,
        "tone_right": tone_r,
        "tone_left": tone_l,
        "position_right": pos_r,
        "position_left": pos_l,
        "detail_right": detail_r,
        "detail_left": detail_l,
        "checks_right": checks_right,
        "checks_left": checks_left,
        "score_right": score_right,
        "score_left": score_left,
        "bias20": None if d_bias20 is None else round(d_bias20, 2),
        "weekly_bias6": None if w_bias6 is None else round(w_bias6, 2),
        "vol_ratio": None if d_vol is None else round(d_vol, 2),
        "weekly_vol_ratio": None if w_vol is None else round(w_vol, 2),
        "support": support_text if support else "无",
        "stop": stop,
        "rs_text": rs["text"],
        "weekly_above_ma20": weekly_above,
        "macd_daily": _hist_state(d_hist, d_hist_prev),
        "macd_weekly": _hist_state(w_hist, w_hist_prev),
    }
