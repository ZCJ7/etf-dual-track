from __future__ import annotations

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.data import BENCH_CODE, ROOT, load_bars, load_instruments, maybe_autostart, start_refresh, status
from app.indicators import add_daily_features
from app.strategy import analyze_etf, market_regime

app = FastAPI(title="ETF 左右侧决策看板")
STATIC = ROOT / "static"

_cache: dict = {"key": None, "payload": None}


def _chart_payload(code: str) -> dict:
    raw = load_bars(code)
    if raw.empty:
        return {"code": code, "candles": [], "ma20": [], "ma60": [], "hist": []}
    daily = add_daily_features(raw).tail(220)
    candles, ma20, ma60, hist = [], [], [], []
    for row in daily.itertuples(index=False):
        day = row.date.strftime("%Y-%m-%d")
        candles.append({"time": day, "open": round(row.open, 4), "high": round(row.high, 4), "low": round(row.low, 4), "close": round(row.close, 4)})
        if row.ma20 == row.ma20:
            ma20.append({"time": day, "value": round(float(row.ma20), 4)})
        if row.ma60 == row.ma60:
            ma60.append({"time": day, "value": round(float(row.ma60), 4)})
        if row.hist == row.hist:
            hist.append({"time": day, "value": round(float(row.hist), 6)})
    return {"code": code, "candles": candles, "ma20": ma20, "ma60": ma60, "hist": hist}


def build_dashboard() -> dict:
    info = status()
    instruments = load_instruments()
    bench = load_bars(BENCH_CODE)
    if bench.empty or instruments.empty:
        return {"ready": False, "status": info, "market": None, "right": [], "left": []}
    regime = market_regime(bench)
    rows = []
    for item in instruments.itertuples(index=False):
        bars = load_bars(item.code)
        if bars.empty:
            continue
        rows.append(analyze_etf(item.code, item.name, item.category, bars, bench, regime))

    def rank(tone: str, score: int) -> tuple:
        order = {"buy": 0, "sell": 1, "wait": 2, "off": 3}
        return (order.get(tone, 9), -score)

    right = sorted(rows, key=lambda row: rank(row["tone_right"], row["score_right"]))
    left = sorted(rows, key=lambda row: rank(row["tone_left"], row["score_left"]))
    return {
        "ready": True,
        "status": info,
        "market": regime,
        "right": right,
        "left": left,
    }


@app.get("/api/status")
def api_status():
    maybe_autostart()
    return status()


@app.post("/api/refresh")
def api_refresh(
    limit: int = Query(120, ge=10, le=400),
    min_amount: float = Query(20_000_000, ge=0),
):
    started = start_refresh(limit=limit, min_amount=min_amount)
    _cache["key"] = None
    return {"started": started, "status": status()}


@app.get("/api/dashboard")
def api_dashboard():
    maybe_autostart()
    info = status()
    key = (info.get("updated_at"), info.get("symbols"), info.get("running"))
    if _cache["key"] != key or _cache["payload"] is None:
        _cache["key"] = key
        _cache["payload"] = build_dashboard()
    return _cache["payload"]


@app.get("/api/etf/{code}")
def api_etf(code: str):
    payload = _chart_payload(code)
    board = api_dashboard()
    row = next((item for item in board.get("right", []) if item["code"] == code), None)
    payload["signal"] = row
    return payload


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
