"""AKShare 行情缓存。全量历史只拉一次，之后按最后交易日增量更新。"""

from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "market.sqlite"
BENCH_CODE = "000300"
TZ = ZoneInfo("Asia/Shanghai")


def _disable_system_proxy():
    """本机 127.0.0.1 代理无响应时，让 AKShare 直连新浪行情。"""
    import requests
    from curl_cffi import requests as creq

    if getattr(requests.Session, "_etf_direct", False):
        return

    def _request(self, method, url, **kwargs):
        return creq.request(
            method=method,
            url=url,
            params=kwargs.get("params"),
            data=kwargs.get("data"),
            headers=kwargs.get("headers"),
            timeout=kwargs.get("timeout") or 25,
            proxies={"http": None, "https": None},
            impersonate="chrome",
        )

    requests.Session.request = _request
    requests.Session._etf_direct = True


def _sina_symbol(code: str) -> str:
    if code.startswith(("5", "6", "9")):
        return f"sh{code}"
    return f"sz{code}"

_lock = threading.Lock()
_status = {
    "running": False,
    "done": 0,
    "total": 0,
    "message": "还没有更新过",
    "error": None,
    "updated_at": None,
}

BROAD_KEYS = (
    "沪深300", "中证500", "中证800", "中证1000", "中证2000", "中证A50", "中证A500",
    "上证50", "上证180", "深证100", "创业板ETF", "创业板50", "创业板指",
    "科创50", "科创100", "科创综", "中证红利", "红利低波", "中证全指", "MSCI", "A500", "A50ETF",
)
INDUSTRY_KEYS = (
    "医药", "医疗", "芯片", "半导体", "新能源", "酒", "银行", "证券", "军工", "光伏",
    "人工智能", "消费", "金融", "地产", "传媒", "计算机", "电子", "通信", "有色",
    "煤炭", "钢铁", "农业", "养殖", "旅游", "游戏", "机器人", "电池", "汽车", "电力",
    "环保", "建材", "化工", "机械", "食品", "家电", "生物", "创新药", "卫星", "数据",
    "云计算", "软件", "互联网", "稀土", "券商", "保险", "基建", "交运", "物流", "养殖",
    "畜牧", "中药", "疫苗", "脑机", "算力", "储能", "锂电", "稀土", "钢铁", "煤炭",
)
COMMODITY_KEYS = ("黄金", "白银", "原油", "豆粕", "有色金属", "能源化工", "商品", "铜", "铝", "玉米")
OVERSEAS_KEYS = (
    "纳指", "纳斯达克", "标普", "恒生", "港股", "日经", "德国", "法国", "沙特", "越南",
    "印度", "亚太", "中概", "海外", "美国", "道琼", "韩国",
)
MONEY_KEYS = ("货币", "现金", "日利", "添益", "保证金")
BOND_KEYS = ("国债", "政金债", "信用债", "公司债", "地方债", "可转债", "短融", "债券")


COMPANIES = (
    "华泰柏瑞", "易方达", "汇添富", "景顺长城", "兴证全球", "前海开源", "西部利得", "创金合信",
    "西藏东财", "信达澳亚", "国联安", "嘉实", "华夏", "南方", "广发", "富国", "博时", "招商",
    "鹏华", "工银", "建信", "国泰", "华宝", "景顺", "万家", "永赢", "大成", "银华", "天弘",
    "平安", "华安", "中银", "交银", "兴全", "摩根", "国寿", "泰康", "中金", "东财", "华富",
    "浦银", "中欧", "融通", "诺安", "长盛", "长城", "农银", "财通", "海富通", "国投", "金鹰",
    "华商", "泓德", "中庚", "朱雀", "睿远", "工银瑞信", "海富通", "鹏扬", "方正富邦", "富国",
)


def product_key(name: str) -> str:
    """同一标的、不同基金公司视为同一只，保留成交额更高的那只。"""
    text = str(name)
    for company in sorted(set(COMPANIES), key=len, reverse=True):
        text = text.replace(company, "")
    text = text.replace("港股通", "港股")
    for token in ("联接基金", "联接", "ETF", "基金", "指数", "发起式", "LOF"):
        text = text.replace(token, "")
    text = "".join(text.split())
    return text or str(name)


def dedupe_products(frame: pd.DataFrame, name_col: str, amount_col: str) -> pd.DataFrame:
    ordered = frame.sort_values(amount_col, ascending=False)
    ordered = ordered.assign(_product=ordered[name_col].map(product_key))
    kept = ordered.drop_duplicates("_product", keep="first").drop(columns="_product")
    return kept


def categorize(name: str) -> str:
    if any(key in name for key in MONEY_KEYS):
        return "货币"
    if any(key in name for key in BOND_KEYS) or name.endswith("债ETF"):
        return "债券"
    if any(key in name for key in COMMODITY_KEYS):
        return "商品"
    if any(key in name for key in OVERSEAS_KEYS):
        return "跨境"
    if any(key in name for key in INDUSTRY_KEYS):
        return "行业主题"
    if any(key in name for key in BROAD_KEYS):
        return "宽基"
    if "债" in name:
        return "债券"
    return "行业主题"


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bars (
            code TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            volume REAL, amount REAL,
            PRIMARY KEY (code, date)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS instruments (
            code TEXT PRIMARY KEY,
            name TEXT,
            category TEXT,
            amount REAL,
            pct REAL,
            price REAL
        )
        """
    )
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS quotes (
            code TEXT PRIMARY KEY,
            quoted_on TEXT NOT NULL,
            price REAL NOT NULL,
            open REAL, high REAL, low REAL,
            volume REAL, amount REAL
        )
        """
    )
    return conn


def status() -> dict:
    with _lock:
        data = dict(_status)
    conn = connect()
    row = conn.execute("SELECT value FROM meta WHERE key='updated_at'").fetchone()
    count = conn.execute("SELECT COUNT(DISTINCT code) FROM bars").fetchone()[0]
    conn.close()
    data["updated_at"] = data["updated_at"] or (row[0] if row else None)
    data["symbols"] = count
    return data


def _set_status(**kwargs):
    with _lock:
        _status.update(kwargs)


def _rename_hist(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
    out = df.rename(columns=mapping)
    keep = [col for col in ("date", "open", "high", "low", "close", "volume", "amount") if col in out.columns]
    out = out[keep].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out.dropna(subset=["date", "close"])


def _fetch_hist(code: str, start: str, end: str) -> pd.DataFrame:
    _disable_system_proxy()
    import akshare as ak

    last_error = None
    for attempt in range(3):
        try:
            if code == BENCH_CODE:
                df = ak.stock_zh_index_daily(symbol="sh000300")
            else:
                df = ak.fund_etf_hist_sina(symbol=_sina_symbol(code))
            if df is None or df.empty:
                return pd.DataFrame()
            out = _rename_hist(df)
            if start:
                out = out[out["date"] >= datetime.strptime(start, "%Y%m%d").strftime("%Y-%m-%d")]
            if end:
                out = out[out["date"] <= datetime.strptime(end, "%Y%m%d").strftime("%Y-%m-%d")]
            return out
        except Exception as exc:  # noqa: BLE001 - 网络源失败要重试
            last_error = exc
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"{code} 拉取失败: {last_error}")


def _upsert_bars(conn: sqlite3.Connection, code: str, df: pd.DataFrame):
    if df.empty:
        return
    rows = [
        (
            code,
            row.date,
            float(row.open),
            float(row.high),
            float(row.low),
            float(row.close),
            None if pd.isna(row.volume) else float(row.volume),
            None if "amount" not in df.columns or pd.isna(row.amount) else float(row.amount),
        )
        for row in df.itertuples(index=False)
    ]
    conn.executemany(
        """
        INSERT INTO bars (code, date, open, high, low, close, volume, amount)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(code, date) DO UPDATE SET
            open=excluded.open, high=excluded.high, low=excluded.low,
            close=excluded.close, volume=excluded.volume, amount=excluded.amount
        """,
        rows,
    )


def _last_date(conn: sqlite3.Connection, code: str) -> str | None:
    row = conn.execute("SELECT MAX(date) FROM bars WHERE code=?", (code,)).fetchone()
    return row[0] if row and row[0] else None


def select_universe(spot: pd.DataFrame, limit: int, min_amount: float, categories: set[str]) -> pd.DataFrame:
    frame = spot.copy()
    frame["代码"] = frame["代码"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    frame["名称"] = frame["名称"].astype(str)
    frame["成交额"] = pd.to_numeric(frame.get("成交额"), errors="coerce").fillna(0)
    frame["最新价"] = pd.to_numeric(frame.get("最新价"), errors="coerce")
    frame["涨跌幅"] = pd.to_numeric(frame.get("涨跌幅"), errors="coerce")
    frame["分类"] = frame["名称"].map(categorize)
    picked = frame[(frame["成交额"] >= min_amount) & (frame["分类"].isin(categories))]
    picked = dedupe_products(picked, "名称", "成交额").head(limit)
    return picked


def refresh(limit: int = 120, min_amount: float = 20_000_000, categories: list[str] | None = None):
    _disable_system_proxy()
    cats = set(categories or ["宽基", "行业主题", "商品", "跨境"])
    _set_status(running=True, done=0, total=0, message="正在获取 ETF 列表", error=None)
    try:
        import akshare as ak

        spot = ak.fund_etf_category_sina(symbol="ETF基金")
        universe = select_universe(spot, limit, min_amount, cats)
        quoted_on = datetime.now(TZ).strftime("%Y-%m-%d")
        quotes = []
        for _, item in universe.iterrows():
            fields = _quote_from_row(item)
            if fields:
                quotes.append((str(item["代码"]), fields))
        try:
            bench_row = _bench_spot_row()
            bench_fields = _quote_from_row(bench_row) if bench_row is not None else None
            if bench_fields:
                quotes.append((BENCH_CODE, bench_fields))
        except Exception:
            pass
        conn = connect()
        _save_quotes(conn, quoted_on, quotes)
        conn.commit()
        conn.execute("DELETE FROM instruments")
        conn.executemany(
            "INSERT INTO instruments (code, name, category, amount, pct, price) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (row.代码, row.名称, row.分类, float(row.成交额), None if pd.isna(row.涨跌幅) else float(row.涨跌幅), None if pd.isna(row.最新价) else float(row.最新价))
                for row in universe.itertuples(index=False)
            ],
        )
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('bench_name', '沪深300') ON CONFLICT(key) DO UPDATE SET value='沪深300'"
        )
        codes = [BENCH_CODE] + universe["代码"].tolist()
        _set_status(total=len(codes), done=0, message="开始下载日线")
        end = datetime.now(TZ).strftime("%Y%m%d")
        for index, code in enumerate(codes, start=1):
            name = "沪深300" if code == BENCH_CODE else universe.loc[universe["代码"] == code, "名称"].iloc[0]
            last = _last_date(conn, code)
            if last:
                start = (datetime.strptime(last, "%Y-%m-%d") - timedelta(days=10)).strftime("%Y%m%d")
            else:
                start = "20190101"
            _set_status(done=index - 1, message=f"正在更新 {code} {name}")
            frame = _fetch_hist(code, start, end)
            _upsert_bars(conn, code, frame)
            conn.commit()
            _set_status(done=index)
            time.sleep(0.15)
        updated = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('updated_at', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (updated,),
        )
        conn.commit()
        conn.close()
        _set_status(running=False, message="更新完成", updated_at=updated, error=None)
    except Exception as exc:  # noqa: BLE001
        _set_status(running=False, message="更新中断", error=str(exc))


def start_refresh(**kwargs) -> bool:
    with _lock:
        if _status["running"]:
            return False
        _status["running"] = True
        _status["message"] = "准备更新"
    thread = threading.Thread(target=refresh, kwargs=kwargs, daemon=True)
    thread.start()
    return True


def _positive(value):
    if value is None or value <= 0:
        return None
    return value


def _optional_float(value):
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return None
    return float(number)


def _align_volume(volume, history: pd.DataFrame):
    """新浪列表的成交量有时按手，日线按股。差出两个数量级时换成股。"""
    if volume is None or volume <= 0 or history.empty or "volume" not in history.columns:
        return volume
    recent = pd.to_numeric(history["volume"], errors="coerce").tail(20)
    recent = recent[recent > 0]
    if recent.empty:
        return volume
    median = float(recent.median())
    if median <= 0:
        return volume
    if volume / median < 0.05 and 0.05 <= (volume * 100) / median <= 20:
        return volume * 100
    return volume


def apply_quote(bars: pd.DataFrame, quote: dict | None) -> pd.DataFrame:
    """把更新时刻的价格并进最后一根 K 线，供买卖信号使用。"""
    if bars is None or bars.empty or not quote:
        return bars
    price = quote.get("price")
    day = quote.get("quoted_on")
    if price is None or price <= 0 or not day:
        return bars
    out = bars.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    last = out["date"].iloc[-1]
    open_ = quote.get("open") or price
    high = max(value for value in (quote.get("high"), price, open_) if value)
    low = min(value for value in (quote.get("low"), price, open_) if value)
    volume = _align_volume(quote.get("volume"), out)
    if last == day:
        idx = out.index[-1]
        out.loc[idx, "close"] = price
        out.loc[idx, "high"] = max(float(out.loc[idx, "high"]), high, price)
        out.loc[idx, "low"] = min(float(out.loc[idx, "low"]), low, price)
        if volume is not None:
            out.loc[idx, "volume"] = volume
        if quote.get("amount") is not None and "amount" in out.columns:
            out.loc[idx, "amount"] = quote["amount"]
        return out
    if last > day or datetime.strptime(day, "%Y-%m-%d").weekday() >= 5:
        return out
    if volume is None and pd.notna(out["volume"].iloc[-1]):
        volume = float(out["volume"].iloc[-1])
    row = {
        "date": day,
        "open": open_,
        "high": high,
        "low": low,
        "close": price,
        "volume": 0.0 if volume is None else volume,
    }
    if "amount" in out.columns:
        row["amount"] = quote.get("amount")
    return pd.concat([out, pd.DataFrame([row])], ignore_index=True)


def _quote_from_row(row) -> dict | None:
    def field(*names):
        for name in names:
            if name in row.index:
                value = _optional_float(row[name])
                if value is not None:
                    return value
        return None

    price = field("最新价", "price")
    if price is None or price <= 0:
        return None
    return {
        "price": price,
        "open": field("今开", "open"),
        "high": field("最高", "high"),
        "low": field("最低", "low"),
        "volume": _positive(field("成交量", "volume")),
        "amount": _positive(field("成交额", "amount")),
    }


def _read_quote(conn: sqlite3.Connection, code: str) -> dict | None:
    row = conn.execute(
        "SELECT quoted_on, price, open, high, low, volume, amount FROM quotes WHERE code=?",
        (code,),
    ).fetchone()
    if not row or row[1] is None:
        return None
    return {
        "quoted_on": row[0],
        "price": row[1],
        "open": row[2],
        "high": row[3],
        "low": row[4],
        "volume": row[5],
        "amount": row[6],
    }


def _save_quotes(conn: sqlite3.Connection, quoted_on: str, quotes: list[tuple]):
    conn.execute("DELETE FROM quotes")
    conn.executemany(
        """
        INSERT INTO quotes (code, quoted_on, price, open, high, low, volume, amount)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [(code, quoted_on, item["price"], item["open"], item["high"], item["low"], item["volume"], item["amount"]) for code, item in quotes],
    )


def _bench_spot_row():
    import akshare as ak

    frame = ak.stock_zh_index_spot_em(symbol="沪深重要指数")
    if frame is None or frame.empty or "代码" not in frame.columns:
        return None
    code = frame["代码"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    hit = frame.loc[code == BENCH_CODE]
    if hit.empty:
        return None
    return hit.iloc[0]


def load_bars(code: str) -> pd.DataFrame:
    conn = connect()
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume, amount FROM bars WHERE code=? ORDER BY date",
        conn,
        params=(code,),
    )
    quote = _read_quote(conn, code)
    conn.close()
    return apply_quote(df, quote)


def load_instruments() -> pd.DataFrame:
    conn = connect()
    df = pd.read_sql_query("SELECT code, name, category, amount, pct, price FROM instruments ORDER BY amount DESC", conn)
    conn.close()
    if df.empty:
        return df
    return dedupe_products(df, "name", "amount")


def maybe_autostart():
    now = datetime.now(TZ)
    if now.weekday() >= 5 or now.hour < 16:
        return
    info = status()
    updated = info.get("updated_at") or ""
    if updated.startswith(now.strftime("%Y-%m-%d")):
        return
    if info["symbols"] == 0:
        return
    start_refresh()
