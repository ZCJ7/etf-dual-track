"""把看板算成静态文件，供 GitHub Pages 给手机打开。"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from app.data import ROOT, refresh
from app.main import _chart_payload, build_dashboard

SITE = ROOT / "site"
STATIC = ROOT / "static"


def _scrub(value):
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, dict):
        return {key: _scrub(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    return value


def write_site() -> None:
    board = _scrub(build_dashboard())
    if not board.get("ready"):
        raise SystemExit("没有可发布的行情。先更新数据再发布。")
    if SITE.exists():
        shutil.rmtree(SITE)
    (SITE / "static").mkdir(parents=True)
    (SITE / "charts").mkdir()
    for name in ("index.html", "styles.css", "app.js", "lightweight-charts.js"):
        shutil.copy(STATIC / name, SITE / "static" / name)
    shutil.copy(SITE / "static" / "index.html", SITE / "index.html")
    codes = {row["code"] for row in board.get("right", [])}
    by_code = {row["code"]: row for row in board.get("right", [])}
    for code in sorted(codes):
        payload = _chart_payload(code)
        payload["signal"] = by_code.get(code)
        payload = _scrub(payload)
        (SITE / "charts" / f"{code}.json").write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
    (SITE / "dashboard.json").write_text(
        json.dumps(board, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    (SITE / ".nojekyll").write_text("", encoding="utf-8")
    print(f"wrote {SITE} symbols={len(codes)}")


def main() -> None:
    if os.environ.get("PUBLISH_REFRESH", "1") != "0":
        refresh(limit=120, min_amount=20_000_000)
    write_site()


if __name__ == "__main__":
    main()
