"""Watchlist — CRUD data/watchlist.json cho dashboard (tab Watchlist).

Định dạng tệp:
    {
      "version": 1,
      "updated_at": "2026-09-29T10:00:00+07:00",
      "watchlist": [{"ticker": "TCB", "added_at": "..."}, ...]
    }

Quy ước:
- Mã lưu HOA, khớp ^[A-Z0-9]{2,10}$ (HOSE chủ yếu mã 3 ký tự, nới 2–10 cho an toàn).
- Thêm trùng (so sánh không phân biệt hoa/thường) → không thêm, trả False.
- Ghi file nguyên tử (ghi tệp tạm rồi os.replace) — app chết giữa chừng không hỏng file.
- File JSON hỏng/sai cấu trúc → raise WatchlistError, KHÔNG tự xoá/khôi phục
  (tránh mất dữ liệu người dùng một cách âm thầm).
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Thư mục data/ của dự án.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_PATH = DATA_DIR / "watchlist.json"

TICKER_RE = re.compile(r"^[A-Z0-9]{2,10}$")


class WatchlistError(RuntimeError):
    """Tệp watchlist hỏng/không đọc được/sai cấu trúc."""


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Watchlist:
    """CRUD watchlist trên một tệp JSON (mặc định data/watchlist.json)."""

    def __init__(self, path: Path | str = DEFAULT_PATH) -> None:
        self._path = Path(path)

    # ---- tra cứu ----
    def entries(self) -> list[dict[str, str]]:
        """Danh sách mục {"ticker", "added_at"} theo thứ tự đã thêm."""
        return [dict(entry) for entry in self._load()["watchlist"]]

    def list_tickers(self) -> list[str]:
        """Danh sách mã (HOA) theo thứ tự đã thêm."""
        return [entry["ticker"] for entry in self.entries()]

    def contains(self, ticker: str) -> bool:
        """Mã đã có trong watchlist chưa (không phân biệt hoa/thường)."""
        return str(ticker).strip().upper() in set(self.list_tickers())

    # ---- ghi ----
    def add(self, ticker: str) -> bool:
        """Thêm mã. Trả True nếu thêm mới; False nếu đã có; ValueError nếu mã không hợp lệ."""
        cleaned = self._validate(ticker)
        data = self._load()
        if any(entry["ticker"] == cleaned for entry in data["watchlist"]):
            return False
        data["watchlist"].append({"ticker": cleaned, "added_at": _now_iso()})
        data["updated_at"] = _now_iso()
        self._save(data)
        logger.info("Watchlist: thêm %s", cleaned)
        return True

    def remove(self, ticker: str) -> bool:
        """Xoá mã. Trả True nếu xoá được; False nếu mã không có trong watchlist."""
        cleaned = self._validate(ticker)
        data = self._load()
        kept = [entry for entry in data["watchlist"] if entry["ticker"] != cleaned]
        if len(kept) == len(data["watchlist"]):
            return False
        data["watchlist"] = kept
        data["updated_at"] = _now_iso()
        self._save(data)
        logger.info("Watchlist: xoá %s", cleaned)
        return True

    # ---- nội bộ ----
    @staticmethod
    def _validate(ticker: str) -> str:
        cleaned = str(ticker).strip().upper()
        if not TICKER_RE.match(cleaned):
            raise ValueError(f"Mã cổ phiếu không hợp lệ: {ticker!r} (chỉ A–Z/0–9, 2–10 ký tự)")
        return cleaned

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"version": 1, "updated_at": _now_iso(), "watchlist": []}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WatchlistError(f"Tệp watchlist hỏng hoặc không đọc được ({self._path}): {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("watchlist"), list):
            raise WatchlistError(f"Tệp watchlist sai cấu trúc ({self._path})")
        return data

    def _save(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)


__all__ = ["DATA_DIR", "DEFAULT_PATH", "Watchlist", "WatchlistError"]
