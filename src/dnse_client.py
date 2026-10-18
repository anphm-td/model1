"""Client DNSE LightSpeed API — CHỈ ĐỌC (không bao giờ đặt/sửa/hủy lệnh).

Đọc tài khoản, vị thế (deal nắm giữ) và sổ lệnh của một tiểu khoản qua
LightSpeed API V1 (trước KRX) — "II. Trading API", tài liệu chính thức
https://hdsd.dnse.com.vn (LightSpeed API trước KRX → Trading API):

- Số dư tài khoản:  GET https://services.entrade.com.vn/dnse-order-service/account-balances/{account}
- Deal nắm giữ:     GET https://services.entrade.com.vn/dnse-deal-service/deals?accountNo={account}
- Sổ lệnh:          GET https://services.entrade.com.vn/dnse-order-service/v2/orders?accountNo={account}
- Xác thực:         header ``Authorization: Bearer {jwt_token}``

QUY ĐỔI ĐƠN VỊ: REST trả GIÁ/TIỀN theo VND THÔ (trong khi toàn bộ app hiển thị
NGHÌN ĐỒNG — cùng quy ước với TradingView/DNSE trên giao diện). Client chuẩn
hóa mọi trường tiền/giá về NGHÌN ĐỒNG (chia 1000); KHỐI LƯỢNG (cổ phiếu) giữ
nguyên. ``unit``/``price_unit`` trong kết quả ghi rõ đơn vị để caller không nhầm.

LƯU Ý QUAN TRỌNG:
- Module này CHỈ có các hàm ĐỌC. KHÔNG có — và sẽ không thêm — bất kỳ hàm
  đặt/sửa/hủy lệnh nào (tồn tại trong API nhưng ngoài phạm vi app dự báo).
- Token nằm trong config.yaml (app.dnse.jwt_token) — KHÔNG hardcode, KHÔNG log
  token, KHÔNG đưa token vào message lỗi.
- Token rỗng → client ở trạng thái disabled (``enabled=False``); mọi hàm đọc
  raise ``DnseDisabledError`` TRƯỚC KHI tạo bất kỳ request mạng nào.
"""
from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

__all__ = [
    "DnseAuthError",
    "DnseClient",
    "DnseDisabledError",
    "DnseError",
    "DnseNetworkError",
    "VND_PER_THOUSAND",
]

#: 1000 VND thô = 1 nghìn đồng (đơn vị hiển thị chuẩn của app).
VND_PER_THOUSAND: float = 1000.0


class DnseError(Exception):
    """Lỗi chung của DNSE API (HTTP >= 400 khác 401, body không parse được...)."""


class DnseDisabledError(DnseError):
    """Client chưa đủ thông tin (token rỗng / thiếu số tiểu khoản) — không gọi mạng."""


class DnseAuthError(DnseError):
    """HTTP 401 — token hết hạn hoặc không hợp lệ (cần đăng nhập lại LightSpeed)."""


class DnseNetworkError(DnseError):
    """Lỗi mạng (timeout, DNS, connection reset...) khi gọi DNSE."""


def _to_thousand(value: Any) -> float | None:
    """VND thô → nghìn đồng (float/None). None/NaN/không ép số được → None."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number / VND_PER_THOUSAND


def _to_int(value: Any) -> int | None:
    """Khối lượng cổ phiếu (KHÔNG chia 1000). None/không ép số được → None."""
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


class DnseClient:
    """Client đọc-only DNSE LightSpeed API cho một tiểu khoản.

    Args:
        jwt_token: JWT từ bước đăng nhập LightSpeed (đọc từ config, không hardcode).
        account_no: mã tiểu khoản (vd. ``0001234567``).
        timeout: thời gian chờ mỗi request (giây).

    Ghi chú: không có phương thức đặt/sửa/hủy lệnh — client này CHỈ ĐỌC.
    """

    BASE_URL = "https://services.entrade.com.vn"
    PATH_ACCOUNT_BALANCE = "/dnse-order-service/account-balances/{account}"
    PATH_DEALS = "/dnse-deal-service/deals"
    PATH_ORDERS = "/dnse-order-service/v2/orders"

    def __init__(
        self,
        jwt_token: str,
        account_no: str,
        *,
        timeout: float = 10.0,
        session: requests.Session | None = None,
    ) -> None:
        self.jwt_token = str(jwt_token or "").strip()
        self.account_no = str(account_no or "").strip()
        self.timeout = float(timeout)
        self._session = session if session is not None else requests.Session()

    # ------------------------------------------------------------------
    # Trạng thái
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """True khi đủ token + số tiểu khoản — điều kiện để dashboard render."""
        return bool(self.jwt_token) and bool(self.account_no)

    def _require_enabled(self) -> None:
        if not self.jwt_token:
            raise DnseDisabledError(
                "DNSE chưa cấu hình token (app.dnse.jwt_token rỗng) — không gọi API."
            )
        if not self.account_no:
            raise DnseDisabledError(
                "DNSE chưa cấu hình số tiểu khoản (app.dnse.account_no rỗng) — không gọi API."
            )

    # ------------------------------------------------------------------
    # Tầng HTTP (duy nhất chỗ đụng mạng — chỉ GET)
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, *, params: dict[str, Any] | None = None) -> Any:
        """Gọi một endpoint đọc (GET) và trả JSON đã parse.

        Lỗi được chuẩn hóa: token rỗng → DnseDisabledError (không gọi mạng);
        401 → DnseAuthError; HTTP khác >= 400 → DnseError; lỗi mạng →
        DnseNetworkError. Token KHÔNG bao giờ xuất hiện trong message lỗi/log.
        """
        self._require_enabled()
        url = self.BASE_URL + path
        headers = {
            "Authorization": f"Bearer {self.jwt_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        logger.info("DNSE GET %s (account=%s)", path, self.account_no)  # KHÔNG log token
        try:
            response = self._session.request(
                method, url, headers=headers, params=params, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise DnseNetworkError(
                f"Không kết nối được DNSE API ({type(exc).__name__}) — kiểm tra mạng rồi thử lại."
            ) from exc

        if response.status_code == 401:
            raise DnseAuthError(
                "DNSE từ chối truy cập (HTTP 401) — token hết hạn hoặc không hợp lệ, "
                "hãy đăng nhập lại LightSpeed để lấy token mới."
            )
        if response.status_code >= 400:
            body = (response.text or "")[:300]
            raise DnseError(f"DNSE API trả HTTP {response.status_code} cho {path}: {body}")
        if response.status_code == 204 or not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise DnseError(f"DNSE API trả body không phải JSON cho {path}.") from exc

    # ------------------------------------------------------------------
    # Endpoint ĐỌC 1 — số dư tài khoản
    # ------------------------------------------------------------------

    def get_account_summary(self) -> dict[str, Any]:
        """Số dư tài khoản — tiền đã chuẩn hóa về NGHÌN ĐỒNG (REST trả VND thô)."""
        self._require_enabled()
        data = self._request("GET", self.PATH_ACCOUNT_BALANCE.format(account=self.account_no))
        data = data or {}
        return {
            "account_no": str(data.get("investorAccountId") or self.account_no),
            "custody_code": data.get("custodyCode"),
            "total_cash": _to_thousand(data.get("totalCash")),
            "available_cash": _to_thousand(data.get("availableCash")),
            "withdrawable_cash": _to_thousand(data.get("withdrawableCash")),
            "receiving_amount": _to_thousand(data.get("receivingAmount")),
            "secure_amount": _to_thousand(data.get("secureAmount")),
            "stock_value": _to_thousand(data.get("stockValue")),
            "net_asset_value": _to_thousand(data.get("netAssetValue")),
            "total_debt": _to_thousand(data.get("totalDebt")),
            "unit": "nghìn đồng",
        }

    # ------------------------------------------------------------------
    # Endpoint ĐỌC 2 — vị thế (deal nắm giữ)
    # ------------------------------------------------------------------

    def get_positions(self, *, only_open: bool = True) -> list[dict[str, Any]]:
        """Deal nắm giữ — giá về NGHÌN ĐỒNG, khối lượng giữ nguyên (cổ phiếu).

        Args:
            only_open: chỉ lấy deal ``status=OPEN`` (vị thế đang nắm giữ); False
                trả cả deal đã đóng (mỗi dict vẫn có ``status`` để phân biệt).
        """
        self._require_enabled()
        data = self._request("GET", self.PATH_DEALS, params={"accountNo": self.account_no})
        deals = (data or {}).get("deals") or []
        positions: list[dict[str, Any]] = []
        for deal in deals:
            if not isinstance(deal, dict):
                continue
            status = str(deal.get("status") or "").strip().upper()
            if only_open and status and status != "OPEN":
                continue
            positions.append(
                {
                    "symbol": str(deal.get("symbol") or "").strip().upper(),
                    "status": status or None,
                    "side": deal.get("side"),
                    # KL nắm giữ = tích lũy mua − đã đóng (cổ phiếu, không quy đổi)
                    "quantity": _accumulate_quantity(deal),
                    "accumulate_quantity": _to_int(deal.get("accumulateQuantity")),
                    "closed_quantity": _to_int(deal.get("closedQuantity")),
                    "avg_cost_price": _to_thousand(deal.get("averageCostPrice")),
                    "market_price": _to_thousand(deal.get("marketPrice")),
                    "break_even_price": _to_thousand(deal.get("breakEvenPrice")),
                    "unrealized_profit": _to_thousand(deal.get("unrealizedProfit")),
                    "price_unit": "nghìn đồng",
                }
            )
        return positions

    # ------------------------------------------------------------------
    # Endpoint ĐỌC 3 — sổ lệnh
    # ------------------------------------------------------------------

    def get_orders(self) -> list[dict[str, Any]]:
        """Sổ lệnh của tiểu khoản — giá về NGHÌN ĐỒNG, khối lượng giữ nguyên.

        Lưu ý: chỉ liệt kê (GET) — KHÔNG có hàm đặt/sửa/hủy lệnh trong client này.
        """
        self._require_enabled()
        data = self._request("GET", self.PATH_ORDERS, params={"accountNo": self.account_no})
        orders = (data or {}).get("orders") or []
        normalized: list[dict[str, Any]] = []
        for order in orders:
            if not isinstance(order, dict):
                continue
            normalized.append(
                {
                    "order_id": _to_int(order.get("id")),
                    "symbol": str(order.get("symbol") or "").strip().upper(),
                    "side": order.get("side"),  # NB = mua, NS = bán (mã gốc DNSE)
                    "order_type": order.get("orderType"),
                    "order_status": order.get("orderStatus"),
                    "quantity": _to_int(order.get("quantity")),
                    "fill_quantity": _to_int(order.get("fillQuantity")),
                    "leave_quantity": _to_int(order.get("leaveQuantity")),
                    "price": _to_thousand(order.get("price")),
                    "average_price": _to_thousand(order.get("averagePrice")),
                    "trans_date": order.get("transDate"),
                    "created_date": order.get("createdDate"),
                    "price_unit": "nghìn đồng",
                }
            )
        return normalized


def _accumulate_quantity(deal: dict[str, Any]) -> int | None:
    """KL đang nắm giữ = accumulateQuantity − closedQuantity (None nếu thiếu)."""
    accumulate = _to_int(deal.get("accumulateQuantity"))
    if accumulate is None:
        return None
    closed = _to_int(deal.get("closedQuantity")) or 0
    return accumulate - closed
