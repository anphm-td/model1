"""Kiểm thử ``src.dnse_client`` — mock HTTP bằng monkeypatch, KHÔNG gọi mạng thật.

Phủ yêu cầu:
- chuẩn hóa vị thế: giá/tiền VND thô → NGHÌN ĐỒNG (chia 1000), khối lượng giữ
  nguyên, lọc deal OPEN, KL nắm giữ = accumulate − closed;
- HTTP 401 → DnseAuthError; HTTP 500 → DnseError; lỗi mạng → DnseNetworkError;
- token rỗng / thiếu tiểu khoản → DnseDisabledError và TUYỆT ĐỐI không tạo
  request mạng (fake request sẽ fail test nếu bị gọi);
- client CHỈ ĐỌC: không tồn tại hàm đặt/sửa/hủy lệnh.

Endpoint/field names đối chiếu với tài liệu chính thức hdsd.dnse.com.vn
(LightSpeed API trước KRX → II. Trading API): account-balances/{account},
dnse-deal-service/deals?accountNo=..., dnse-order-service/v2/orders?accountNo=...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.dnse_client as dnse_module  # noqa: E402
from src.dnse_client import (  # noqa: E402
    DnseAuthError,
    DnseClient,
    DnseDisabledError,
    DnseError,
    DnseNetworkError,
)

TOKEN = "test-jwt-token-abc"
ACCOUNT = "0001234567"


# ---------------------------------------------------------------------------
# Fake HTTP: monkeypatch requests.Session.request — không request nào ra ngoài
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: Any = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")
        self.content = self.text.encode("utf-8") if self.text else b""

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("No JSON")
        return self._payload


class RequestRecorder:
    """Ghi nhận mọi request + trả response cài sẵn; bị gọi khi không mong đợi → lỗi."""

    def __init__(self, response: FakeResponse | None = None, error: Exception | None = None):
        self.calls: list[dict[str, Any]] = []
        self.response = response
        self.error = error

    def __call__(self, method: str, url: str, **kwargs: Any):
        self.calls.append({"method": method, "url": url, "headers": kwargs.get("headers"),
                           "params": kwargs.get("params"), "timeout": kwargs.get("timeout")})
        if self.error is not None:
            raise self.error
        assert self.response is not None, "request bị gọi nhưng test không cài response"
        return self.response


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch):
    """Gắn recorder vào requests.Session.request cho cả test (auto-revert).

    Chấp nhận cả ``FakeResponse`` (bọc sẵn thành RequestRecorder) lẫn
    ``RequestRecorder``; luôn trả về recorder để test assert số lần gọi.
    """

    def install(source: FakeResponse | RequestRecorder) -> RequestRecorder:
        rec = source if isinstance(source, RequestRecorder) else RequestRecorder(response=source)

        def fake_request(session_self, method: str, url: str, **kwargs: Any):
            # hàm thường → binding chuẩn: session_self là Session thật
            return rec(method, url, **kwargs)

        monkeypatch.setattr(requests.Session, "request", fake_request)
        return rec

    return install


def make_client() -> DnseClient:
    return DnseClient(TOKEN, ACCOUNT)


# ---------------------------------------------------------------------------
# Chuẩn hóa vị thế
# ---------------------------------------------------------------------------


DEALS_PAYLOAD = {
    "deals": [
        {
            "id": 1, "symbol": "tcb", "accountNo": ACCOUNT, "status": "OPEN", "side": "NB",
            "accumulateQuantity": 1000, "closedQuantity": 200,
            "averageCostPrice": 24500.0, "marketPrice": 25200.0,
            "breakEvenPrice": 24600.0, "unrealizedProfit": 560000.0,
        },
        {
            "id": 2, "symbol": "MBB", "accountNo": ACCOUNT, "status": "CLOSED", "side": "NB",
            "accumulateQuantity": 500, "closedQuantity": 500,
            "averageCostPrice": 21000.0, "marketPrice": 22000.0, "unrealizedProfit": 0.0,
        },
    ]
}


def test_get_positions_normalizes_prices_to_thousand(recorder):
    rec = recorder(FakeResponse(payload=DEALS_PAYLOAD))
    positions = make_client().get_positions()

    # GET đúng endpoint tài liệu, đúng tham số tiểu khoản, đúng header Bearer
    assert len(rec.calls) == 1
    call = rec.calls[0]
    assert call["method"] == "GET"
    assert call["url"] == DnseClient.BASE_URL + "/dnse-deal-service/deals"
    assert call["params"] == {"accountNo": ACCOUNT}
    assert call["headers"]["Authorization"] == f"Bearer {TOKEN}"

    # only_open=True (mặc định): deal CLOSED bị lọc
    assert [p["symbol"] for p in positions] == ["TCB"]
    pos = positions[0]
    assert pos["status"] == "OPEN"
    # Giá: VND thô → nghìn đồng (chia 1000)
    assert pos["avg_cost_price"] == pytest.approx(24.5)
    assert pos["market_price"] == pytest.approx(25.2)
    assert pos["break_even_price"] == pytest.approx(24.6)
    assert pos["unrealized_profit"] == pytest.approx(560.0)
    # Khối lượng: KHÔNG chia 1000; KL nắm giữ = accumulate − closed
    assert pos["quantity"] == 800
    assert pos["accumulate_quantity"] == 1000
    assert pos["closed_quantity"] == 200
    assert pos["price_unit"] == "nghìn đồng"


def test_get_positions_only_open_false_keeps_closed_deals(recorder):
    recorder(FakeResponse(payload=DEALS_PAYLOAD))
    positions = make_client().get_positions(only_open=False)
    assert [p["status"] for p in positions] == ["OPEN", "CLOSED"]


def test_get_positions_missing_fields_are_none(recorder):
    recorder(FakeResponse(payload={"deals": [{"symbol": "FPT"}]}))
    positions = make_client().get_positions()
    assert positions == [
        {
            "symbol": "FPT", "status": None, "side": None, "quantity": None,
            "accumulate_quantity": None, "closed_quantity": None,
            "avg_cost_price": None, "market_price": None, "break_even_price": None,
            "unrealized_profit": None, "price_unit": "nghìn đồng",
        }
    ]


def test_get_positions_empty_response(recorder):
    recorder(FakeResponse(payload={"deals": []}))
    assert make_client().get_positions() == []


# ---------------------------------------------------------------------------
# Chuẩn hóa số dư tài khoản và sổ lệnh
# ---------------------------------------------------------------------------


def test_get_account_summary_normalizes_money_to_thousand(recorder):
    rec = recorder(FakeResponse(payload={
        "custodyCode": "CUST1", "investorAccountId": ACCOUNT,
        "totalCash": 500_000_000.0, "availableCash": 250_000_000.0,
        "withdrawableCash": 240_000_000.0, "stockValue": 700_000_000.0,
        "netAssetValue": 1_200_000_000.0, "receivingAmount": 10_000_000.0,
        "secureAmount": 5_000_000.0, "totalDebt": 0.0,
    }))
    summary = make_client().get_account_summary()

    assert rec.calls[0]["url"].endswith(f"/dnse-order-service/account-balances/{ACCOUNT}")
    assert summary["account_no"] == ACCOUNT
    assert summary["total_cash"] == pytest.approx(500_000.0)
    assert summary["available_cash"] == pytest.approx(250_000.0)
    assert summary["withdrawable_cash"] == pytest.approx(240_000.0)
    assert summary["stock_value"] == pytest.approx(700_000.0)
    assert summary["net_asset_value"] == pytest.approx(1_200_000.0)
    assert summary["unit"] == "nghìn đồng"


def test_get_orders_normalizes_prices_keeps_quantities(recorder):
    rec = recorder(FakeResponse(payload={
        "orders": [
            {"id": 9, "symbol": "VCB", "side": "NB", "orderType": "LO",
             "orderStatus": "filled", "quantity": 100, "fillQuantity": 100,
             "leaveQuantity": 0, "price": 58000.0, "averagePrice": 58100.0,
             "transDate": "2026-09-28", "createdDate": "2026-09-28T09:15:00.000+07:00"},
        ]
    }))
    orders = make_client().get_orders()

    assert rec.calls[0]["params"] == {"accountNo": ACCOUNT}
    assert rec.calls[0]["url"].endswith("/dnse-order-service/v2/orders")
    order = orders[0]
    assert order["order_id"] == 9
    assert order["side"] == "NB"  # mã gốc DNSE (NB mua / NS bán)
    assert order["quantity"] == 100  # khối lượng KHÔNG chia 1000
    assert order["price"] == pytest.approx(58.0)  # VND thô → nghìn đồng
    assert order["average_price"] == pytest.approx(58.1)
    assert order["order_status"] == "filled"


# ---------------------------------------------------------------------------
# Lỗi: 401, HTTP khác, mạng, body hỏng
# ---------------------------------------------------------------------------


def test_401_raises_dnse_auth_error(recorder):
    recorder(FakeResponse(status_code=401, payload={"message": "Unauthorized"}))
    with pytest.raises(DnseAuthError, match="401"):
        make_client().get_positions()
    with pytest.raises(DnseAuthError):
        make_client().get_account_summary()
    with pytest.raises(DnseAuthError):
        make_client().get_orders()


def test_http_500_raises_dnse_error(recorder):
    recorder(FakeResponse(status_code=500, text="Internal Server Error"))
    with pytest.raises(DnseError, match="500") as exc_info:
        make_client().get_positions()
    # Token KHÔNG bao giờ xuất hiện trong message lỗi
    assert TOKEN not in str(exc_info.value)


def test_network_error_raises_dnse_network_error(recorder):
    recorder(RequestRecorder(error=requests.ConnectionError("connection reset")))
    with pytest.raises(DnseNetworkError):
        make_client().get_positions()


def test_timeout_raises_dnse_network_error(recorder):
    recorder(RequestRecorder(error=requests.Timeout("timed out")))
    with pytest.raises(DnseNetworkError):
        make_client().get_orders()


def test_non_json_body_raises_dnse_error(recorder):
    recorder(FakeResponse(status_code=200, text="<html>gateway</html>"))
    with pytest.raises(DnseError, match="JSON"):
        make_client().get_account_summary()


def test_timeout_value_passed_to_requests(recorder):
    rec = recorder(FakeResponse(payload={"deals": []}))
    DnseClient(TOKEN, ACCOUNT, timeout=3.5).get_positions()
    assert rec.calls[0]["timeout"] == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# Token rỗng / thiếu tiểu khoản → disabled, KHÔNG gọi mạng
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("token", ["", "   "])
def test_empty_token_is_disabled_and_makes_no_request(recorder, token):
    rec = recorder(RequestRecorder(error=AssertionError("KHÔNG ĐƯỢC gọi mạng khi token rỗng")))
    client = DnseClient(token, ACCOUNT)
    assert client.enabled is False
    with pytest.raises(DnseDisabledError, match="token"):
        client.get_positions()
    with pytest.raises(DnseDisabledError):
        client.get_account_summary()
    with pytest.raises(DnseDisabledError):
        client.get_orders()
    assert rec.calls == []  # chắc chắn không request nào được tạo


def test_missing_account_no_is_disabled_and_makes_no_request(recorder):
    rec = recorder(RequestRecorder(error=AssertionError("KHÔNG ĐƯỢC gọi mạng khi thiếu account_no")))
    client = DnseClient(TOKEN, "")
    assert client.enabled is False
    with pytest.raises(DnseDisabledError):
        client.get_positions()
    assert rec.calls == []


def test_token_surrounding_whitespace_is_stripped(recorder):
    rec = recorder(FakeResponse(payload={"deals": []}))
    client = DnseClient(f"  {TOKEN}  ", f" {ACCOUNT} ")
    assert client.enabled is True
    client.get_positions()
    assert rec.calls[0]["headers"]["Authorization"] == f"Bearer {TOKEN}"


# ---------------------------------------------------------------------------
# Client CHỈ ĐỌC — không tồn tại hàm đặt/sửa/hủy lệnh
# ---------------------------------------------------------------------------


def test_client_has_no_order_writing_methods():
    forbidden = ("place", "cancel", "modify", "amend", "replace", "submit", "new_order")
    names = [name for name in dir(DnseClient) if not name.startswith("_")]
    assert not any(any(word in name.lower() for word in forbidden) for name in names), names
    # Chỉ có đúng 3 hàm đọc công khai
    assert sorted(n for n in names if callable(getattr(DnseClient, n)) and not isinstance(getattr(DnseClient, n), property)) == [
        "get_account_summary", "get_orders", "get_positions"
    ]


def test_module_exports_only_read_api():
    assert dnse_module.__all__ == [
        "DnseAuthError", "DnseClient", "DnseDisabledError", "DnseError", "DnseNetworkError",
        "VND_PER_THOUSAND",
    ]
