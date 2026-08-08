"""데이터 품질 규칙 DQ-01~08 (PRD 4.6).

검증 실패 시 (ok=False, code, message)를 반환한다. 호출부(수집 API)는
실패 건을 DLQ로 보내고 판정을 스킵한다.
"""
from __future__ import annotations

from datetime import timedelta
from typing import NamedTuple

from ..db import now_kst
from ..models import Transaction


class DQResult(NamedTuple):
    ok: bool
    code: str = ""
    message: str = ""


VALID_CHANNELS = {"ONLINE", "OFFLINE", "OVERSEAS"}
VALID_TX_TYPES = {"APPROVAL", "CANCEL", "PARTIAL_CANCEL"}


def validate(tx: Transaction) -> DQResult:
    """단건 거래 검증. 첫 위반에서 즉시 반환한다."""
    # DQ-01: amount > 0, KRW는 정수
    if tx.amount is None or tx.amount <= 0:
        return DQResult(False, "DQ-01", "amount must be > 0")
    if tx.currency == "KRW" and int(tx.amount) != tx.amount:
        return DQResult(False, "DQ-01", "KRW amount must be integer")

    # DQ-02: approved_at이 미래이거나 5년 이전이면 거부
    now = now_kst()
    if tx.approved_at > now + timedelta(minutes=5):
        return DQResult(False, "DQ-02", "approved_at is in the future")
    if tx.approved_at < now - timedelta(days=365 * 5):
        return DQResult(False, "DQ-02", "approved_at older than 5 years")

    # 채널·타입 열거값
    if tx.channel not in VALID_CHANNELS:
        return DQResult(False, "DQ-CH", f"invalid channel: {tx.channel}")
    if tx.tx_type not in VALID_TX_TYPES:
        return DQResult(False, "DQ-TX", f"invalid tx_type: {tx.tx_type}")

    return DQResult(True)
