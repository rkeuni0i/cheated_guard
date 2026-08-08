"""수집 파이프라인 (PRD 3.2 ①, 3.3.B 1단계).

collector → validator → normalizer → dedup(멱등) → store.
FastAPI 대신 순수 함수로 구현했으나 책임 분리는 동일하다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..db import DB
from ..models import Transaction
from .normalizer import build_merchant, normalize_transaction
from .validator import DQResult, validate


@dataclass
class IngestResult:
    ok: bool
    tx_id: Optional[int] = None
    dq: Optional[DQResult] = None
    duplicate: bool = False
    cancelled: bool = False


def ingest(db: DB, tx: Transaction, *, learn_only: bool = False) -> IngestResult:
    """거래 1건을 수집·저장한다.

    learn_only=True는 과거 이력 일괄 적재(batch)용으로 판정을 트리거하지 않음을
    호출부에 알리는 신호일 뿐, 저장 로직은 동일하다.
    """
    # 1) 검증 (DQ)
    dq = validate(tx)
    if not dq.ok:
        db.audit("SYSTEM", "DLQ", "transaction", tx.approval_no,
                 {"code": dq.code, "message": dq.message})
        return IngestResult(ok=False, dq=dq)

    # 2) 정규화 + 가맹점 해석
    normalize_transaction(tx)
    tx.merchant_id = db.resolve_merchant(build_merchant(tx))

    # 3) 취소 거래 처리 (DQ-05): 원거래를 찾아 학습 제외
    if tx.tx_type in ("CANCEL", "PARTIAL_CANCEL"):
        orig = db.conn.execute(
            "SELECT tx_id FROM transactions WHERE card_id=? AND approval_no=? "
            "AND tx_type='APPROVAL' ORDER BY approved_at DESC LIMIT 1",
            (tx.card_id, tx.original_approval_no or tx.approval_no)).fetchone()
        if orig:
            db.mark_excluded(orig["tx_id"])
        return IngestResult(ok=True, cancelled=True)

    # 4) 멱등 삽입 (DQ-03)
    tx_id = db.insert_transaction(tx)
    if tx_id is None:
        return IngestResult(ok=True, duplicate=True)
    tx.tx_id = tx_id
    return IngestResult(ok=True, tx_id=tx_id)
