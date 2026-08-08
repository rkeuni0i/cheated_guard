"""카드 상태 머신 (PRD 7.4).

ACTIVE ⇄ FREEZING → FROZEN/FREEZE_FAILED, FROZEN → UNFREEZE_PENDING → ACTIVE,
FROZEN → REPORTED(단방향 종단). 정지 API 호출은 멱등키로 보호한다.
자족형이므로 카드사 API는 Mock으로 항상 성공 처리한다(RK-05).
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from ..db import DB, now_kst

VALID_STATES = {"ACTIVE", "FREEZING", "FROZEN", "FREEZE_FAILED",
                "UNFREEZE_PENDING", "REPORTED"}

# 허용 전이 (from -> {to})
_TRANSITIONS = {
    "ACTIVE": {"FREEZING"},
    "FREEZING": {"FROZEN", "FREEZE_FAILED"},
    "FROZEN": {"UNFREEZE_PENDING", "REPORTED", "ACTIVE"},
    "FREEZE_FAILED": {"FROZEN", "ACTIVE"},
    "UNFREEZE_PENDING": {"ACTIVE", "FROZEN"},
    "REPORTED": set(),
}


class CardStateError(Exception):
    pass


def _can(frm: str, to: str) -> bool:
    return to in _TRANSITIONS.get(frm, set())


def _mock_issuer_freeze(card_id: str, idempotency_key: str) -> bool:
    """Mock 카드사 정지 API. 실제로는 HTTP 호출 + 멱등키."""
    return True


def _mock_issuer_unfreeze(card_id: str, idempotency_key: str) -> bool:
    return True


def freeze(db: DB, card_id: str, triggered_by: str,
           assessment_id: Optional[int] = None, reason: str = "") -> str:
    """카드 정지. 성공 시 FROZEN, 실패 시 FREEZE_FAILED 반환."""
    card = db.get_card(card_id)
    if not card:
        raise CardStateError(f"unknown card {card_id}")
    state = card["status"]
    if state == "FROZEN":
        return "FROZEN"                 # 멱등: 이미 정지됨
    if not _can(state, "FREEZING"):
        raise CardStateError(f"cannot freeze from {state}")

    ok = _mock_issuer_freeze(card_id, f"frz-{assessment_id}")
    if ok:
        db.set_card_status(card_id, "FROZEN", frozen_at=now_kst())
        db.record_card_action(card_id, "FREEZE", triggered_by, "SUCCESS",
                              assessment_id, reason)
        db.audit(triggered_by, "CARD_FREEZE", "card", card_id,
                 {"reason": reason, "assessment_id": assessment_id})
        return "FROZEN"
    db.set_card_status(card_id, "FREEZE_FAILED")
    db.record_card_action(card_id, "FREEZE", triggered_by, "FAILED",
                          assessment_id, reason)
    db.audit("SYSTEM", "FREEZE_FAILED", "card", card_id, {"assessment_id": assessment_id})
    return "FREEZE_FAILED"


def unfreeze(db: DB, card_id: str, triggered_by: str, two_fa_ok: bool) -> str:
    """카드 해제. 2FA 필수 (PRD 7.4 UNFREEZE_PENDING → ACTIVE)."""
    card = db.get_card(card_id)
    if not card:
        raise CardStateError(f"unknown card {card_id}")
    if card["status"] not in ("FROZEN", "UNFREEZE_PENDING", "FREEZE_FAILED"):
        raise CardStateError(f"cannot unfreeze from {card['status']}")
    if not two_fa_ok:
        db.set_card_status(card_id, "UNFREEZE_PENDING")
        return "UNFREEZE_PENDING"
    _mock_issuer_unfreeze(card_id, f"unf-{card_id}")
    db.set_card_status(card_id, "ACTIVE", frozen_at=None)
    db.record_card_action(card_id, "UNFREEZE", triggered_by, "SUCCESS")
    db.audit(triggered_by, "CARD_UNFREEZE", "card", card_id, None)
    return "ACTIVE"


def report_lost(db: DB, card_id: str, triggered_by: str = "USER") -> str:
    """도난·분실 신고 → REPORTED (단방향 종단)."""
    card = db.get_card(card_id)
    if card and card["status"] != "ACTIVE" or card is None:
        pass
    db.set_card_status(card_id, "REPORTED")
    db.record_card_action(card_id, "REPORT", triggered_by, "SUCCESS")
    db.audit(triggered_by, "CARD_REPORTED", "card", card_id, None)
    return "REPORTED"
