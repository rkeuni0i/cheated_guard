"""액션 매트릭스 + 실행 (PRD 7.1, 7.2).

등급 → 액션 매핑. 실행 순서(PRD 7.2): 감사로그 → (R4)카드정지 → 알림.
정지를 알림보다 먼저 하는 이유: 알림 먼저면 사용자가 앱 여는 동안 추가 부정
결제가 진행될 수 있다.
"""
from __future__ import annotations

from datetime import timedelta
from typing import NamedTuple

from ..db import DB, now_kst
from ..models import Assessment, Transaction
from . import card_state
from .notifier import build_message, build_sms


class ActionPlan(NamedTuple):
    feed: bool          # 인앱 피드 노출
    notify: bool        # 소리·진동 푸시
    silent: bool        # 무음 푸시
    stepup: bool        # 본인확인 요청
    freeze: bool        # 카드 자동 정지
    sms: bool           # SMS 백업


ACTION_MATRIX = {
    "R0": ActionPlan(False, False, False, False, False, False),
    "R1": ActionPlan(True, False, False, False, False, False),
    "R2": ActionPlan(True, False, True, False, False, False),
    "R3": ActionPlan(True, True, False, True, False, False),
    "R4": ActionPlan(True, True, False, True, True, True),
}


def execute_actions(db: DB, assessment: Assessment, tx: Transaction) -> list[dict]:
    """등급에 따른 액션 실행. 실행된 액션 목록 반환."""
    plan = ACTION_MATRIX[assessment.grade]
    taken: list[dict] = []

    # 1) 감사 로그 먼저
    db.audit("SYSTEM", f"ASSESS_{assessment.grade}", "transaction",
             str(assessment.tx_id),
             {"score": assessment.score,
              "reasons": [r.to_dict() for r in assessment.reasons]})

    # 2) 카드 정지 (R4, 알림보다 먼저)
    if plan.freeze:
        state = card_state.freeze(
            db, tx.card_id, "SYSTEM", assessment.assessment_id,
            reason=assessment.reasons[0].label if assessment.reasons else "R4")
        taken.append({"action": "CARD_FREEZE", "status": state})

    # 3) 알림
    if plan.notify or plan.silent:
        channel = "PUSH" if plan.notify else "SILENT_PUSH"
        expires = now_kst() + timedelta(minutes=15)
        alert_id = db.save_alert(assessment.assessment_id, assessment.user_id,
                                 channel, build_message(assessment, tx), expires)
        taken.append({"action": "NOTIFY", "channel": channel, "alert_id": alert_id})
    if plan.sms:
        taken.append({"action": "SMS", "body": build_sms(assessment, tx)})
    if plan.stepup:
        taken.append({"action": "STEP_UP",
                      "expires_at": (now_kst() + timedelta(minutes=15)).isoformat()})

    db.commit()
    return taken
