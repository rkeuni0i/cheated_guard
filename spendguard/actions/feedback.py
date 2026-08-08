"""피드백 루프 — 사용자 응답 처리 (PRD 7.5).

응답이 라벨이 되어 시스템을 개선한다. 무응답은 절대 '정상'으로 라벨링하지
않는다(학습 데이터 오염 방지, ADR-005).
"""
from __future__ import annotations

from datetime import timedelta

from ..config import SUPPRESS_DAYS
from ..db import DB, now_kst
from ..models import Transaction
from . import card_state
from .fusion_bands import amount_band


def respond(db: DB, alert_id: int, response: str) -> dict:
    """알림 응답 처리. response ∈ {CONFIRMED_MINE, NOT_MINE, REPORT_LOST}."""
    alert = db.conn.execute(
        "SELECT * FROM alerts WHERE alert_id=?", (alert_id,)).fetchone()
    if not alert:
        return {"ok": False, "error": "alert not found"}

    ra = db.conn.execute(
        "SELECT * FROM risk_assessments WHERE assessment_id=?",
        (alert["assessment_id"],)).fetchone()
    tx = db.conn.execute(
        "SELECT * FROM transactions WHERE tx_id=?", (ra["tx_id"],)).fetchone() \
        if ra and ra["tx_id"] else None

    db.conn.execute(
        "UPDATE alerts SET responded_at=?, response=? WHERE alert_id=?",
        (now_kst().isoformat(), response, alert_id))

    consequences: list[dict] = []

    if response == "CONFIRMED_MINE":
        # 억제 규칙 등록 (가맹점×금액대 90일 강등)
        if tx and tx["merchant_id"] is not None:
            db.add_suppression(tx["user_id"], tx["merchant_id"],
                               amount_band(tx["amount"]),
                               now_kst() + timedelta(days=SUPPRESS_DAYS))
        if tx:
            db.save_label(tx["tx_id"], tx["user_id"], is_anomaly=False,
                          source="USER", confidence=0.95)
        consequences.append({"action": "SUPPRESSION_ADDED", "days": SUPPRESS_DAYS})
        consequences.append({"action": "LABEL_RECORDED", "is_anomaly": False})

    elif response in ("NOT_MINE", "REPORT_LOST"):
        if tx:
            db.save_label(tx["tx_id"], tx["user_id"], is_anomaly=True,
                          source="USER", confidence=1.0)
            state = card_state.freeze(db, tx["card_id"], "USER",
                                      alert["assessment_id"], reason=response)
            consequences.append({"action": "CARD_FREEZE", "status": state})
        consequences.append({"action": "LABEL_RECORDED", "is_anomaly": True})
        consequences.append({"action": "MONITORING_ENHANCED", "duration_hours": 72})
        if response == "REPORT_LOST" and tx:
            card_state.report_lost(db, tx["card_id"], "USER")
            consequences.append({"action": "REPORTED"})

    db.audit("USER", f"ALERT_RESPONSE_{response}", "alert", str(alert_id),
             {"consequences": consequences})
    db.commit()
    return {"ok": True, "response": response, "consequences": consequences}
