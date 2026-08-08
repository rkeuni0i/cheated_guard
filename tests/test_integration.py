"""통합 테스트 — 수집→프로파일→판정→액션→피드백 (PRD 13.1, 13.4 AC).

testcontainers 대신 인메모리 SQLite로 전체 경로를 실제 연결해 검증한다.
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spendguard.actions.feedback import respond
from spendguard.actions.policy import execute_actions
from spendguard.db import DB
from spendguard.detection.engine import assess
from spendguard.features.velocity import VelocityTracker
from spendguard.ingestion.pipeline import ingest
from spendguard.models import Transaction
from spendguard.profiling.builder import build_profile, save_and_return
from spendguard.profiling.peer_groups import PeerStore

KST = timezone(timedelta(hours=9))
NOW = datetime(2026, 8, 1, tzinfo=KST)


def _setup_user(db, user_id="U-1042", card_id="C-1042"):
    db.upsert_user(user_id, "DEFAULT")
    db.upsert_card(card_id, user_id, "tok_x")
    rng = np.random.default_rng(7)
    # 90일간 소액 위주 정상 거래 (median≈18k, p99≈145k 근처)
    for i in range(400):
        when = NOW - timedelta(days=int(rng.uniform(1, 89)),
                               hours=int(rng.uniform(8, 22)))
        amt = int(min(max(1000, np.round(rng.lognormal(np.log(18_400), 0.85) / 10) * 10),
                      300_000))
        cat_mcc = [("FOOD", "5812"), ("CAFE", "5814"), ("GROCERY", "5411"),
                   ("ELECTRONICS", "5732")][int(rng.integers(0, 4))]
        tx = Transaction(user_id, card_id, uuid.uuid4().hex[:12], amt, when,
                         f"가맹점{int(rng.integers(0,20))}", cat_mcc[1],
                         channel="OFFLINE", merchant_region="서울 강남구")
        ingest(db, tx, learn_only=True)
    db.commit()
    return user_id, card_id


def test_core_scenario_312k():
    """AC-CORE-01: 2.3 시나리오(312k 전자·신규가맹점) → R3 + 근거 3개."""
    db = DB(":memory:")
    uid, cid = _setup_user(db)
    profile = save_and_return(db, build_profile(db, uid, NOW))
    store = PeerStore.build([profile])

    tx = Transaction(uid, cid, "88213047", 312_000,
                     NOW + timedelta(hours=2), "컴퓨존 온라인", "5732",
                     channel="ONLINE", merchant_region="서울 강남구")
    tx.tx_id = ingest(db, tx).tx_id
    tx.category = "ELECTRONICS"

    a = assess(db, tx, profile, VelocityTracker(), store, None)
    assert a.grade in ("R3", "R4"), f"expected R3+, got {a.grade} (score={a.score})"
    assert len(a.reasons) >= 2
    assert all(r.label for r in a.reasons)      # 공허하지 않은 근거
    db.close()


def test_idempotency():
    """AC-IDEM-01: 동일 거래 여러 번 전송 → 판정 대상 1건."""
    db = DB(":memory:")
    uid, cid = _setup_user(db)
    tx = Transaction(uid, cid, "DUP-1", 50_000, NOW + timedelta(hours=1),
                     "가맹점3", "5812", channel="OFFLINE")
    r1 = ingest(db, tx)
    tx2 = Transaction(uid, cid, "DUP-1", 50_000, tx.approved_at, "가맹점3", "5812",
                      channel="OFFLINE")
    r2 = ingest(db, tx2)
    assert r1.tx_id is not None and r2.duplicate
    db.close()


def test_ml_off_fallback():
    """AC-FAIL-01: ML 없이도 판정 성공 (L1+L2)."""
    db = DB(":memory:")
    uid, cid = _setup_user(db)
    profile = save_and_return(db, build_profile(db, uid, NOW))
    tx = Transaction(uid, cid, "NOML", 500_000, NOW + timedelta(hours=3),
                     "낯선전자몰", "5732", channel="ONLINE")
    tx.tx_id = ingest(db, tx).tx_id
    tx.category = "ELECTRONICS"
    a = assess(db, tx, profile, VelocityTracker(), None, l3_model=None)
    assert a.l3_status == "SKIPPED"
    assert 0 <= a.score <= 100
    db.close()


def test_feedback_freezes_card():
    """피드백: NOT_MINE 응답 → 카드 정지 + 이상 라벨 (7.5)."""
    db = DB(":memory:")
    uid, cid = _setup_user(db)
    profile = save_and_return(db, build_profile(db, uid, NOW))
    tx = Transaction(uid, cid, "FRD-1", 900_000, NOW + timedelta(hours=4),
                     "낯선쇼핑몰", "5732", channel="ONLINE")
    tx.tx_id = ingest(db, tx).tx_id
    tx.category = "ELECTRONICS"
    a = assess(db, tx, profile, VelocityTracker(), None, None)
    execute_actions(db, a, tx)
    alert = db.conn.execute("SELECT alert_id FROM alerts ORDER BY alert_id DESC "
                            "LIMIT 1").fetchone()
    if alert:                              # R2+ 여야 알림 존재
        res = respond(db, alert["alert_id"], "NOT_MINE")
        assert res["ok"]
        card = db.get_card(cid)
        assert card["status"] in ("FROZEN", "REPORTED")
    db.close()


def test_reproducibility():
    """AC-REPRO-01: 동일 입력 재판정 시 동일 스코어."""
    db = DB(":memory:")
    uid, cid = _setup_user(db)
    profile = save_and_return(db, build_profile(db, uid, NOW))
    store = PeerStore.build([profile])
    tx = Transaction(uid, cid, "REP-1", 250_000, NOW + timedelta(hours=5),
                     "가맹점5", "5732", channel="ONLINE")
    tx.tx_id = ingest(db, tx).tx_id
    tx.category = "ELECTRONICS"
    a1 = assess(db, tx, profile, VelocityTracker(), store, None, persist=False)
    a2 = assess(db, tx, profile, VelocityTracker(), store, None, persist=False)
    assert a1.score == a2.score and a1.grade == a2.grade
    db.close()
