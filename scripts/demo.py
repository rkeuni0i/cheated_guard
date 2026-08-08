#!/usr/bin/env python3
"""라이브 데모 — 2.3 대표 시나리오를 처음부터 끝까지 실행 (PRD 2.3).

거래 수신 → 피처 → 3계층 스코어 → 융합/등급 → 근거 → 액션 → 사용자 응답.
발표/면접에서 '한 거래가 어떻게 처리되는가'를 보여주는 용도.

    python scripts/demo.py
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
NOW = datetime(2026, 8, 8, 19, 30, tzinfo=KST)


def _seed_history(db, uid, cid):
    """PRD 2.3 예시 프로파일 재현: median≈18k, p99≈145k, max≈289k,
    전자 카테고리 소수(cat_max≈89k), 저녁 시간대가 흔함."""
    rng = np.random.default_rng(1042)
    for _ in range(400):
        # 저녁(19시)·점심(12시) 이중 봉우리 → 금요일 저녁은 흔한 시간
        hour = float(np.clip(rng.normal(19, 2) if rng.random() < 0.6
                             else rng.normal(12.5, 1.5), 8, 23))
        when = NOW - timedelta(days=int(rng.uniform(1, 89)))
        when = when.replace(hour=int(hour), minute=int((hour % 1) * 60))
        amt = int(min(max(1000, round(rng.lognormal(np.log(18_400), 0.9) / 10) * 10), 289_000))
        cm = [("FOOD", "5812"), ("CAFE", "5814"), ("GROCERY", "5411")][
            int(rng.integers(0, 3))]
        ingest(db, Transaction(uid, cid, uuid.uuid4().hex[:12], amt, when,
               f"단골가맹점{int(rng.integers(0,15))}", cm[1], channel="OFFLINE",
               merchant_region="서울 강남구"), learn_only=True)
    # 전자 카테고리 소수 이력 (median 42k, max 89k) — 신규 아님
    for a in (35_000, 42_000, 58_000, 89_000):
        when = NOW - timedelta(days=int(rng.uniform(20, 80)))
        ingest(db, Transaction(uid, cid, uuid.uuid4().hex[:12], a,
               when.replace(hour=20), "하이마트", "5732", channel="OFFLINE",
               merchant_region="서울 강남구"), learn_only=True)
    db.commit()


def main():
    db = DB(":memory:")
    uid, cid = "U-1042", "C-1042"
    db.upsert_user(uid, "DEFAULT")
    db.upsert_card(cid, uid, "tok_9d1f4a7b2c8e")
    _seed_history(db, uid, cid)

    profile = save_and_return(db, build_profile(db, uid, NOW))
    store = PeerStore.build([profile])
    print(f"\n[프로파일] median={profile.stats['median']:,.0f}  "
          f"p95={profile.stats['p95']:,.0f}  p99={profile.stats['p99']:,.0f}  "
          f"콜드스타트={profile.is_cold_start}")

    # 대표 시나리오: 312,000원 PC부품, 신규 온라인 가맹점, 금요일 저녁
    tx = Transaction(uid, cid, "88213047", 312_000, NOW, "컴퓨존 온라인", "5732",
                     channel="ONLINE", merchant_region="서울 강남구")
    tx.tx_id = ingest(db, tx).tx_id
    tx.category = "ELECTRONICS"
    print(f"\n[거래] {tx.merchant_name} {tx.amount:,}원 ({tx.channel}, "
          f"{tx.approved_at.strftime('%m/%d %H:%M')})")

    a = assess(db, tx, profile, VelocityTracker(), store, None)
    print(f"\n[판정] score={a.score} → {a.grade}   "
          f"(L1={a.l1_score} L2={a.l2_score} L3={a.l3_score}/{a.l3_status} "
          f"override={a.override})")
    print("[근거]")
    for r in a.reasons:
        print(f"   · {r.label}  (기여 {r.contribution:.0%})")

    taken = execute_actions(db, a, tx)
    print(f"\n[액션] {[t['action'] for t in taken]}")

    alert = db.conn.execute("SELECT alert_id FROM alerts ORDER BY alert_id DESC "
                            "LIMIT 1").fetchone()
    if alert:
        print("\n[사용자 응답] '내가 안 했어요' →")
        res = respond(db, alert["alert_id"], "NOT_MINE")
        for c in res["consequences"]:
            print(f"   · {c}")
        print(f"[카드 상태] {db.get_card(cid)['status']}")
    db.close()
    print()


if __name__ == "__main__":
    main()
