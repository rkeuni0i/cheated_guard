#!/usr/bin/env python3
"""진단: 특정 시나리오 거래의 판정 상세를 출력."""
from __future__ import annotations
import sys
from datetime import datetime, timedelta
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spendguard.db import DB
from spendguard.ops.backtest import _build_profiles, _row_to_tx
from spendguard.features.transforms import build_features
from spendguard.features.velocity import VelocityTracker
from spendguard.detection.engine import assess

db = DB(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "spendguard.db"))
scenario = sys.argv[2] if len(sys.argv) > 2 else "SC-03"
end = datetime.fromisoformat(db.conn.execute("SELECT MAX(approved_at) b FROM transactions").fetchone()["b"])
split = end - timedelta(days=60)
profs, store = _build_profiles(db, split)

rows = db.conn.execute(
    "SELECT t.* FROM transactions t JOIN injected_labels il ON t.tx_id=il.tx_id "
    "WHERE il.scenario=? AND t.approved_at>=? ORDER BY t.approved_at LIMIT 12",
    (scenario, split.isoformat())).fetchall()

velo = VelocityTracker()
print(f"=== {scenario}: {len(rows)} txs ===")
for r in rows:
    tx = _row_to_tx(r)
    p = profs.get(tx.user_id)
    if not p:
        continue
    a = assess(db, tx, p, velo, store, None, persist=False)
    u = db.get_user(tx.user_id)
    codes = [rr.code for rr in a.reasons]
    print(f"  {tx.amount:>8,} {tx.channel:8} {tx.category:11} "
          f"score={a.score:3} {a.grade} L1={a.l1_score} L2={a.l2_score} "
          f"ovr={a.override} sens={u['sensitivity']} p99={p.stats.get('p99'):,.0f} "
          f"cold={p.is_cold_start} reasons={codes}")
