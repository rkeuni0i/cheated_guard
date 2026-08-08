"""특수(고분산) 가맹점 분류 + routine 거래 판정 (PRD 4.5).

변동계수(CV)와 사분위범위비(QCD)를 함께 봐서 극단값 한 건에 흔들리지 않게 한다.
주 1회 배치로 merchants.variance_class / is_special을 갱신한다.
"""
from __future__ import annotations

import numpy as np

from ..db import DB


def classify_variance(amounts: list[int]) -> str:
    """가맹점 금액 분포 → LOW|MID|HIGH (PRD 4.5 의사코드)."""
    if len(amounts) < 30:
        return "MID"                       # 표본 부족 → 중립
    a = np.asarray(amounts, dtype=float)
    mean = a.mean()
    cv = a.std() / mean if mean else 0.0
    q1, q3 = np.percentile(a, [25, 75])
    qcd = (q3 - q1) / (q3 + q1) if (q3 + q1) else 0.0
    if cv >= 1.0 or qcd >= 0.45:
        return "HIGH"                       # 올리브영·다이소형
    if cv <= 0.35 and qcd <= 0.20:
        return "LOW"                        # 지하철·구독형
    return "MID"


def classify_all_merchants(db: DB) -> dict[str, int]:
    """전체 가맹점 분산등급 재분류 배치. HIGH는 is_special=True로 표시."""
    counts = {"LOW": 0, "MID": 0, "HIGH": 0}
    rows = db.conn.execute("SELECT merchant_id FROM merchants").fetchall()
    for r in rows:
        mid = r["merchant_id"]
        vc = classify_variance(db.all_merchant_amounts(mid))
        db.set_merchant_special(mid, is_special=(vc == "HIGH"), variance_class=vc)
        counts[vc] += 1
    db.commit()
    return counts


def is_routine(days_between: list[float], amounts: list[int]) -> bool:
    """주기성(월 1회±) + 금액 안정성(CV≤0.15) → routine 여부 (PRD 4.5)."""
    if len(amounts) < 3 or len(days_between) < 2:
        return False
    iv = np.asarray(days_between, dtype=float)
    period_stable = iv.std() <= 5 and 25 <= np.median(iv) <= 35
    a = np.asarray(amounts, dtype=float)
    amount_stable = (a.std() / a.mean() if a.mean() else 1.0) <= 0.15
    return bool(period_stable and amount_stable)


def mark_routine_transactions(db: DB) -> int:
    """사용자×가맹점 단위로 routine 패턴을 찾아 routine_flag를 세운다.

    구독·통신·교통처럼 주기·금액이 안정적인 조합만 표시. 갱신 건수 반환.
    """
    marked = 0
    rows = db.conn.execute(
        "SELECT user_id, merchant_id, COUNT(*) c FROM transactions "
        "WHERE tx_type='APPROVAL' AND merchant_id IS NOT NULL "
        "GROUP BY user_id, merchant_id HAVING c >= 3").fetchall()
    for r in rows:
        txs = db.conn.execute(
            "SELECT tx_id, amount, approved_at FROM transactions "
            "WHERE user_id=? AND merchant_id=? AND tx_type='APPROVAL' "
            "ORDER BY approved_at", (r["user_id"], r["merchant_id"])).fetchall()
        from datetime import datetime
        times = [datetime.fromisoformat(t["approved_at"]) for t in txs]
        days = [(times[i] - times[i - 1]).total_seconds() / 86400
                for i in range(1, len(times))]
        amounts = [t["amount"] for t in txs]
        if is_routine(days, amounts):
            ids = [t["tx_id"] for t in txs]
            db.conn.executemany(
                "UPDATE transactions SET routine_flag=1 WHERE tx_id=?",
                [(i,) for i in ids])
            marked += len(ids)
    db.commit()
    return marked
