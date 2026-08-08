"""백테스트 하네스 — walk-forward (PRD 13.3).

원칙: '그 시점에 알 수 있었던 정보만' 사용. 프로파일은 window_end ≤ 판정일로만
구성하고, 피처의 모든 DB 조회는 approved_at 미만으로 제한된다(누수 방지 5.1).
프로파일은 주 단위로 재구성해 walk-forward를 근사한다.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from ..db import DB
from ..detection.engine import assess
from ..detection.ml import L3Model, L3_FEATURES, train_stage1
from ..features.transforms import build_features
from ..features.velocity import VelocityTracker
from ..models import Transaction
from ..profiling.builder import build_profile
from ..profiling.peer_groups import PeerStore
from .metrics import evaluate


def _row_to_tx(row) -> Transaction:
    tx = Transaction(
        user_id=row["user_id"], card_id=row["card_id"],
        approval_no=row["approval_no"], amount=row["amount"],
        approved_at=datetime.fromisoformat(row["approved_at"]),
        merchant_name="", mcc=row["mcc"] or "", channel=row["channel"],
        tx_type=row["tx_type"], installment_months=row["installment_months"],
        merchant_region=row["merchant_region"])
    tx.tx_id = row["tx_id"]
    tx.merchant_id = row["merchant_id"]
    tx.category = row["category"]
    tx.routine_flag = bool(row["routine_flag"])
    return tx


def _date_bounds(db: DB) -> tuple[datetime, datetime]:
    row = db.conn.execute(
        "SELECT MIN(approved_at) a, MAX(approved_at) b FROM transactions").fetchone()
    return datetime.fromisoformat(row["a"]), datetime.fromisoformat(row["b"])


def _build_profiles(db: DB, window_end: datetime) -> tuple[dict, PeerStore]:
    profiles = {}
    for uid in db.user_ids():
        profiles[uid] = build_profile(db, uid, window_end)
    store = PeerStore.build(list(profiles.values()))
    for p in profiles.values():
        if not p.peer_id:
            from ..profiling.peer_groups import peer_id_from_profile
            p.peer_id = peer_id_from_profile(p.stats, p.behavior)
    return profiles, store


def _train_l3(db: DB, profiles: dict, store: PeerStore, cutoff: datetime,
              sample: int = 4000) -> L3Model | None:
    """이력 구간 정상 거래 피처로 IsolationForest 학습 (비지도)."""
    rows = db.conn.execute(
        "SELECT * FROM transactions WHERE approved_at < ? AND tx_type='APPROVAL' "
        "AND is_excluded=0 ORDER BY approved_at DESC LIMIT ?",
        (cutoff.isoformat(), sample)).fetchall()
    velo = VelocityTracker()
    feature_rows = []
    for row in reversed(rows):
        tx = _row_to_tx(row)
        p = profiles.get(tx.user_id)
        if not p:
            continue
        vf = velo.query(tx.user_id, VelocityTracker.ts_of(tx.approved_at))
        feats = build_features(tx, p, vf, db, store)
        velo.record(tx.user_id, VelocityTracker.ts_of(tx.approved_at),
                    tx.amount, tx.merchant_id)
        feature_rows.append(feats)
    if len(feature_rows) < 100:
        return None
    try:
        return train_stage1(feature_rows)
    except Exception:
        return None


def run_backtest(db: DB, test_days: int = 60, cadence_days: int = 7,
                 use_ml: bool = True, persist: bool = False) -> dict:
    """테스트 구간 거래를 시간순 판정하고 성능을 집계한다."""
    _, end = _date_bounds(db)
    test_start = end - timedelta(days=test_days)

    profiles, store = _build_profiles(db, test_start)
    l3 = _train_l3(db, profiles, store, test_start) if use_ml else None

    test_rows = db.conn.execute(
        "SELECT * FROM transactions WHERE approved_at >= ? AND tx_type='APPROVAL' "
        "ORDER BY approved_at", (test_start.isoformat(),)).fetchall()

    velo = VelocityTracker()
    next_rebuild = test_start + timedelta(days=cadence_days)
    results = []
    for row in test_rows:
        tx = _row_to_tx(row)
        if tx.approved_at >= next_rebuild:
            profiles, store = _build_profiles(db, tx.approved_at)
            next_rebuild = tx.approved_at + timedelta(days=cadence_days)
        p = profiles.get(tx.user_id)
        if not p:
            continue
        a = assess(db, tx, p, velo, store, l3, persist=persist)
        truth = db.injected_label(tx.tx_id)
        sc = db.conn.execute(
            "SELECT scenario, event_id FROM injected_labels WHERE tx_id=?",
            (tx.tx_id,)).fetchone()
        results.append({
            "tx_id": tx.tx_id, "score": a.score, "grade": a.grade,
            "l1": a.l1_score, "l2": a.l2_score, "l3": a.l3_score,
            "l3_status": a.l3_status,
            "truth": bool(truth) if truth is not None else False,
            "scenario": sc["scenario"] if sc else None,
            "event_id": sc["event_id"] if sc else None,
        })

    return _summarize(results, l3 is not None)


def _summarize(results: list[dict], ml_on: bool) -> dict:
    from .scenario_report import scenario_summary, grade_distribution

    y = [1 if r["truth"] else 0 for r in results]
    scores = [r["score"] for r in results]
    metrics = evaluate(y, scores)
    metrics["event_recall_R3+"] = _event_recall(results)
    return {
        "ml_enabled": ml_on,
        "metrics": metrics,
        "grade_distribution": grade_distribution(results),
        "scenarios": scenario_summary(results),
        "results": results,
    }


def _event_recall(results: list[dict], catch_grades=("R3", "R4")) -> float:
    """이상 '사건'을 카드 정지 이전에 잡았는가 — 사건 내 1건이라도 R3+이면 탐지."""
    events: dict[int, bool] = {}
    for r in results:
        if r["truth"] and r["event_id"] is not None:
            caught = r["grade"] in catch_grades
            events[r["event_id"]] = events.get(r["event_id"], False) or caught
    if not events:
        return float("nan")
    return sum(events.values()) / len(events)
