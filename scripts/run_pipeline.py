#!/usr/bin/env python3
"""엔드투엔드 파이프라인 (PRD 15장 M1~M4 통합 실행).

생성 → 수집/정규화/DQ → 특수가맹점 분류 → routine 표시 → 프로파일 배치 →
백테스트(walk-forward) → 드리프트 → HTML 대시보드.

    python scripts/run_pipeline.py --users 200 --months 12
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spendguard.config import RULE_VERSION
from spendguard.db import DB
from spendguard.ingestion.pipeline import ingest
from spendguard.ops.backtest import run_backtest
from spendguard.ops.dashboard import render_dashboard
from spendguard.ops.drift import feature_drift
from spendguard.detection.ml import L3_FEATURES
from spendguard.profiling.builder import build_all_profiles
from spendguard.profiling.special_merchants import (classify_all_merchants,
                                                    mark_routine_transactions)
from simulator.generator import generate


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--db", default=str(ROOT / "data" / "spendguard.db"))
    ap.add_argument("--test-days", type=int, default=60)
    ap.add_argument("--no-ml", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "data" / "report.html"))
    args = ap.parse_args()

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    if Path(args.db).exists():
        Path(args.db).unlink()
    db = DB(args.db)

    # ── M1: 생성 + 수집 ────────────────────────────────────────────────
    t0 = time.time()
    _log(f"합성 데이터 생성: users={args.users}, months={args.months}")
    users, gen_txs, travel_windows = generate(args.users, args.months, seed=args.seed)
    for u in users:
        db.upsert_user(u["user_id"], u["sensitivity"])
        db.upsert_card(u["card_id"], u["user_id"], card_ref=f"tok_{u['user_id']}")
    for uid, ts, te in travel_windows:
        db.add_travel_mode(uid, ts, te)
    db.commit()
    _log(f"거래 {len(gen_txs):,}건 수집 중...")

    ingested = dropped = dup = 0
    for g in gen_txs:
        res = ingest(db, g.tx, learn_only=True)
        if not res.ok:
            dropped += 1
            continue
        if res.duplicate or res.cancelled:
            dup += 1
            continue
        ingested += 1
        if g.label is not None and res.tx_id is not None:
            db.save_injected_label(res.tx_id, g.label, g.scenario or "", g.event_id)
    db.commit()
    dq_rate = dropped / max(len(gen_txs), 1)
    _log(f"수집 완료: 적재 {ingested:,} / 중복·취소 {dup} / DLQ {dropped} "
         f"(DQ 위반율 {dq_rate*100:.3f}%)")

    # ── M1: 특수 가맹점 분류 + routine 표시 ────────────────────────────
    vc = classify_all_merchants(db)
    _log(f"가맹점 분산등급: {vc}")
    routine = mark_routine_transactions(db)
    _log(f"routine 거래 표시: {routine:,}건")

    # ── M2: 프로파일 배치 (전체 윈도우 기준, 참고용) ───────────────────
    from spendguard.db import now_kst
    from datetime import datetime, timezone, timedelta
    end = db.conn.execute("SELECT MAX(approved_at) b FROM transactions").fetchone()["b"]
    end_dt = datetime.fromisoformat(end)
    profiles = build_all_profiles(db, end_dt)
    cold = sum(1 for p in profiles if p.is_cold_start)
    _log(f"프로파일 생성: {len(profiles)}명 (콜드스타트 {cold}명)")

    # ── M3~M4: 백테스트 ────────────────────────────────────────────────
    _log(f"백테스트 실행 (test_days={args.test_days}, ML={'OFF' if args.no_ml else 'ON'})...")
    summary = run_backtest(db, test_days=args.test_days, use_ml=not args.no_ml)

    # ── 드리프트: 이력 vs 테스트 구간 피처 분포 ────────────────────────
    summary["drift"] = _drift(db, end_dt, args.test_days)

    _print_summary(summary)

    # ── 대시보드 ───────────────────────────────────────────────────────
    render_dashboard(summary, args.out, meta={"rule_version": RULE_VERSION})
    _log(f"대시보드 저장: {args.out}")
    _log(f"총 소요 {time.time()-t0:.1f}s")
    db.close()


def _drift(db, end_dt, test_days):
    from datetime import timedelta
    from spendguard.ops.backtest import _build_profiles, _row_to_tx
    from spendguard.features.transforms import build_features
    from spendguard.features.velocity import VelocityTracker
    split = end_dt - timedelta(days=test_days)
    profs, store = _build_profiles(db, split)

    def feats_for(where, params):
        rows = db.conn.execute(
            f"SELECT * FROM transactions WHERE {where} AND tx_type='APPROVAL' "
            "ORDER BY approved_at LIMIT 1500", params).fetchall()
        velo = VelocityTracker()
        out = []
        for r in rows:
            tx = _row_to_tx(r)
            p = profs.get(tx.user_id)
            if not p:
                continue
            out.append(build_features(tx, p, velo.query(tx.user_id,
                       VelocityTracker.ts_of(tx.approved_at)), db, store))
        return out
    exp = feats_for("approved_at < ?", (split.isoformat(),))
    act = feats_for("approved_at >= ?", (split.isoformat(),))
    watch = ["amt_over_p95", "log_robust_z", "merchant_novelty", "hour_rarity",
             "sum_24h_over_daily_avg", "mtd_over_monthly_ewma"]
    return feature_drift(exp, act, watch)


def _print_summary(s):
    m = s["metrics"]
    print("\n" + "=" * 60)
    print("  백테스트 결과 (합성 데이터 기준)")
    print("=" * 60)
    print(f"  거래 {m['n']:,}건 / 이상 {m['positives']}건  ML={'ON' if s['ml_enabled'] else 'OFF'}")
    print(f"  PR-AUC              : {m['pr_auc']:.3f}")
    print(f"  Event Recall (R3+)  : {m['event_recall_R3+']*100:.1f}%  (사건 단위, 카드정지 기준)")
    print(f"  Recall @ FPR 0.5%   : {m['recall_at_fpr_0.5%']*100:.1f}%  (건 단위, 목표 ≥85%)")
    print(f"  Precision @ top0.5% : {m['precision_at_top_0.5%']*100:.1f}%  (목표 ≥60%)")
    print(f"  Lift @ 1%           : {m['lift_at_1%']:.1f}×  (목표 ≥20×)")
    print("\n  등급 분포:")
    for g, i in s["grade_distribution"].items():
        print(f"    {g}: {i['count']:>6,}  ({i['pct']:.2f}%)")
    print("\n  시나리오 (SC / 다수결 / 기대 / 부합):")
    for sc, i in s["scenarios"].items():
        flag = "OK " if i["match_rate"] >= 0.5 else "?? "
        print(f"    {flag}{sc}: {i['modal_grade']:>2}  기대={'/'.join(i['expected'])}  "
              f"부합 {i['match_rate']*100:.0f}%  (n={i['n']})")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
