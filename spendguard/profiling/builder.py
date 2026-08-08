"""프로파일 배치 빌더 (PRD 3.3.A, 4.3 JSONB, 5.2).

SQLite에는 percentile_cont가 없으므로 numpy로 로버스트 통계를 산출한다.
핵심 판단(PRD 6.3): 평균·표준편차 대신 median·MAD·분위수를 쓴다.
누수 방지(5.1): window_end 미만 거래만, 현재 거래는 자연히 제외된다.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

from ..config import (CAT_FALLBACK_MIN_N, COLD_START_MIN_DAYS,
                      COLD_START_MIN_TX, PROFILE_WINDOW_DAYS)
from ..db import DB
from ..models import Profile
from ..taxonomy import category_variance
from .peer_groups import peer_id_from_profile


def _percentiles(a: np.ndarray) -> dict:
    return {
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def build_profile(db: DB, user_id: str, window_end: datetime,
                  window_days: int = PROFILE_WINDOW_DAYS) -> Profile:
    """window_end 시점 기준 사용자 프로파일을 산출한다 (저장은 별도)."""
    rows = db.window_transactions(user_id, window_end, window_days)
    n = len(rows)

    span_days = window_days
    if rows:
        first = datetime.fromisoformat(rows[0]["approved_at"])
        span_days = max(1, (window_end - first).days)

    is_cold = n < COLD_START_MIN_TX or span_days < COLD_START_MIN_DAYS
    version = db.next_profile_version(user_id)
    ws = (window_end - timedelta(days=window_days)).date().isoformat()
    we = window_end.date().isoformat()

    if n == 0:
        return Profile(user_id, version, ws, we, 0, True,
                       stats=_empty_stats(), cat_stats={}, time_stats=_empty_time(),
                       behavior=_empty_behavior())

    amounts = np.array([r["amount"] for r in rows], dtype=float)
    log_amt = np.log(amounts)

    # ── 전체 통계 (robust) ──────────────────────────────────────────────
    pcts = _percentiles(amounts)
    median = pcts["median"]
    mad = float(np.median(np.abs(amounts - median)))
    mad_log = float(np.median(np.abs(log_amt - np.log(median))))
    # 월별 합계 → EWMA/표준편차
    monthly = _monthly_sums(rows)
    stats = {
        "n": n, "sum": float(amounts.sum()), "mad": mad, "mad_log": mad_log,
        "log_mean": float(log_amt.mean()), "log_std": float(log_amt.std(ddof=1) if n > 1 else 0.5),
        "monthly_sum_ewma": float(np.mean(monthly)) if monthly else float(amounts.sum()),
        "monthly_sum_std": float(np.std(monthly)) if len(monthly) > 1 else 0.0,
        **pcts,
    }

    # ── 카테고리별 통계 ─────────────────────────────────────────────────
    cat_amounts: dict[str, list] = defaultdict(list)
    for r in rows:
        cat_amounts[r["category"]].append(r["amount"])
    total = amounts.sum()
    cat_stats = {}
    for cat, vals in cat_amounts.items():
        v = np.array(vals, dtype=float)
        cat_stats[cat] = {
            "n": len(vals), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "max": float(v.max()),
            "share": round(float(v.sum() / total), 4),
            "fallback": len(vals) < CAT_FALLBACK_MIN_N,
            "special": category_variance(cat) == "HIGH",
        }

    # ── 특수 가맹점 조건부 분포 (PRD 4.5, axis_context용) ────────────────
    merch_amounts: dict[int, list] = defaultdict(list)
    for r in rows:
        if r["merchant_id"] is not None:
            m = db.get_merchant(r["merchant_id"])
            if m and m["is_special"]:
                merch_amounts[r["merchant_id"]].append(r["amount"])
    merchant_stats = {}
    for mid, vals in merch_amounts.items():
        if len(vals) >= 3:
            v = np.array(vals, dtype=float)
            merchant_stats[str(mid)] = {
                "n": len(vals), "median": float(np.median(v)),
                "p95": float(np.percentile(v, 95)), "max": float(v.max()),
                "log_mean": float(np.log(v).mean()),
                "log_std": float(np.log(v).std(ddof=1) if len(v) > 1 else 0.5),
            }

    # ── 시간 통계 ──────────────────────────────────────────────────────
    hours = np.array([datetime.fromisoformat(r["approved_at"]).hour for r in rows])
    dows = np.array([datetime.fromisoformat(r["approved_at"]).weekday() for r in rows])
    hour_hist = (np.bincount(hours, minlength=24) / n).tolist()
    dow_hist = (np.bincount(dows, minlength=7) / n).tolist()
    time_stats = {
        "hour_hist": hour_hist, "dow_hist": dow_hist,
        "night_ratio": float(np.mean((hours >= 0) & (hours < 6))),
        "weekend_ratio": float(np.mean(dows >= 5)),
    }

    # ── 행동 프로파일 ──────────────────────────────────────────────────
    merchants = [r["merchant_id"] for r in rows if r["merchant_id"] is not None]
    seen: set = set()
    new_count = 0
    for mid in merchants:
        if mid not in seen:
            new_count += 1
            seen.add(mid)
    top = _top_merchants(db, rows, k=3)
    behavior = {
        "new_merchant_ratio": round(new_count / n, 4),
        "online_ratio": round(float(np.mean([r["channel"] == "ONLINE" for r in rows])), 4),
        "distinct_merchants": len(seen),
        "top_merchants": top,
        "weekly_tx_count_ewma": round(n * 7 / span_days, 2),
        "installment_ratio": round(float(np.mean([r["installment_months"] > 0 for r in rows])), 4),
        "merchant_stats": merchant_stats,
    }

    p = Profile(user_id, version, ws, we, n, is_cold, stats, cat_stats,
                time_stats, behavior)
    p.peer_id = peer_id_from_profile(stats, behavior)
    u = db.get_user(user_id)
    if u:
        p.sensitivity = u["sensitivity"]
    return p


def _monthly_sums(rows) -> list[float]:
    buckets: dict[str, float] = defaultdict(float)
    for r in rows:
        key = r["approved_at"][:7]      # YYYY-MM
        buckets[key] += r["amount"]
    return list(buckets.values())


def _top_merchants(db: DB, rows, k=3) -> list[str]:
    counts: dict[int, int] = defaultdict(int)
    for r in rows:
        if r["merchant_id"] is not None:
            counts[r["merchant_id"]] += 1
    top_ids = sorted(counts, key=counts.get, reverse=True)[:k]
    names = []
    for mid in top_ids:
        m = db.get_merchant(mid)
        if m:
            names.append(m["name_normalized"])
    return names


def save_and_return(db: DB, profile: Profile) -> Profile:
    profile.profile_id = db.save_profile(profile)
    db.commit()
    return profile


def build_all_profiles(db: DB, window_end: datetime) -> list[Profile]:
    """전체 사용자 프로파일 배치 생성 + 저장."""
    profiles = []
    for uid in db.user_ids():
        p = build_profile(db, uid, window_end)
        save_and_return(db, p)
        profiles.append(p)
    return profiles


def _empty_stats() -> dict:
    return {"n": 0, "sum": 0, "median": 30_000, "mad": 15_000, "mad_log": 0.9,
            "p75": 40_000, "p90": 60_000, "p95": 100_000, "p99": 200_000,
            "max": 300_000, "log_mean": 10.0, "log_std": 1.0,
            "monthly_sum_ewma": 800_000, "monthly_sum_std": 0.0}


def _empty_time() -> dict:
    return {"hour_hist": [1 / 24] * 24, "dow_hist": [1 / 7] * 7,
            "night_ratio": 0.02, "weekend_ratio": 0.28}


def _empty_behavior() -> dict:
    return {"new_merchant_ratio": 0.5, "online_ratio": 0.4, "distinct_merchants": 0,
            "top_merchants": [], "weekly_tx_count_ewma": 0.0,
            "installment_ratio": 0.0, "merchant_stats": {}}
