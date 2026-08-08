"""피처 계산 — 단일 구현 원칙 (PRD 5.1, 5.3).

배치(백테스트)와 실시간(판정)이 반드시 '이 함수 하나'를 호출한다.
누수 방지(5.1): 모든 참조는 현재 거래를 제외한 과거 정보(profile, velocity,
db before=approved_at)에서만 온다. 현재 거래 자신을 통계에 넣지 않는다.
"""
from __future__ import annotations

import math
from datetime import datetime

from ..db import DB
from ..models import Profile, Transaction
from ..profiling.peer_groups import PeerStore, get_reference_stats

SQRT2 = math.sqrt(2.0)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / SQRT2))


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


def build_features(tx: Transaction, profile: Profile, velo: dict, db: DB,
                   peer_store: PeerStore | None = None) -> dict:
    s = profile.stats
    amount = float(tx.amount)
    log_amt = math.log(amount)

    # ── A. 금액 상대성 ──────────────────────────────────────────────────
    median = max(s.get("median", 1) or 1, 1)
    p95 = max(s.get("p95", 1) or 1, 1)
    p99 = max(s.get("p99", 1) or 1, 1)
    amt_over_median = _safe_div(amount, median, 1.0)
    amt_over_p95 = _safe_div(amount, p95, 1.0)
    amt_over_p99 = _safe_div(amount, p99, 1.0)

    scale = max(1.4826 * s.get("mad_log", 0.9), 0.35)
    log_robust_z = _safe_div(log_amt - math.log(median), scale, 0.0)

    log_mean = s.get("log_mean", log_amt)
    log_std = max(s.get("log_std", 1.0), 0.2)
    amt_percentile = _normal_cdf((log_amt - log_mean) / log_std)

    # 카테고리 맥락 (폴백 체인)
    ref, ref_source = get_reference_stats(profile, tx.category, peer_store)
    cat_p95 = max(ref.get("p95", p95) or p95, 1)
    cat_max = max(ref.get("max", p95) or p95, 1)
    cat_amt_over_p95 = _safe_div(amount, cat_p95, 1.0)
    cat_amt_over_max = _safe_div(amount, cat_max, 1.0)

    # 특수 가맹점 조건부 z
    merchant_amt_z = 0.0
    mstats = profile.behavior.get("merchant_stats", {})
    if tx.merchant_id is not None and str(tx.merchant_id) in mstats:
        ms = mstats[str(tx.merchant_id)]
        msd = max(ms.get("log_std", 0.5), 0.2)
        merchant_amt_z = _safe_div(log_amt - ms.get("log_mean", log_amt), msd, 0.0)

    # ── B. 가맹점 신규성 ────────────────────────────────────────────────
    mtc = db.merchant_tx_count(tx.user_id, tx.merchant_id, tx.approved_at) \
        if tx.merchant_id is not None else 0
    merchant_novelty = 1.0 if mtc == 0 else (0.0 if mtc >= 10 else 0.5)
    days_since_first_seen = _days_since_first(db, tx)
    cat = profile.cat(tx.category)
    cat_n = cat.get("n", 0) if cat else 0
    cat_novelty = 1.0 if cat_n == 0 else (0.5 if cat_n < 5 else max(0.0, 1 - cat_n / 20))

    # ── C. 시간 이례도 ──────────────────────────────────────────────────
    hh = profile.time_stats.get("hour_hist", [1 / 24] * 24)
    dh = profile.time_stats.get("dow_hist", [1 / 7] * 7)
    hmax = max(hh) or 1e-9
    hour_rarity = 1.0 - _safe_div(hh[tx.hour], hmax, 0.0)
    dmax = max(dh) or 1e-9
    dow_rarity = 1.0 - _safe_div(dh[tx.dow], dmax, 0.0)
    is_night = 1.0 if 0 <= tx.hour < 6 else 0.0
    secs_since_prev = _secs_since_prev(db, tx)

    # ── D. 벨로시티 ─────────────────────────────────────────────────────
    daily_avg = max(s.get("monthly_sum_ewma", 800_000) / 30, 10_000)
    sum_24h_over_daily_avg = _safe_div(velo["tx_sum_24h"] + amount, daily_avg, 0.0)

    # ── E. 채널·맥락 ────────────────────────────────────────────────────
    is_online = 1.0 if tx.channel == "ONLINE" else 0.0
    is_overseas = 1.0 if tx.channel == "OVERSEAS" else 0.0
    online_rate_deviation = is_online - profile.behavior.get("online_ratio", 0.4)
    region_novelty = _region_novelty(db, tx)
    is_installment = 1.0 if tx.installment_months > 0 else 0.0

    # ── F. 누적 소비 추세 ───────────────────────────────────────────────
    mtd = _month_to_date_sum(db, tx)
    monthly_ewma = max(s.get("monthly_sum_ewma", 800_000), 1)
    mtd_over_monthly_ewma = _safe_div(mtd + amount, monthly_ewma, 0.0)
    days_into_month_ratio = tx.approved_at.day / 30.0
    budget_burn_deviation = _safe_div(mtd_over_monthly_ewma,
                                      max(days_into_month_ratio, 0.05), 0.0) - 1.0

    return {
        "amount": amount,
        "amt_over_median": amt_over_median, "amt_over_p95": amt_over_p95,
        "amt_over_p99": amt_over_p99, "log_robust_z": log_robust_z,
        "amt_percentile": amt_percentile,
        "cat_amt_over_p95": cat_amt_over_p95, "cat_amt_over_max": cat_amt_over_max,
        "merchant_amt_z": merchant_amt_z,
        "merchant_novelty": merchant_novelty, "merchant_tx_count": mtc,
        "days_since_first_seen": days_since_first_seen, "cat_novelty": cat_novelty,
        "hour_rarity": hour_rarity, "is_night": is_night, "dow_rarity": dow_rarity,
        "secs_since_prev_tx": secs_since_prev,
        "tx_count_1h": velo["tx_count_1h"], "tx_sum_1h": velo["tx_sum_1h"],
        "tx_count_24h": velo["tx_count_24h"], "tx_sum_24h": velo["tx_sum_24h"],
        "sum_24h_over_daily_avg": sum_24h_over_daily_avg,
        "distinct_merchants_1h": velo["distinct_merchants_1h"],
        "is_online": is_online, "online_rate_deviation": online_rate_deviation,
        "is_overseas": is_overseas, "region_novelty": region_novelty,
        "is_installment": is_installment, "routine_flag": 1.0 if tx.routine_flag else 0.0,
        "mtd_over_monthly_ewma": mtd_over_monthly_ewma,
        "days_into_month_ratio": days_into_month_ratio,
        "budget_burn_deviation": budget_burn_deviation,
        "_reference_source": ref_source,
    }


def _days_since_first(db: DB, tx: Transaction) -> float:
    if tx.merchant_id is None:
        return 0.0
    row = db.conn.execute(
        "SELECT MIN(approved_at) f FROM transactions WHERE user_id=? AND merchant_id=? "
        "AND approved_at < ?",
        (tx.user_id, tx.merchant_id, tx.approved_at.isoformat())).fetchone()
    if not row or row["f"] is None:
        return 0.0
    return max(0.0, (tx.approved_at - datetime.fromisoformat(row["f"])).total_seconds() / 86400)


def _secs_since_prev(db: DB, tx: Transaction) -> float:
    row = db.last_tx_before(tx.user_id, tx.approved_at)
    if not row:
        return 1e9
    return (tx.approved_at - datetime.fromisoformat(row["approved_at"])).total_seconds()


def _region_novelty(db: DB, tx: Transaction) -> float:
    if not tx.merchant_region:
        return 0.0
    regions = db.distinct_regions(tx.user_id, tx.approved_at)
    return 0.0 if tx.merchant_region in regions else 1.0


def _month_to_date_sum(db: DB, tx: Transaction) -> float:
    month_start = tx.approved_at.replace(day=1, hour=0, minute=0, second=0,
                                         microsecond=0)
    row = db.conn.execute(
        "SELECT COALESCE(SUM(amount),0) s FROM transactions WHERE user_id=? "
        "AND approved_at >= ? AND approved_at < ? AND tx_type='APPROVAL' "
        "AND is_excluded=0",
        (tx.user_id, month_start.isoformat(), tx.approved_at.isoformat())).fetchone()
    return float(row["s"])
