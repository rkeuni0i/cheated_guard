"""L2 통계 스코어러 — 4개 축 (PRD 6.3).

핵심 판단: 평균±3σ 대신 로그변환 + MAD + 분위수(로버스트 통계)를 쓴다.
개인 결제 분포는 롱테일이라 평균 기반 임계값은 스스로 무력화된다(PRD 6.3 박스).
"""
from __future__ import annotations

import math

from ..config import (L2_SOFTMAX_MIX, L2_WEIGHTS, MAD_LOG_FLOOR,
                      ROUTINE_DAMP, ROUTINE_DAMP_RELEASE)
from ..models import Profile, Transaction
from ..profiling.peer_groups import PeerStore, get_reference_stats

SQRT2 = math.sqrt(2.0)


def _ncdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / SQRT2))


def axis_amount(amount: float, profile: Profile) -> float:
    """축1: 로그-로버스트 Z → 0~100 (전체 분포 이탈도)."""
    s = profile.stats
    median = max(s.get("median", 1) or 1, 1)
    scale = max(1.4826 * s.get("mad_log", 0.9), MAD_LOG_FLOOR)
    z = (math.log(amount) - math.log(median)) / scale
    return 100 * (1 - math.exp(-max(z - 1.0, 0) / 2.2))


def _empirical_percentile(amount: float, ref: dict) -> float:
    """참조 통계(median, p95)에서 로그정규 근사 분위수 (0~1)."""
    median = max(ref.get("median", 1) or 1, 1)
    p95 = max(ref.get("p95", median * 3) or median * 3, median + 1)
    sigma = max(math.log(p95 / median) / 1.645, 0.2)
    return _ncdf((math.log(amount) - math.log(median)) / sigma)


def axis_context(tx: Transaction, profile: Profile,
                 peer_store: PeerStore | None) -> tuple[float, str]:
    """축2: 카테고리/가맹점 조건부 분위수 → 상위 꼬리만 점수화."""
    ref, source = get_reference_stats(profile, tx.category, peer_store)

    # 특수 가맹점: 이 가맹점에서의 내 분포 우선
    mstats = profile.behavior.get("merchant_stats", {})
    if tx.merchant_id is not None and str(tx.merchant_id) in mstats:
        ms = mstats[str(tx.merchant_id)]
        ref = {"median": ms["median"], "p95": ms["p95"], "max": ms["max"]}
        source = "MERCHANT"

    q = _empirical_percentile(float(tx.amount), ref)
    if q < 0.95:
        return 0.0, source
    score = (q - 0.95) / 0.05 * 100
    if source not in ("PERSONAL", "MERCHANT"):
        score *= 0.7                     # peer/blended 기반이면 신뢰도 감쇠
    return min(100.0, score), source


def axis_time(tx: Transaction, profile: Profile) -> float:
    """축3: 시간대 이례도 (라플라스 스무딩 기반 정보량)."""
    hh = profile.time_stats.get("hour_hist", [1 / 24] * 24)
    n = max(profile.stats.get("n", 1), 1)
    p_hour = hh[tx.hour]
    p_smooth = (p_hour * n + 1) / (n + 24)
    surprise = -math.log(p_smooth)
    baseline = -math.log(1 / 24)
    return min(100.0, max(0.0, (surprise - baseline) / baseline * 100))


def axis_velocity(feats: dict, profile: Profile) -> float:
    """축4: 벨로시티 이탈 (24시간 합계/건수의 평소 대비 배율)."""
    s = profile.stats
    daily_avg = max(s.get("monthly_sum_ewma", 800_000) / 30, 10_000)
    ratio_sum = (feats["tx_sum_24h"] + feats["amount"]) / daily_avg
    weekly_cnt = max(profile.behavior.get("weekly_tx_count_ewma", 7) / 7, 1)
    ratio_cnt = (feats["tx_count_24h"] + 1) / weekly_cnt
    r = max(ratio_sum, ratio_cnt)
    if r <= 2.0:
        return 0.0
    return min(100.0, (r - 2.0) / 6.0 * 100)


def score_l2(tx: Transaction, profile: Profile, feats: dict,
             peer_store: PeerStore | None) -> tuple[int, dict, str]:
    """4개 축을 가중결합 + soft-max 보정 → 0~100."""
    ctx_score, ctx_source = axis_context(tx, profile, peer_store)
    axes = {
        "amount": axis_amount(float(tx.amount), profile),
        "context": ctx_score,
        "time": axis_time(tx, profile),
        "velocity": axis_velocity(feats, profile),
    }
    weighted = sum(L2_WEIGHTS[k] * v for k, v in axes.items())
    max_axis = max(axes.values())
    score = (1 - L2_SOFTMAX_MIX) * weighted + L2_SOFTMAX_MIX * max_axis

    # routine 감쇠 (PRD 원안의 '단순 거래 제외')
    if tx.routine_flag and feats["amt_over_median"] < ROUTINE_DAMP_RELEASE:
        score *= ROUTINE_DAMP

    return round(min(100.0, score)), axes, ctx_source
