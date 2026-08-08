"""스코어 융합 + 등급 판정 (PRD 6.5, 7.1).

융합 순서: 가중결합 → 최대계층 가산 → 맥락 보정 → HARD OVERRIDE.
등급: 경계 매핑 후 R4 자격검사(HIGH 룰 히트 필수 + not 콜드스타트) → 피로도 →
민감도 프리셋.
"""
from __future__ import annotations

from datetime import datetime

from ..config import (BASE_WEIGHTS, DAILY_ALERT_CAP, FALLBACK_WEIGHTS,
                      GRADE_BOUNDS, STRONGEST_MIX)
from ..actions.fusion_bands import amount_band as _amount_band
from ..db import DB
from ..models import Profile, Reason, Transaction


def fuse(l1: int, l2: int, l3: int, l3_status: str, tx: Transaction,
         profile: Profile, feats: dict, l1_hits: list[Reason], db: DB) -> tuple[int, str]:
    w = BASE_WEIGHTS if l3_status == "OK" else FALLBACK_WEIGHTS
    if l3_status != "OK":
        l3 = 0

    base = w["l1"] * l1 + w["l2"] * l2 + w["l3"] * l3
    strongest = max(l1, l2, l3)
    score = (1 - STRONGEST_MIX) * base + STRONGEST_MIX * strongest

    # 맥락 보정 (각 항목 상한 ±20)
    adj = 0
    if feats["mtd_over_monthly_ewma"] < 1.0:
        adj -= 3
    if feats["is_installment"]:
        adj -= 12                       # 할부 = 계획된 소비
    if profile.is_cold_start:
        adj -= 8                        # 근거 부족 → 보수적
    if feats["merchant_tx_count"] >= 10:
        adj -= 12                       # 단골 가맹점
    if feats["is_overseas"]:
        adj += 10
    if feats["tx_count_1h"] + 1 >= 5:
        adj += 8
    adj = max(-20, min(20, adj))
    score = max(0, min(100, round(score + adj)))

    # HARD OVERRIDE
    if db.is_whitelisted(tx.user_id, tx.merchant_id):
        return 0, "WHITELIST"
    band = _amount_band(tx.amount)
    if db.is_suppressed(tx.user_id, tx.merchant_id, band, tx.approved_at):
        return min(score, 45), "SUPPRESSED"
    if feats["is_overseas"] and db.in_travel_mode(tx.user_id, tx.approved_at):
        return min(score, 45), "TRAVEL_MODE"       # R1 이하로 제한
    if any(h.code == "R-09" for h in l1_hits):
        return max(score, 92), "FORCED_CRITICAL"

    return score, "NORMAL"


def to_grade(score: int, l1_hits: list[Reason], profile: Profile, db: DB,
             day: datetime) -> str:
    base = next(g for g, lo in GRADE_BOUNDS if score >= lo)

    # R4 자격검사: HIGH 룰 히트 + not 콜드스타트 (PRD 7.1)
    if base == "R4":
        has_high = any(getattr(h, "severity", None) == "HIGH" for h in l1_hits) \
            or any(h.code in _HIGH_RULES for h in l1_hits)
        if not has_high or profile.is_cold_start:
            base = "R3"

    # 알림 피로도 상한 (R2는 억제, R3+는 무관)
    if base == "R2" and db.alerts_today(profile.user_id, day) >= DAILY_ALERT_CAP:
        base = "R1"

    # 민감도 프리셋
    if profile.sensitivity == "HIGH":
        base = _bump(base, +1, max_grade="R3")   # R4 승격 금지
    elif profile.sensitivity == "LOW":
        base = _bump(base, -1)
    return base


_HIGH_RULES = {"R-01", "R-02", "R-03", "R-05", "R-07", "R-09", "R-15"}
_ORDER = ["R0", "R1", "R2", "R3", "R4"]


def _bump(grade: str, steps: int, max_grade: str = "R4") -> str:
    i = _ORDER.index(grade)
    j = max(0, min(len(_ORDER) - 1, i + steps))
    j = min(j, _ORDER.index(max_grade))
    return _ORDER[j]
