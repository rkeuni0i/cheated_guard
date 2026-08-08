"""설명가능성 — 근거 생성 (PRD 6.8).

세 계층의 근거를 사람의 문장으로 변환하고, 중복 제거 후 기여도 순 최대 3개.
숫자는 반드시 실제값을 바인딩한다 ('이상한 금액입니다' 같은 공허한 문구 금지).
"""
from __future__ import annotations

from ..models import Profile, Reason, Transaction
from ..taxonomy import category_ko

# 같은 현상을 가리키는 근거는 하나로 묶는다 (예: R-01·R-02는 '고액' 패밀리)
REASON_FAMILY = {
    "R-01": "AMOUNT", "R-02": "AMOUNT", "L2-AMOUNT": "AMOUNT",
    "R-11": "CAT_AMOUNT", "L2-CONTEXT": "CAT_AMOUNT",
    "R-05": "NEW_MERCHANT", "L3-merchant_novelty": "NEW_MERCHANT",
    "R-08": "NEW_CATEGORY",
    "R-03": "VELOCITY", "R-04": "VELOCITY", "R-10": "VELOCITY", "L2-VELOCITY": "VELOCITY",
    "R-06": "NIGHT", "L2-TIME": "NIGHT",
    "R-07": "OVERSEAS", "R-13": "REGION",
    "R-09": "TEST_PATTERN", "R-14": "DORMANT", "R-15": "HIGH_RISK",
}

_AXIS_TEMPLATES = {
    "amount": ("L2-AMOUNT", "평소 결제 범위(중앙값 {median:,.0f}원)에서 크게 벗어났어요"),
    "context": ("L2-CONTEXT", "{category_ko}에서 평소 최고액({cat_max:,.0f}원)의 {ratio:.1f}배예요"),
    "time": ("L2-TIME", "평소 결제가 드문 시간대예요"),
    "velocity": ("L2-VELOCITY", "짧은 시간에 지출이 평소보다 몰렸어요"),
}


def l2_reasons(axes: dict, tx: Transaction, profile: Profile,
               k: int = 2) -> list[Reason]:
    """L2 기여도 상위 축을 문장으로 (0점 축 제외)."""
    ranked = sorted(axes.items(), key=lambda kv: -kv[1])
    out = []
    cat = profile.cat(tx.category) or {}
    ctx = {
        "median": profile.stats.get("median", 0),
        "category_ko": category_ko(tx.category),
        "cat_max": cat.get("max", 0) or 1,
        "ratio": tx.amount / (cat.get("max", 0) or 1),
    }
    for name, val in ranked[:k]:
        if val < 30:
            continue
        code, tmpl = _AXIS_TEMPLATES[name]
        try:
            label = tmpl.format(**ctx)
        except Exception:
            label = tmpl
        out.append(Reason(code=code, label=label, contribution=val / 100.0))
    return out


def l3_reasons(top_features: list[tuple[str, float]]) -> list[Reason]:
    """L3 기여 피처 → 사전 정의 문구 (SHAP 근사)."""
    templates = {
        "merchant_novelty": "처음 이용하는 가맹점이라는 점이 크게 작용했어요",
        "amt_over_p95": "평소 대비 큰 금액이라는 점이 작용했어요",
        "is_night": "새벽 시간대라는 점이 작용했어요",
        "is_overseas": "해외 결제라는 점이 작용했어요",
        "cat_amt_over_max": "이 분야에서 이례적으로 큰 금액이에요",
    }
    out = []
    for feat, contrib in top_features:
        if feat in templates:
            out.append(Reason(code=f"L3-{feat}", label=templates[feat],
                              contribution=float(contrib)))
    return out


def rank_and_humanize(reasons: list[Reason], limit: int = 3) -> list[Reason]:
    """패밀리 중복 제거 → 기여도 순 → 최대 limit개 (PRD 6.8)."""
    dedup: dict[str, Reason] = {}
    for r in sorted(reasons, key=lambda x: -x.contribution):
        family = REASON_FAMILY.get(r.code, r.code)
        if family not in dedup:
            dedup[family] = r
    return list(dedup.values())[:limit]
