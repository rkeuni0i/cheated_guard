"""카테고리 택소노미 및 MCC 매핑 (PRD 4.5).

14종 내부 카테고리 + 대표 MCC → 카테고리 매핑, 분산등급, 한글 라벨.
"""
from __future__ import annotations

# 코드 → (한글명, 분산등급, routine 성향)
CATEGORIES = {
    "FOOD":         ("식비·외식",   "LOW",  False),
    "GROCERY":      ("마트·식료품", "HIGH", False),
    "CAFE":         ("카페·베이커리", "LOW", False),
    "TRANSPORT":    ("교통",        "LOW",  True),
    "FUEL":         ("주유",        "MID",  False),
    "BEAUTY":       ("뷰티·헬스케어", "HIGH", False),
    "FASHION":      ("의류·잡화",   "HIGH", False),
    "ELECTRONICS":  ("전자·컴퓨터", "HIGH", False),
    "ECOMMERCE":    ("종합 온라인몰", "HIGH", False),
    "MEDICAL":      ("병원·약국",   "MID",  False),
    "ENTERTAIN":    ("문화·여가",   "MID",  False),
    "SUBSCRIPTION": ("구독·통신",   "LOW",  True),
    "TRAVEL":       ("여행·숙박",   "HIGH", False),
    "ETC":          ("기타",        "MID",  False),
}

# 대표 MCC → 카테고리 (PRD 4.5 표)
_MCC_MAP = {
    "5812": "FOOD", "5814": "CAFE", "5499": "GROCERY", "5411": "GROCERY",
    "4111": "TRANSPORT", "4121": "TRANSPORT",
    "5541": "FUEL", "5542": "FUEL",
    "5977": "BEAUTY", "7230": "BEAUTY",
    "5651": "FASHION", "5661": "FASHION",
    "5732": "ELECTRONICS", "5734": "ELECTRONICS", "5045": "ELECTRONICS",
    "5964": "ECOMMERCE", "5999": "ECOMMERCE",
    "8062": "MEDICAL", "5912": "MEDICAL",
    "7832": "ENTERTAIN", "7999": "ENTERTAIN",
    "4899": "SUBSCRIPTION", "5968": "SUBSCRIPTION",
    "7011": "TRAVEL",
}

# 고위험 MCC (R-15: 귀금속·상품권·현금서비스성)
HIGH_RISK_MCC = {"5944", "5933", "6051", "7995"}  # 귀금속/전당포/유사현금/도박


def mcc_to_category(mcc: str) -> str:
    """MCC 4자리 → 내부 카테고리. 미상은 ETC (DQ-04)."""
    if not mcc:
        return "ETC"
    mcc = str(mcc).strip()
    if mcc in _MCC_MAP:
        return _MCC_MAP[mcc]
    # 여행 MCC 범위 3000-3999 (항공/호텔)
    if mcc.isdigit() and 3000 <= int(mcc) <= 3999:
        return "TRAVEL"
    return "ETC"


def category_variance(category: str) -> str:
    """카테고리 기본 분산등급 (LOW|MID|HIGH)."""
    return CATEGORIES.get(category, ("", "MID", False))[1]


def category_ko(category: str) -> str:
    """사용자 노출용 한글 카테고리명 (PRD 6.8 근거 문구)."""
    return CATEGORIES.get(category, ("기타", "", False))[0]


def is_routine_category(category: str) -> bool:
    """구조적으로 routine 성향이 강한 카테고리 여부 (교통·구독)."""
    return CATEGORIES.get(category, ("", "", False))[2]


VALID_CATEGORIES = frozenset(CATEGORIES.keys())
