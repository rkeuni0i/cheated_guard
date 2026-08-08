"""가맹점명 정규화 + 카테고리 매핑 (PRD 4.6).

정규화 규칙: 소문자화 → 괄호 제거 → 법인격/지점명 제거 → 특수문자 제거.
PG 접두어(COUPANG PAY* 등)는 별칭 사전으로 수동 매핑한다.
"""
from __future__ import annotations

import re

from ..models import Merchant, Transaction
from ..taxonomy import mcc_to_category, category_variance

# 가맹점 별칭 사전 (PRD 4.6: merchant_aliases 테이블의 자족형 이식)
_ALIASES = {
    "coupangpay": "coupang", "coupangpay주문": "coupang", "naverpay": "naver",
    "kakaopay": "kakao", "배민": "baemin", "요기요": "yogiyo",
}

_RE_PAREN = re.compile(r"[（(].*?[）)]")
_RE_CORP = re.compile(r"(주식회사|㈜|주\)|\(주\))")
_RE_BRANCH = re.compile(r"\s*\d+호점|\s*[가-힣a-z]+점$|\s*\d+점$")
_RE_NONWORD = re.compile(r"[^가-힣a-z0-9]")


def normalize_merchant(raw: str) -> str:
    """가맹점 원문 → 정규화 키. 빈 문자열이면 UNKNOWN_MERCHANT (DQ-07)."""
    if not raw:
        return "UNKNOWN_MERCHANT"
    s = raw.strip().lower()
    s = _RE_PAREN.sub("", s)
    s = _RE_CORP.sub("", s)
    s = _RE_BRANCH.sub("", s)
    s = _RE_NONWORD.sub("", s)
    s = _ALIASES.get(s, s)
    return s or "UNKNOWN_MERCHANT"


def build_merchant(tx: Transaction) -> Merchant:
    """거래에서 가맹점 마스터 레코드를 도출한다."""
    norm = normalize_merchant(tx.merchant_name)
    category = mcc_to_category(tx.mcc)
    return Merchant(
        name_raw=tx.merchant_name,
        name_normalized=norm,
        mcc=tx.mcc,
        category=category,
        variance_class=category_variance(category),
    )


def normalize_transaction(tx: Transaction) -> Transaction:
    """카테고리·정규화명 등 파생 필드를 채운다 (DB 저장 전)."""
    tx.merchant_norm = normalize_merchant(tx.merchant_name)
    tx.category = mcc_to_category(tx.mcc)
    return tx
