"""금액대 버킷 — 억제 규칙 키 (PRD 7.3 '가맹점×금액대' 조합).

fusion(override 판정)과 feedback(억제 등록)이 같은 정의를 공유해야 하므로
별도 모듈로 둔다.
"""
from __future__ import annotations

BAND_SIZE = 50_000


def amount_band(amount: int) -> int:
    return int(amount) // BAND_SIZE
