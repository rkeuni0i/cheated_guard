"""알림 카피 생성 (PRD 7.3).

원칙: 사실 → 근거 → 행동. 단정하지 않고, 숫자를 반드시 넣고, 공포를 조장하지 않는다.
R4는 '정지했어요'(이미 조치)를 먼저 알린다.
"""
from __future__ import annotations

from ..models import Assessment, Transaction

_TITLES = {
    "R2": "오늘 지출이 평소보다 많아요",
    "R3": "평소와 다른 결제가 확인됐어요",
    "R4": "카드 사용을 임시로 중단했어요",
}


def build_message(assessment: Assessment, tx: Transaction) -> str:
    """푸시/인앱 본문 생성."""
    title = _TITLES.get(assessment.grade, "결제 알림")
    lines = [title,
             f"{tx.merchant_name} {tx.amount:,}원 "
             f"({tx.approved_at.strftime('%m/%d %H:%M')})"]
    for r in assessment.reasons:
        lines.append(f"· {r.label}")
    if assessment.grade == "R4":
        lines.append("안전을 위해 카드를 정지했습니다.")
        lines.append("[본인 결제예요, 해제하기]  [도난·분실 신고]")
    elif assessment.grade == "R3":
        lines.append("[내가 한 결제예요]  [내가 안 했어요]")
    return "\n".join(lines)


def build_sms(assessment: Assessment, tx: Transaction) -> str:
    return (f"[SpendGuard] {tx.merchant_name} {tx.amount:,}원 결제 확인 필요. "
            f"앱에서 확인해 주세요.")
