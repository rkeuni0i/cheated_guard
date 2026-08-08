"""SpendGuard 전역 설정 — 임계값·가중치·정책 상수.

PRD의 여러 절에 흩어진 매직 넘버를 한 곳에 모은다. 운영에서는 이 값들이
rule_configs 테이블/환경변수로 이동하지만, 자족형 구현에서는 여기서 관리한다.
값의 근거는 각 주석과 PRD 절 번호로 남긴다.
"""
from __future__ import annotations

# ── 버전 태그 (재현성: risk_assessments에 기록, PRD 3.2 원칙 ③) ─────────
RULE_VERSION = "rules-2026.08.1"
MODEL_VERSION = "iforest-v1-20260808"

# ── 프로파일 윈도우 / 콜드스타트 (PRD 3.3.A, 5.4) ──────────────────────
PROFILE_WINDOW_DAYS = 90
COLD_START_MIN_TX = 30          # 이 미만이면 콜드스타트
COLD_START_MIN_DAYS = 14
BLEND_FULL_TX = 100            # peer 혼합 가중 w = n/100, 이 이상이면 개인 100%
CAT_FALLBACK_MIN_N = 5         # 카테고리 표본이 이 미만이면 fallback

# ── 스코어 융합 (PRD 6.5) ──────────────────────────────────────────────
BASE_WEIGHTS = {"l1": 0.35, "l2": 0.35, "l3": 0.30}
FALLBACK_WEIGHTS = {"l1": 0.54, "l2": 0.46, "l3": 0.0}   # L3 실패 시 재정규화
STRONGEST_MIX = 0.25          # score = (1-x)*base + x*max(layer)
L3_TIMEOUT_MS = 300           # L3 예산 (자족형에서는 상징적)

# ── L2 통계 (PRD 6.3) ──────────────────────────────────────────────────
L2_WEIGHTS = {"amount": 0.40, "context": 0.30, "time": 0.12, "velocity": 0.18}
L2_SOFTMAX_MIX = 0.30         # final = (1-x)*weighted + x*max_axis (6.3 각주)
MAD_LOG_FLOOR = 0.35          # 지나치게 균일한 사용자 보호 (axis_amount)
ROUTINE_DAMP = 0.3            # routine 거래 감쇠 배수
ROUTINE_DAMP_RELEASE = 3.0    # 평소 median의 이 배 넘으면 감쇠 해제

# ── 리스크 등급 경계 (PRD 7.1) ─────────────────────────────────────────
GRADE_BOUNDS = [("R4", 90), ("R3", 70), ("R2", 50), ("R1", 30), ("R0", 0)]

# ── 알림 피로도 (PRD 7.3) ──────────────────────────────────────────────
DAILY_ALERT_CAP = 2           # R1+R2 합산 일일 상한 (R3·R4는 무제한)
SUPPRESS_DAYS = 90            # "내가 했어요" 조합 강등 기간
COOLDOWN_HOURS = 24           # 동일 가맹점 재알림 억제

# ── 임계값 튜닝 비용비 (PRD 6.7) ───────────────────────────────────────
COST_RATIO_FN_FP = {"LOW": 5, "DEFAULT": 10, "HIGH": 20}

# ── 민감도 프리셋 (PRD 2.1, 7.1) ───────────────────────────────────────
SENSITIVITY_PRESETS = ("LOW", "DEFAULT", "HIGH")

# ── 드리프트 (PRD 10.3) ────────────────────────────────────────────────
PSI_WARN = 0.10
PSI_ALERT = 0.25

# ── 시간대 (PRD 표기 규약: Asia/Seoul) ─────────────────────────────────
TZ = "Asia/Seoul"
NIGHT_HOURS = range(0, 6)     # 00~06시 = 심야
