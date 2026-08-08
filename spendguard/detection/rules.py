"""L1 룰 엔진 + 15개 룰 카탈로그 (PRD 6.2).

룰 파라미터는 코드가 아니라 RULE_CATALOG(운영에서는 rule_configs 테이블)에서
읽는다. 점수는 단순 합산이 아니라 수확 체감(diminishing returns)으로 산출한다.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Callable

from ..config import RULE_VERSION
from ..models import Profile, Reason, Transaction
from ..taxonomy import HIGH_RISK_MCC, category_ko

# ── 룰 카탈로그 (id, name, weight, severity, P, enabled, params, message) ──
RULE_CATALOG = [
    dict(id="R-01", name="개인 최고액 경신", weight=28, severity="HIGH", enabled=True,
         params=dict(max_mult=1.2, min_amount=100_000),
         message="평소 최고 결제액({max:,.0f}원)을 크게 넘는 금액이에요"),
    dict(id="R-02", name="개인 p99 초과", weight=25, severity="HIGH", enabled=True,
         params=dict(min_amount=100_000),
         message="평소 결제 상위 1%({p99:,.0f}원)를 넘는 금액이에요"),
    dict(id="R-03", name="단시간 연속 결제", weight=22, severity="HIGH", enabled=True,
         params=dict(cnt_1h=4, distinct=3, fast_secs=60, fast_amount=50_000),
         message="최근 1시간 동안 {cnt}건이 서로 다른 가맹점에서 결제됐어요"),
    dict(id="R-04", name="일일 한도 급증", weight=20, severity="HIGH", enabled=True,
         params=dict(ratio=5.0),
         message="오늘 하루 지출이 평소의 {ratio:.1f}배예요"),
    dict(id="R-05", name="신규 가맹점 고액", weight=20, severity="HIGH", enabled=True,
         params=dict(min_amount=200_000),
         message="한 번도 이용한 적 없는 가맹점이에요"),
    dict(id="R-06", name="심야 고액 결제", weight=18, severity="MID", enabled=True,
         params=dict(over_p95=2.0, night_ratio_max=0.05),
         message="평소 결제가 거의 없는 새벽 시간대예요"),
    dict(id="R-07", name="해외 거래", weight=30, severity="HIGH", enabled=True,
         params=dict(min_amount=200_000),
         message="해외 가맹점에서 고액 결제가 발생했어요"),
    dict(id="R-08", name="미경험 카테고리 고액", weight=16, severity="MID", enabled=True,
         params=dict(min_amount=150_000),
         message="{category_ko}는 평소 이용하지 않던 분야예요"),
    dict(id="R-09", name="소액 테스트 후 고액", weight=32, severity="HIGH", enabled=True,
         params=dict(small=5_000, window_min=10, big=300_000),
         message="소액 결제 직후 고액 결제가 이어졌어요 (카드 도용 의심 패턴)"),
    dict(id="R-10", name="동일 가맹점 반복 승인", weight=18, severity="MID", enabled=True,
         params=dict(window_min=10, count=3),
         message="같은 가맹점에서 10분 내 {count}건이 반복 결제됐어요"),
    dict(id="R-11", name="카테고리 최대 초과", weight=15, severity="MID", enabled=True,
         params=dict(over_max=2.0, min_n=3),
         message="{category_ko}에서 평소 최고액({cat_max:,.0f}원)의 {ratio:.1f}배예요"),
    dict(id="R-12", name="월 예산 초과 급증", weight=12, severity="LOW", enabled=True,
         params=dict(burn=1.5),
         message="이번 달 지출 진도가 평소보다 빠르게 늘고 있어요"),
    dict(id="R-13", name="신규 지역 결제", weight=14, severity="MID", enabled=True,
         params=dict(min_amount=100_000),
         message="평소 방문하지 않던 지역에서 결제됐어요"),
    dict(id="R-14", name="장기 미사용 후 고액", weight=20, severity="MID", enabled=True,
         params=dict(dormant_days=30, over_p95=2.0),
         message="오랜만의 결제인데 금액이 평소보다 커요"),
    dict(id="R-15", name="고위험 MCC 첫 이용", weight=26, severity="HIGH", enabled=True,
         params=dict(),
         message="귀금속·상품권 등 고위험 업종의 첫 결제예요"),
]

_CATALOG_BY_ID = {r["id"]: r for r in RULE_CATALOG}


# ── 룰 술어(predicate): (feats, tx, profile, db, params) -> bool ─────────
def _r01(f, tx, p, db, pr):
    return (not tx.installment_months and f["amount"] > p.stats.get("max", 0) * pr["max_mult"]
            and f["amount"] >= pr["min_amount"])

def _r02(f, tx, p, db, pr):
    return (f["amt_over_p99"] >= 1.0 and not tx.routine_flag
            and not tx.installment_months and f["amount"] >= pr["min_amount"])

def _r03(f, tx, p, db, pr):
    burst = (f["tx_count_1h"] + 1 >= pr["cnt_1h"]
             and f["distinct_merchants_1h"] + 1 >= pr["distinct"])
    fast = (f["secs_since_prev_tx"] <= pr["fast_secs"] and f["amount"] >= pr["fast_amount"])
    return burst or fast

def _r04(f, tx, p, db, pr):
    return f["sum_24h_over_daily_avg"] >= pr["ratio"]

def _r05(f, tx, p, db, pr):
    return f["merchant_novelty"] == 1.0 and f["amount"] >= pr["min_amount"]

def _r06(f, tx, p, db, pr):
    return (f["is_night"] == 1.0 and f["amt_over_p95"] >= pr["over_p95"]
            and p.time_stats.get("night_ratio", 0) < pr["night_ratio_max"])

def _r07(f, tx, p, db, pr):
    return f["is_overseas"] == 1.0 and f["amount"] >= pr["min_amount"]

def _r08(f, tx, p, db, pr):
    return f["cat_novelty"] == 1.0 and f["amount"] >= pr["min_amount"]

def _r09(f, tx, p, db, pr):
    if f["amount"] < pr["big"]:
        return False
    since = tx.approved_at - timedelta(minutes=pr["window_min"])
    row = db.conn.execute(
        "SELECT 1 FROM transactions WHERE user_id=? AND approved_at >= ? "
        "AND approved_at < ? AND amount < ? AND tx_type='APPROVAL' LIMIT 1",
        (tx.user_id, since.isoformat(), tx.approved_at.isoformat(), pr["small"])).fetchone()
    return row is not None

def _r10(f, tx, p, db, pr):
    if tx.merchant_id is None:
        return False
    since = tx.approved_at - timedelta(minutes=pr["window_min"])
    row = db.conn.execute(
        "SELECT COUNT(*) c FROM transactions WHERE user_id=? AND merchant_id=? "
        "AND approved_at >= ? AND approved_at < ? AND tx_type='APPROVAL'",
        (tx.user_id, tx.merchant_id, since.isoformat(),
         tx.approved_at.isoformat())).fetchone()
    return (row["c"] + 1) >= pr["count"]

def _r11(f, tx, p, db, pr):
    cat = p.cat(tx.category)
    return (not tx.installment_months and f["cat_amt_over_max"] >= pr["over_max"]
            and cat is not None and cat.get("n", 0) >= pr["min_n"])

def _r12(f, tx, p, db, pr):
    return f["budget_burn_deviation"] >= pr["burn"]

def _r13(f, tx, p, db, pr):
    return (tx.channel == "OFFLINE" and f["region_novelty"] == 1.0
            and f["amount"] >= pr["min_amount"])

def _r14(f, tx, p, db, pr):
    dormant = f["secs_since_prev_tx"] >= pr["dormant_days"] * 86400
    return dormant and f["amt_over_p95"] >= pr["over_p95"]

def _r15(f, tx, p, db, pr):
    return tx.mcc in HIGH_RISK_MCC and f["merchant_novelty"] == 1.0


_PREDICATES: dict[str, Callable] = {
    "R-01": _r01, "R-02": _r02, "R-03": _r03, "R-04": _r04, "R-05": _r05,
    "R-06": _r06, "R-07": _r07, "R-08": _r08, "R-09": _r09, "R-10": _r10,
    "R-11": _r11, "R-12": _r12, "R-13": _r13, "R-14": _r14, "R-15": _r15,
}


def _render(rule: dict, feats: dict, tx: Transaction, profile: Profile) -> str:
    ctx = {
        "max": profile.stats.get("max", 0),
        "p99": profile.stats.get("p99", 0),
        "cnt": int(feats["tx_count_1h"] + 1),
        "ratio": feats.get("sum_24h_over_daily_avg", feats.get("cat_amt_over_max", 0)),
        "category_ko": category_ko(tx.category),
        "cat_max": (profile.cat(tx.category) or {}).get("max", 0),
        "count": rule["params"].get("count", 0),
    }
    try:
        return rule["message"].format(**ctx)
    except Exception:
        return rule["name"]


def evaluate_rules(feats: dict, tx: Transaction, profile: Profile, db,
                   active: dict | None = None) -> list[dict]:
    """활성 룰을 평가해 히트 목록을 반환한다."""
    hits = []
    for rule in RULE_CATALOG:
        rid = rule["id"]
        cfg = (active or {}).get(rid, rule)
        if not cfg.get("enabled", True):
            continue
        pred = _PREDICATES[rid]
        try:
            if pred(feats, tx, profile, db, rule["params"]):
                hits.append({
                    "id": rid, "weight": cfg.get("weight", rule["weight"]),
                    "severity": rule["severity"],
                    "label": _render(rule, feats, tx, profile),
                })
        except Exception:
            continue
    return hits


def score_l1(hits: list[dict]) -> tuple[int, list[Reason]]:
    """수확 체감 합산 (PRD 6.2). 1등 100%, 2등 60%, 3등 36% ..."""
    if not hits:
        return 0, []
    ordered = sorted(hits, key=lambda h: -h["weight"])
    raw = sum(h["weight"] * (0.6 ** i) for i, h in enumerate(ordered))
    score = min(100, round(raw * 100 / 45))
    reasons = [Reason(code=h["id"], label=h["label"],
                      contribution=h["weight"] * (0.6 ** i) / raw)
               for i, h in enumerate(ordered)]
    return score, reasons


def active_rule_version() -> str:
    return RULE_VERSION
