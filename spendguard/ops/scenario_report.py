"""시나리오/등급 집계 (PRD 13.2, 13.4 수용기준 대조)."""
from __future__ import annotations

from collections import defaultdict

from simulator.scenarios import EXPECTED_GRADE

GRADES = ["R0", "R1", "R2", "R3", "R4"]
_ORDER = {g: i for i, g in enumerate(GRADES)}


def grade_distribution(results: list[dict]) -> dict:
    dist = {g: 0 for g in GRADES}
    for r in results:
        dist[r["grade"]] = dist.get(r["grade"], 0) + 1
    total = max(len(results), 1)
    return {g: {"count": c, "pct": round(100 * c / total, 3)} for g, c in dist.items()}


def scenario_summary(results: list[dict]) -> dict:
    """시나리오별 판정 등급 분포 + 기대등급 부합 여부."""
    by_scenario: dict[str, list] = defaultdict(list)
    for r in results:
        if r["scenario"]:
            by_scenario[r["scenario"]].append(r)

    out = {}
    for sc, rows in sorted(by_scenario.items()):
        dist = {g: 0 for g in GRADES}
        for r in rows:
            dist[r["grade"]] += 1
        expected = EXPECTED_GRADE.get(sc, set())
        # 다수결 등급
        modal = max(dist, key=dist.get)
        # 부합: 기대 집합 안에 들어오는 비율
        in_expected = sum(1 for r in rows if r["grade"] in expected) / max(len(rows), 1)
        out[sc] = {
            "n": len(rows), "modal_grade": modal,
            "expected": sorted(expected, key=lambda g: _ORDER[g]),
            "match_rate": round(in_expected, 3),
            "distribution": dist,
        }
    return out
