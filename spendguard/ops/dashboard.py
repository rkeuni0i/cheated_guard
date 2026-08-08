"""운영 대시보드 — 자족형 HTML 리포트 (PRD 8.3 운영 대시보드, 14장 지표).

Streamlit 서버 대신 정적 HTML을 생성한다(경량 자족형). 등급 분포, 시나리오별
탐지 결과, 성능 지표, 드리프트를 한 페이지에 담는다.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

_CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Malgun Gothic',sans-serif;
 background:#f7f8fa;color:#16181d;margin:0;padding:32px;line-height:1.6}
.wrap{max-width:1000px;margin:0 auto}
h1{font-size:26px;margin:0 0 4px} .sub{color:#7c8496;margin:0 0 28px;font-size:14px}
h2{font-size:18px;margin:34px 0 12px;border-top:2px solid #16181d;padding-top:12px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.kpi{background:#fff;border:1px solid #e3e6ec;border-radius:10px;padding:14px 16px}
.kpi .k{font-size:11px;color:#7c8496;font-weight:700} .kpi .v{font-size:22px;font-weight:800;color:#3d5afe}
.kpi .d{font-size:11px;color:#7c8496}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e3e6ec;
 border-radius:8px;overflow:hidden;font-size:13px;margin-bottom:8px}
th{background:#f0f2f6;text-align:left;padding:9px 11px;font-size:12px}
td{padding:9px 11px;border-top:1px solid #eef0f4}
.pass{color:#0f9d58;font-weight:700}.fail{color:#d93025;font-weight:700}
.bar{height:14px;background:#eef1ff;border-radius:3px;overflow:hidden;display:inline-block;width:120px;vertical-align:middle}
.bar>i{display:block;height:100%;background:#3d5afe}
.badge{font-family:monospace;font-size:11px;padding:2px 7px;border-radius:4px}
.R0{background:#eef0f4;color:#5a6274}.R1{background:#eef1ff;color:#2c3fb8}
.R2{background:#fff5e5;color:#8a5300}.R3{background:#fff5e5;color:#c77700}
.R4{background:#fdecea;color:#a5150c}
.note{background:#fff5e5;border:1px solid #f2dcb0;border-radius:8px;padding:12px 16px;
 font-size:13px;color:#7a4d00;margin:16px 0}
"""


def _kpi(k, v, d=""):
    return f'<div class="kpi"><div class="k">{k}</div><div class="v">{v}</div><div class="d">{d}</div></div>'


def _fmt(x, pct=False):
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{x*100:.1f}%" if pct else (f"{x:.3f}" if isinstance(x, float) else str(x))


def render_dashboard(summary: dict, out_path: str | Path,
                     meta: dict | None = None) -> str:
    m = summary["metrics"]
    meta = meta or {}
    parts = ['<div class="wrap">']
    parts.append("<h1>SpendGuard — 백테스트 리포트</h1>")
    parts.append(f'<p class="sub">합성 데이터 기준 · ML {"ON" if summary["ml_enabled"] else "OFF"} · '
                 f'거래 {m["n"]:,}건 (이상 {m["positives"]}건) · '
                 f'rule {html.escape(meta.get("rule_version",""))}</p>')

    # KPI (PRD 1.3 / 10.2)
    parts.append("<h2>핵심 성능 지표</h2><div class='kpis'>")
    parts.append(_kpi("PR-AUC", _fmt(m["pr_auc"]), "불균형 데이터 주 지표"))
    parts.append(_kpi("Event Recall (R3+)", _fmt(m.get("event_recall_R3+"), True),
                      "사건 단위 · 카드정지 기준"))
    parts.append(_kpi("Recall @ FPR 0.5%", _fmt(m["recall_at_fpr_0.5%"], True), "건 단위 · 목표 ≥ 85%"))
    parts.append(_kpi("Precision @ top 0.5%", _fmt(m["precision_at_top_0.5%"], True), "목표 ≥ 60%"))
    parts.append(_kpi("Lift @ 1%", _fmt(m["lift_at_1%"]) + "×", "목표 ≥ 20×"))
    parts.append(_kpi("ROC-AUC", _fmt(m["roc_auc"]), ""))
    parts.append("</div>")

    # 등급 분포
    parts.append("<h2>등급 분포 (PRD 7.1)</h2><table><tr><th>등급</th><th>건수</th><th>비율</th><th></th></tr>")
    for g, info in summary["grade_distribution"].items():
        w = min(info["pct"] * 3, 100)
        parts.append(f'<tr><td><span class="badge {g}">{g}</span></td>'
                     f'<td>{info["count"]:,}</td><td>{info["pct"]:.2f}%</td>'
                     f'<td><span class="bar"><i style="width:{w}%"></i></span></td></tr>')
    parts.append("</table>")

    # 시나리오별 (PRD 13.2)
    parts.append("<h2>시나리오별 탐지 (PRD 13.2 / AC-CORE)</h2>")
    parts.append("<table><tr><th>SC</th><th>n</th><th>다수결 등급</th>"
                 "<th>기대 등급</th><th>부합률</th><th>결과</th></tr>")
    for sc, info in summary["scenarios"].items():
        ok = info["match_rate"] >= 0.5
        cls = "pass" if ok else "fail"
        exp = " / ".join(info["expected"])
        parts.append(f'<tr><td>{sc}</td><td>{info["n"]}</td>'
                     f'<td><span class="badge {info["modal_grade"]}">{info["modal_grade"]}</span></td>'
                     f'<td>{exp}</td><td>{info["match_rate"]*100:.0f}%</td>'
                     f'<td class="{cls}">{"PASS" if ok else "CHECK"}</td></tr>')
    parts.append("</table>")

    if summary.get("drift"):
        parts.append("<h2>드리프트 (PSI, PRD 10.3)</h2><table>"
                     "<tr><th>피처</th><th>PSI</th><th>상태</th></tr>")
        for f, info in summary["drift"].items():
            parts.append(f'<tr><td>{f}</td><td>{info["psi"]}</td><td>{info["status"]}</td></tr>')
        parts.append("</table>")

    parts.append('<div class="note">⚠️ 합성 데이터 기준 성능입니다. 생성 가정은 '
                 'simulator/generator.py(PRD 부록 A)에 명시되어 있으며, 같은 생성기가 만든 '
                 '이상을 탐지하므로 실제 성능을 과대평가할 수 있습니다(PRD 부록 A 경고).</div>')
    parts.append("</div>")

    doc = (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
           f"<title>SpendGuard 백테스트 리포트</title><style>{_CSS}</style></head>"
           f"<body>{''.join(parts)}</body></html>")
    Path(out_path).write_text(doc, encoding="utf-8")
    return str(out_path)
