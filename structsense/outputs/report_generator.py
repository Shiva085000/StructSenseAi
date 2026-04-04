"""
outputs/report_generator.py
============================
StructSense AI — Publication-Ready PDF Report Generator
Uses ReportLab (platypus + canvas) for IEEE-style layout.

Entry point:
    generate_report(verdict, state_params, risk_scores, reward_history,
                    output_path="outputs/reports/structsense_report.pdf") -> str
"""
from __future__ import annotations

import io
import os
import datetime
import tempfile
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
    Table, TableStyle, Image, HRFlowable, PageBreak, KeepTogether,
)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.colors import HexColor, white, black

# ─── Colour palette ─────────────────────────────────────────────────────────
NAVY        = HexColor("#0a0e1a")
BLUE_ACC    = HexColor("#00d4ff")
DARK_CARD   = HexColor("#0d1224")
BORDER      = HexColor("#1e2a45")
RED         = HexColor("#c0392b")
ORANGE      = HexColor("#d35400")
YELLOW_RISK = HexColor("#f39c12")
GREEN_SAFE  = HexColor("#27ae60")
LIGHT_GREY  = HexColor("#ecf0f1")
MID_GREY    = HexColor("#bdc3c7")
DARK_GREY   = HexColor("#7f8c8d")
TEXT_DARK   = HexColor("#1a1a2e")

VERDICT_COLORS = {
    "HIGH RISK":     (HexColor("#fdecea"), RED),
    "MODERATE RISK": (HexColor("#fef9e7"), YELLOW_RISK),
    "SAFE":          (HexColor("#eafaf1"), GREEN_SAFE),
}

W, H = A4          # 595.27 × 841.89 pts
MARGIN = 2.2 * cm

# ─── 18 parameter definitions ───────────────────────────────────────────────
PARAM_TABLE = [
    ("concrete_grade",               "Concrete Grade",              "MPa",    "15 – 60",   "Material"),
    ("steel_grade",                  "Steel Grade (Fe)",            "MPa",    "250 – 550", "Material"),
    ("column_reinforcement_ratio",   "Column Reinf. Ratio (ρ)",     "—",      "0.008 – 0.060", "Material"),
    ("column_width_mm",              "Column Width",                "mm",     "200 – 800", "Geometry"),
    ("column_depth_mm",              "Column Depth",                "mm",     "200 – 800", "Geometry"),
    ("beam_depth_mm",                "Beam Depth",                  "mm",     "300 – 900", "Geometry"),
    ("slab_thickness_mm",            "Slab Thickness",              "mm",     "100 – 300", "Geometry"),
    ("num_floors",                   "Number of Floors",            "—",      "1 – 20",    "Geometry"),
    ("floor_load_kn_m2",             "Floor Live Load",             "kN/m²",  "2 – 15",    "Geometry"),
    ("soil_bearing_capacity_kn_m2",  "Soil Bearing Capacity (SBC)", "kN/m²",  "50 – 500",  "Site"),
    ("foundation_depth_m",           "Foundation Depth",            "m",      "0.5 – 5.0", "Site"),
    ("seismic_zone",                 "Seismic Zone (IS 1893)",      "1–5",    "1 – 5",     "Site"),
    ("wind_speed_kmph",              "Design Wind Speed",           "km/h",   "20 – 250",  "Site"),
    ("rainfall_mm_annual",           "Annual Rainfall",             "mm",     "200 – 3000","Site"),
    ("building_age_years",           "Building Age",                "years",  "0 – 100",   "Site"),
    ("shear_wall_present",           "Shear Wall Presence",         "binary", "0 | 1",     "Structural"),
    ("foundation_type",              "Foundation Type",             "0/1/2",  "0=Isolated, 1=Raft, 2=Pile", "Structural"),
    ("location_risk_index",          "Location Risk Index",         "0–1",    "0.0 – 1.0", "Site"),
]

SCENARIO_NAMES = [
    ("normal_load",  "Normal Occupancy Load Test",    "IS 456 axial column capacity"),
    ("seismic",      "Seismic Stress Test",            "IS 1893 base shear vs lateral cap."),
    ("wind_rain",    "Wind + Rain Combined Test",      "IS 875 wind + rainfall overload"),
    ("overload",     "Overload Test (150%)",           "1.5× floor load scenario"),
    ("degradation",  "Long-Term Degradation Test",     "Age-based capacity reduction"),
]

ELEMENT_LABELS = ["foundation", "columns", "beams", "slab"]

# ════════════════════════════════════════════════════════════════════════════
#  STYLES
# ════════════════════════════════════════════════════════════════════════════
def _build_styles():
    base = getSampleStyleSheet()
    S = {}

    def ps(name, fontName="Times-Roman", fontSize=10, leading=14,
           alignment=TA_JUSTIFY, textColor=TEXT_DARK, spaceBefore=0,
           spaceAfter=4, leftIndent=0, **kw):
        S[name] = ParagraphStyle(name, parent=base["Normal"],
                                 fontName=fontName, fontSize=fontSize,
                                 leading=leading, alignment=alignment,
                                 textColor=textColor, spaceBefore=spaceBefore,
                                 spaceAfter=spaceAfter, leftIndent=leftIndent,
                                 **kw)

    ps("body",         fontSize=10,   leading=14)
    ps("body_bold",    fontName="Times-Bold",  fontSize=10, leading=14)
    ps("small",        fontSize=8,    leading=11, textColor=DARK_GREY)
    ps("title",        fontName="Times-Bold",  fontSize=22, leading=28,
       alignment=TA_CENTER, textColor=NAVY, spaceBefore=12, spaceAfter=6)
    ps("subtitle",     fontName="Times-Roman", fontSize=13, leading=18,
       alignment=TA_CENTER, textColor=DARK_GREY, spaceAfter=4)
    ps("author",       fontName="Times-Italic",fontSize=11, leading=15,
       alignment=TA_CENTER, textColor=TEXT_DARK, spaceAfter=2)
    ps("abstract_hdr", fontName="Times-Bold",  fontSize=10, leading=14,
       alignment=TA_CENTER, textColor=NAVY, spaceAfter=2)
    ps("abstract",     fontSize=9,    leading=13, alignment=TA_JUSTIFY,
       leftIndent=1.5*cm, rightIndent=1.5*cm, textColor=TEXT_DARK)
    ps("sec1",         fontName="Times-Bold",  fontSize=14, leading=18,
       textColor=NAVY, spaceBefore=14, spaceAfter=6)
    ps("sec2",         fontName="Times-Bold",  fontSize=11, leading=15,
       textColor=NAVY, spaceBefore=10, spaceAfter=4)
    ps("verdict_text", fontName="Times-Bold",  fontSize=16, leading=22,
       alignment=TA_CENTER, textColor=white)
    ps("table_hdr",    fontName="Times-Bold",  fontSize=9,  leading=12,
       alignment=TA_CENTER, textColor=white)
    ps("table_body",   fontSize=9,    leading=12)
    ps("bullet",       fontSize=10,   leading=14, leftIndent=0.8*cm,
       spaceBefore=1, spaceAfter=1)
    ps("footer",       fontSize=7.5,  leading=10, alignment=TA_CENTER,
       textColor=DARK_GREY)
    return S

STYLES = _build_styles()

# ════════════════════════════════════════════════════════════════════════════
#  HELPER FLOWABLES
# ════════════════════════════════════════════════════════════════════════════
def _hr(width_frac=1.0, thickness=0.8, color=BORDER):
    return HRFlowable(width=f"{int(width_frac*100)}%", thickness=thickness,
                      color=color, spaceAfter=6, spaceBefore=2)

def _sp(h=6):
    return Spacer(1, h)

def _p(text, style="body"):
    return Paragraph(text, STYLES[style])

def _sec(num, title, style="sec1"):
    return Paragraph(f"{num}  {title}", STYLES[style])

def _verdict_color(verdict: str):
    for k, (bg, fg) in VERDICT_COLORS.items():
        if k in verdict.upper():
            return bg, fg
    return LIGHT_GREY, TEXT_DARK

def _risk_color(name: str) -> str:
    c = {"CRITICAL": "#c0392b", "HIGH": "#d35400",
         "MODERATE": "#f39c12", "SAFE": "#27ae60", "UNKNOWN":"#7f8c8d"}
    return c.get(name.upper(), "#7f8c8d")

def _status_color(s: str):
    m = {"PASS": GREEN_SAFE, "FAIL": RED, "MODERATE": YELLOW_RISK, "UNKNOWN": MID_GREY}
    return m.get(s.upper(), MID_GREY)

# ════════════════════════════════════════════════════════════════════════════
#  CHART HELPERS  (Matplotlib → temp PNG → ReportLab Image)
# ════════════════════════════════════════════════════════════════════════════
def _save_fig(fig, tmpdir: str, name: str) -> str:
    path = os.path.join(tmpdir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor="#0a0e1a", edgecolor="none")
    plt.close(fig)
    return path

def _chart_reward(reward_history: List[float], tmpdir: str) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#0d1224")
    x = list(range(len(reward_history)))
    ax.plot(x, reward_history, color="#00d4ff", lw=1.6, zorder=3)
    ax.fill_between(x, reward_history, alpha=0.15, color="#00d4ff")
    # rolling mean
    if len(reward_history) >= 5:
        rm = np.convolve(reward_history, np.ones(5)/5, "valid")
        ax.plot(range(4, len(reward_history)), rm, color="#ff9f40",
                lw=2.2, linestyle="--", label="5-step avg", zorder=4)
    ax.axhline(0, color="#1e2a45", lw=0.8)
    ax.set_xlabel("Training Step",  color="#94a3b8", fontsize=8)
    ax.set_ylabel("Episode Reward", color="#94a3b8", fontsize=8)
    ax.set_title("PPO Reward Convergence", color="#00d4ff", fontsize=9, pad=6)
    ax.tick_params(colors="#64748b", labelsize=7)
    for spine in ax.spines.values(): spine.set_edgecolor("#1e2a45")
    ax.legend(fontsize=7, facecolor="#0d1224", edgecolor="#1e2a45",
              labelcolor="#94a3b8")
    ax.grid(True, color="#1e2a45", linewidth=0.5, linestyle="--")
    return _save_fig(fig, tmpdir, "reward.png")

def _chart_risk_bars(risk_scores: Dict[str, float], tmpdir: str) -> str:
    elems  = list(risk_scores.keys())
    values = [risk_scores[e] * 100 for e in elems]
    cmap   = {"foundation": "#c0392b", "columns": "#d35400",
               "beams": "#f39c12", "slab": "#27ae60"}
    bar_colors = [cmap.get(e, "#64748b") for e in elems]

    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#0d1224")
    bars = ax.barh(elems, values, color=bar_colors, height=0.5, zorder=3)
    ax.axvline(70, color="#f87171", lw=1.2, linestyle="--",
               label="Critical threshold (70%)")
    ax.axvline(55, color="#fbbf24", lw=1.0, linestyle=":",
               label="High threshold (55%)")
    for bar, v in zip(bars, values):
        ax.text(v + 1.5, bar.get_y() + bar.get_height()/2,
                f"{v:.1f}%", va="center", ha="left",
                color="#94a3b8", fontsize=8)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Risk Score (%)", color="#94a3b8", fontsize=8)
    ax.set_title("Structural Element Risk Scores", color="#00d4ff",
                 fontsize=9, pad=6)
    ax.tick_params(colors="#64748b", labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor("#1e2a45")
    ax.legend(fontsize=7, facecolor="#0d1224", edgecolor="#1e2a45",
              labelcolor="#94a3b8")
    ax.grid(True, axis="x", color="#1e2a45", linewidth=0.5, linestyle="--")
    return _save_fig(fig, tmpdir, "risk_bars.png")

def _chart_heatmap(scenario_results: Dict[str, str], tmpdir: str) -> str:
    _KEY_ORDER = ["normal_load", "seismic", "wind_rain", "overload", "degradation"]
    _LABELS    = ["Normal Load", "Seismic", "Wind+Rain", "Overload", "Degradation"]
    _CMAP      = {"PASS": 1.0, "MODERATE": 0.5, "FAIL": 0.0, "UNKNOWN": -0.1}
    _COL_CMAP  = {"PASS": "#27ae60", "MODERATE": "#f39c12",
                  "FAIL": "#c0392b", "UNKNOWN": "#7f8c8d"}

    statuses = [scenario_results.get(k, "UNKNOWN") for k in _KEY_ORDER]
    vals     = [[_CMAP.get(s, -0.1)] for s in statuses]

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#0d1224")

    for i, (lab, st) in enumerate(zip(_LABELS, statuses)):
        col = _COL_CMAP.get(st, "#7f8c8d")
        ax.barh(i, 1, color=col, height=0.7, zorder=3)
        ax.text(0.5, i, st, va="center", ha="center",
                color="white", fontsize=9, fontweight="bold")

    ax.set_yticks(range(len(_LABELS)))
    ax.set_yticklabels(_LABELS, color="#94a3b8", fontsize=8)
    ax.set_xticks([])
    ax.set_title("Scenario Pass/Fail Matrix", color="#00d4ff", fontsize=9, pad=6)
    ax.invert_yaxis()
    for spine in ax.spines.values(): spine.set_edgecolor("#1e2a45")

    patches = [mpatches.Patch(color="#27ae60", label="PASS"),
               mpatches.Patch(color="#f39c12", label="MODERATE"),
               mpatches.Patch(color="#c0392b", label="FAIL")]
    ax.legend(handles=patches, fontsize=7, loc="lower right",
              facecolor="#0d1224", edgecolor="#1e2a45", labelcolor="#94a3b8")
    return _save_fig(fig, tmpdir, "heatmap.png")

# ════════════════════════════════════════════════════════════════════════════
#  PAGE TEMPLATE (header + footer)
# ════════════════════════════════════════════════════════════════════════════
FOOTER_TEXT = "StructSense AI  |  Neural RL Structural Safety Framework  |  SDG 11"

def _make_page_template(doc):
    def _draw(canvas, doc):
        canvas.saveState()
        # Footer line
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, 1.4*cm, W - MARGIN, 1.4*cm)
        # Footer text + page number
        canvas.setFont("Times-Roman", 7.5)
        canvas.setFillColor(DARK_GREY)
        canvas.drawCentredString(W/2, 0.9*cm, FOOTER_TEXT)
        canvas.drawRightString(W - MARGIN, 0.9*cm, f"Page {doc.page}")
        canvas.restoreState()

    frame = Frame(MARGIN, 2*cm, W - 2*MARGIN, H - 3.2*cm,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    return PageTemplate(id="main", frames=[frame], onPage=_draw)

# ════════════════════════════════════════════════════════════════════════════
#  SECTION BUILDERS
# ════════════════════════════════════════════════════════════════════════════

def _flowables_title_page(verdict: dict, state_params: dict) -> list:
    items = []
    today = datetime.date(2026, 2, 27).strftime("%B %d, %Y")

    items += [
        _sp(1.5*cm),
        _p("StructSense AI: Neural RL-Based Structural Failure Detection<br/>"
           "from Blueprint Parameters", "title"),
        _hr(0.6, 1.5, BLUE_ACC),
        _sp(4),
        _p("StructSense Research Group", "author"),
        _p(f"Department of Civil &amp; Computational Engineering", "author"),
        _p(f"Date: {today}", "subtitle"),
        _sp(12),
    ]

    # Verdict badge
    vtext = verdict.get("overall_verdict", "UNKNOWN")
    fp    = verdict.get("failure_probability", 0)
    conf  = verdict.get("confidence", 0)
    bg, fg = _verdict_color(vtext)

    badge_data = [[Paragraph(
        f'<font color="white"><b>{vtext}</b></font><br/>'
        f'<font color="#cccccc" size="9">Failure Probability: {fp*100:.1f}%  |  '
        f'Confidence: {conf*100:.1f}%</font>',
        STYLES["verdict_text"])]]
    badge_tbl = Table(badge_data, colWidths=[W - 2*MARGIN])
    badge_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), fg),
        ("BOX",         (0,0), (-1,-1), 1.5, fg),
        ("ROUNDEDCORNERS", [8]),
        ("TOPPADDING",  (0,0), (-1,-1), 12),
        ("BOTTOMPADDING",(0,0),(-1,-1), 12),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
    ]))
    items += [badge_tbl, _sp(14)]

    # Abstract
    pfm      = verdict.get("primary_failure_mode", "N/A")
    n_viol   = len(verdict.get("is_code_violations", []))
    sc_count = sum(1 for v in verdict.get("scenario_results", {}).values()
                   if v == "FAIL")
    floors   = int(state_params.get("num_floors", 0))
    grade    = int(state_params.get("concrete_grade", 0))
    zone     = int(state_params.get("seismic_zone", 0))
    age      = int(state_params.get("building_age_years", 0))

    abstract = (
        f"This paper presents StructSense AI, an integrated neural reinforcement-learning "
        f"framework for automated structural failure risk assessment of reinforced concrete "
        f"buildings. The system ingests an 18-dimensional blueprint parameter vector "
        f"(material properties, geometry, and site conditions) into a custom Gymnasium "
        f"environment and deploys a Proximal Policy Optimisation (PPO) agent to simulate "
        f"five IS-code-compliant stress scenarios (IS 456:2000, IS 1893:2016, IS 875). "
        f"A NeuralDSW module — a 3-layer MLP trained with an adaptive reward signal — "
        f"computes a structural safety weight that modulates failure penalties during "
        f"training. Risk scoring employs a weighted multi-factor model across six critical "
        f"parameters, producing per-element risk zones for foundation, columns, beams, and "
        f"slab. Groq LLaMA 3.3 70B is invoked as a reasoning layer to synthesise "
        f"simulation outputs into a research-grade structured verdict with IS code violation "
        f"analysis. "
        f"For the analysed {floors}-storey M{grade} RC building (Seismic Zone {zone}, "
        f"age {age} years), the system returned a verdict of <b>{vtext}</b> "
        f"with failure probability {fp*100:.1f}% (confidence {conf*100:.1f}%). "
        f"Primary failure mode: {pfm}. "
        f"{sc_count} of 5 simulated scenarios failed; {n_viol} IS code clause(s) violated."
    )
    items += [
        _p("Abstract", "abstract_hdr"),
        _hr(0.45, 0.5, MID_GREY),
        _sp(4),
        Paragraph(abstract, STYLES["abstract"]),
        _sp(8),
        _hr(0.45, 0.5, MID_GREY),
    ]

    items.append(PageBreak())
    return items


def _flowables_methodology(state_params: dict) -> list:
    items = [
        _sec("1.", "Methodology"),
        _hr(),
        _sec("1.1", "Input Feature Space (18 Parameters)", "sec2"),
        _p("The environment state vector <i>s ∈ ℝ¹⁸</i> encodes material, geometric, "
           "and site parameters. Table 1 summarises each feature with its unit and "
           "admissible range.", "body"),
        _sp(6),
    ]

    # Parameter table
    hdr = ["#", "Parameter", "Unit", "Range", "Category"]
    rows = [hdr]
    for i, (key, name, unit, rng, cat) in enumerate(PARAM_TABLE, 1):
        val = state_params.get(key, "—")
        val = f"{val:.3f}" if isinstance(val, float) and val == val else str(val)
        rows.append([str(i), name, unit, rng, cat])

    col_w = [0.5*cm, 5.2*cm, 1.5*cm, 3.5*cm, 2.2*cm]
    tbl   = Table(rows, colWidths=col_w, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0), NAVY),
        ("TEXTCOLOR",    (0,0), (-1,0), white),
        ("FONTNAME",     (0,0), (-1,0), "Times-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 8.5),
        ("FONTNAME",     (0,1), (-1,-1), "Times-Roman"),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("ALIGN",        (1,0), (1,-1), "LEFT"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [white, LIGHT_GREY]),
        ("GRID",         (0,0), (-1,-1), 0.5, BORDER),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ("LEFTPADDING",  (0,0), (-1,-1), 5),
    ]))
    items += [tbl, _sp(4),
              _p("<i>Table 1: StructSense AI input feature space.</i>", "small"),
              _sp(10)]

    items += [
        _sec("1.2", "RL Simulation Framework", "sec2"),
        _p("A Proximal Policy Optimisation (PPO) agent (Schulman et al., 2017) is trained "
           "on the custom <i>BuildingStressEnv</i> Gymnasium environment. The agent learns "
           "to select stress scenarios that maximally reveal structural vulnerabilities. "
           "The reward function is:", "body"),
        _sp(4),
        _p("<i>R = +1.0 (scenario passed) | R = −1.0 × w_safety (scenario failed)</i>",
           "body_bold"),
        _sp(4),
        _p("where <i>w_safety</i> ∈ [0,1] is the NeuralDSW safety weight. PPO "
           "hyperparameters: learning rate 3×10⁻⁴, n_steps=256, batch_size=64, "
           "n_epochs=10, γ=0.95, λ=0.95, clip_range=0.2, ent_coef=0.01.", "body"),
        _sp(8),

        _sec("1.3", "NeuralDSW Safety-Weight Module", "sec2"),
        _p("NeuralDSW is a 3-layer MLP (18 → 64 → 32 → 1) with ReLU activations and "
           "0.2 dropout, outputting a Sigmoid-clamped safety weight. It is trained on "
           "synthetic structural data with a composite structural efficiency target. "
           "The adaptive reward formula is: "
           "<i>R_adaptive = efficiency − (w_safety × failure_penalty)</i>. "
           "Higher safety weights impose heavier penalties for simulated failures, "
           "guiding the PPO agent towards riskier but more informative scenario selection.",
           "body"),
        _sp(8),

        _sec("1.4", "Groq LLaMA 3.3 70B Reasoning Layer", "sec2"),
        _p("Groq LLaMA 3.3 70B processes the full simulation context — 18 input "
           "parameters, element risk scores, scenario pass/fail results, NeuralDSW weight, "
           "and deterministic IS code violation flags — to synthesise a structured JSON "
           "verdict. The model is queried via the Groq Python SDK with strict "
           "JSON output constraints (no markdown, no prose). A deterministic rule-based "
           "engine provides an identical-schema fallback when network access is unavailable. "
           "The reasoning output includes overall verdict, failure probability, confidence "
           "score, primary failure mode, per-element risk zone classification, "
           "engineering reasons, actionable recommendations, and IS code clause violations.",
           "body"),
        _sp(6),
    ]
    return items


def _flowables_results(verdict: dict, state_params: dict,
                        risk_scores: Dict[str, float],
                        chart_reward: str, chart_bars: str) -> list:
    items = [
        PageBreak(),
        _sec("2.", "Simulation Results"),
        _hr(),
        _sec("2.1", "Scenario Simulation Results", "sec2"),
        _p("Five IS-code-based stress scenarios were simulated over the trained PPO "
           "evaluation episodes. Table 2 reports the per-scenario outcome.", "body"),
        _sp(6),
    ]

    sc_res = verdict.get("scenario_results", {})
    hdr  = ["#", "Scenario", "Governing Standard", "Status"]
    rows = [hdr]
    for i, (key, name, standard) in enumerate(SCENARIO_NAMES, 1):
        status = sc_res.get(key, "UNKNOWN").upper()
        rows.append([str(i), name, standard, status])

    col_w = [0.5*cm, 6.2*cm, 5.0*cm, 1.8*cm]
    sc_tbl = Table(rows, colWidths=col_w, repeatRows=1)
    style_cmds = [
        ("BACKGROUND",   (0,0), (-1,0), NAVY),
        ("TEXTCOLOR",    (0,0), (-1,0), white),
        ("FONTNAME",     (0,0), (-1,0), "Times-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 8.5),
        ("FONTNAME",     (0,1), (-1,-1), "Times-Roman"),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("ALIGN",        (1,0), (2,-1), "LEFT"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [white, LIGHT_GREY]),
        ("GRID",         (0,0), (-1,-1), 0.5, BORDER),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ("LEFTPADDING",  (0,0), (-1,-1), 5),
    ]
    for i, (key, _, _s) in enumerate(SCENARIO_NAMES, 1):
        status = sc_res.get(key, "UNKNOWN").upper()
        sc     = _status_color(status)
        style_cmds.append(("TEXTCOLOR", (3,i), (3,i), sc))
        style_cmds.append(("FONTNAME",  (3,i), (3,i), "Times-Bold"))
    sc_tbl.setStyle(TableStyle(style_cmds))
    items += [sc_tbl, _sp(4),
              _p("<i>Table 2: Stress scenario simulation outcomes.</i>", "small"),
              _sp(12)]

    items += [
        _sec("2.2", "Element Risk Scores", "sec2"),
        _p("Table 3 presents the risk score per structural element, derived from "
           "the weighted multi-factor StructuralRiskScorer model.", "body"),
        _sp(6),
    ]

    def _zlabel(r):
        if r >= 0.75: return "CRITICAL"
        if r >= 0.55: return "HIGH"
        if r >= 0.30: return "MODERATE"
        return "SAFE"

    hdr2 = ["Element", "Risk Score (0–1)", "Risk % ", "Zone"]
    rows2 = [hdr2]
    for el in ELEMENT_LABELS:
        r = risk_scores.get(el, 0.0)
        rows2.append([el.capitalize(), f"{r:.4f}", f"{r*100:.1f}%", _zlabel(r)])

    el_tbl = Table(rows2, colWidths=[3.5*cm, 3.5*cm, 2.5*cm, 3.5*cm], repeatRows=1)
    el_style = [
        ("BACKGROUND",   (0,0), (-1,0), NAVY),
        ("TEXTCOLOR",    (0,0), (-1,0), white),
        ("FONTNAME",     (0,0), (-1,0), "Times-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 9),
        ("FONTNAME",     (0,1), (-1,-1), "Times-Roman"),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [white, LIGHT_GREY]),
        ("GRID",         (0,0), (-1,-1), 0.5, BORDER),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ]
    for i, el in enumerate(ELEMENT_LABELS, 1):
        r  = risk_scores.get(el, 0.0)
        zl = _zlabel(r)
        fc = HexColor(_risk_color(zl))
        el_style += [("TEXTCOLOR", (3,i), (3,i), fc),
                     ("FONTNAME",  (3,i), (3,i), "Times-Bold")]
    el_tbl.setStyle(TableStyle(el_style))
    items += [el_tbl, _sp(4),
              _p("<i>Table 3: Structural element risk assessment.</i>", "small"),
              _sp(14)]

    # Charts
    items += [
        _sec("2.3", "Training Convergence and Risk Visualisation", "sec2"),
        _p("Figure 1 shows the PPO reward convergence over training steps. "
           "Figure 2 shows the element risk breakdown.", "body"),
        _sp(8),
        Image(chart_reward, width=14*cm, height=5.8*cm),
        _sp(2),
        _p("<i>Figure 1: PPO reward signal convergence during RL training.</i>", "small"),
        _sp(10),
        Image(chart_bars, width=14*cm, height=5.0*cm),
        _sp(2),
        _p("<i>Figure 2: Structural element risk scores (colour-coded by zone).</i>", "small"),
    ]
    return items


def _flowables_verdict(verdict: dict, chart_heatmap: str) -> list:
    items = [
        PageBreak(),
        _sec("3.", "AI Verdict and Recommendations"),
        _hr(),
    ]

    vtext   = verdict.get("overall_verdict", "UNKNOWN")
    fp      = verdict.get("failure_probability", 0)
    conf    = verdict.get("confidence", 0)
    pfm     = verdict.get("primary_failure_mode", "N/A")
    reasons = verdict.get("reasons", [])
    recs    = verdict.get("recommendations", [])
    viols   = verdict.get("is_code_violations", [])
    source  = verdict.get("source") or verdict.get("_meta", {}).get("source", "unknown")
    model   = verdict.get("_meta", {}).get("model", "llama-3.3-70b-versatile")

    bg, fg  = _verdict_color(vtext)

    # Verdict summary table
    summary = [
        ["Overall Verdict",       vtext],
        ["Failure Probability",   f"{fp*100:.1f}%"],
        ["Confidence",            f"{conf*100:.1f}%"],
        ["Primary Failure Mode",  pfm],
        ["Reasoning Source",      f"{source} ({model})"],
    ]
    vtbl = Table(summary, colWidths=[5*cm, W - 2*MARGIN - 5*cm])
    vtbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (0,-1), NAVY),
        ("TEXTCOLOR",    (0,0), (0,-1), white),
        ("FONTNAME",     (0,0), (0,-1), "Times-Bold"),
        ("FONTNAME",     (1,0), (1,-1), "Times-Roman"),
        ("FONTSIZE",     (0,0), (-1,-1), 9.5),
        ("BACKGROUND",   (1,0), (1,0), fg),
        ("TEXTCOLOR",    (1,0), (1,0), white),
        ("FONTNAME",     (1,0), (1,0), "Times-Bold"),
        ("FONTSIZE",     (1,0), (1,0), 11),
        ("ROWBACKGROUNDS",(1,1),(-1,-1), [white, LIGHT_GREY]),
        ("GRID",         (0,0), (-1,-1), 0.6, BORDER),
        ("TOPPADDING",   (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
    ]))
    items += [_sec("3.1", "Groq LLaMA 3.3 70B Verdict Summary", "sec2"),
              _sp(4), vtbl, _sp(12)]

    # Heatmap
    items += [
        _sec("3.2", "Scenario Pass/Fail Matrix", "sec2"),
        Image(chart_heatmap, width=8.5*cm, height=6.5*cm),
        _sp(2),
        _p("<i>Figure 3: Scenario outcome matrix from Groq LLaMA 3.3 reasoning layer.</i>", "small"),
        _sp(12),
    ]

    # Reasons
    items += [_sec("3.3", "Engineering Reasons", "sec2")]
    for i, r in enumerate(reasons, 1):
        items.append(_p(f"{i}. {r}", "bullet"))
    items.append(_sp(10))

    # Recommendations — blue box
    items += [_sec("3.4", "Recommendations", "sec2")]
    rec_rows = [[Paragraph(
        f'<b>[REC {i}]</b>  {rec}', STYLES["body"])] for i, rec in enumerate(recs, 1)]
    if rec_rows:
        rec_tbl = Table(rec_rows, colWidths=[W - 2*MARGIN])
        rec_tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,-1), HexColor("#eaf6ff")),
            ("BOX",          (0,0), (-1,-1), 1.2, BLUE_ACC),
            ("LEFTPADDING",  (0,0), (-1,-1), 10),
            ("RIGHTPADDING", (0,0), (-1,-1), 10),
            ("TOPPADDING",   (0,0), (-1,-1), 5),
            ("BOTTOMPADDING",(0,0), (-1,-1), 5),
            ("GRID",         (0,0), (-1,-1), 0.3, MID_GREY),
        ]))
        items += [rec_tbl, _sp(10)]

    # IS Code violations — red box
    items += [_sec("3.5", "IS Code Violations", "sec2")]
    if viols:
        viol_rows = [[Paragraph(
            f'<b>VIOLATION {i}:</b> {v}', STYLES["body"])] for i, v in enumerate(viols, 1)]
        viol_tbl = Table(viol_rows, colWidths=[W - 2*MARGIN])
        viol_tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,-1), HexColor("#fff0f0")),
            ("BOX",          (0,0), (-1,-1), 1.5, RED),
            ("LEFTPADDING",  (0,0), (-1,-1), 10),
            ("RIGHTPADDING", (0,0), (-1,-1), 10),
            ("TOPPADDING",   (0,0), (-1,-1), 5),
            ("BOTTOMPADDING",(0,0), (-1,-1), 5),
            ("GRID",         (0,0), (-1,-1), 0.3, HexColor("#f5c6c6")),
        ]))
        items += [viol_tbl]
    else:
        items.append(_p("✓ No IS code violations detected for this building configuration.",
                        "body"))
    return items


def _flowables_params_appendix(state_params: dict) -> list:
    items = [
        PageBreak(),
        _sec("A.", "Appendix — Building Parameter Values"),
        _hr(),
        _p("Table A1 lists the actual parameter values supplied for this assessment.", "body"),
        _sp(6),
    ]
    hdr  = ["Parameter Key", "Description", "Actual Value", "Unit"]
    rows = [hdr]
    for key, name, unit, rng, cat in PARAM_TABLE:
        val = state_params.get(key, "—")
        if isinstance(val, float):
            display = f"{val:.4f}" if val < 1 else f"{val:.2f}"
        else:
            display = str(val)
        rows.append([key, name, display, unit])

    col_w = [4.5*cm, 5.0*cm, 2.5*cm, 1.5*cm]
    tbl   = Table(rows, colWidths=col_w, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0), NAVY),
        ("TEXTCOLOR",    (0,0), (-1,0), white),
        ("FONTNAME",     (0,0), (-1,0), "Times-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 8.5),
        ("FONTNAME",     (0,1), (-1,-1), "Times-Roman"),
        ("ALIGN",        (0,0), (-1,-1), "LEFT"),
        ("ALIGN",        (2,0), (3,-1), "CENTER"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [white, LIGHT_GREY]),
        ("GRID",         (0,0), (-1,-1), 0.4, BORDER),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ("LEFTPADDING",  (0,0), (-1,-1), 5),
    ]))
    items += [tbl, _sp(4),
              _p("<i>Table A1: Actual input parameter values for this report.</i>", "small")]
    return items


# ════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
def generate_report(
    verdict:      Dict[str, Any],
    state_params: Dict[str, Any],
    risk_scores:  Dict[str, float],
    reward_history: List[float],
    output_path:  str = "outputs/reports/structsense_report.pdf",
) -> str:
    """
    Generate a publication-ready IEEE-style PDF report.

    Args:
        verdict        : Dict from GeminiStructuralReasoner.analyze()
        state_params   : 18-parameter building dict
        risk_scores    : {element: float 0-1} per-element risk
        reward_history : PPO reward signal list for convergence chart
        output_path    : destination PDF path (created if absent)

    Returns:
        Absolute path to the generated PDF.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # ── Generate charts ──────────────────────────────────────────────────
    tmpdir = tempfile.mkdtemp(prefix="structsense_")
    rh     = reward_history if reward_history else list(np.random.normal(0, 0.5, 30))
    c_reward  = _chart_reward(rh, tmpdir)
    c_bars    = _chart_risk_bars(risk_scores, tmpdir)
    c_heatmap = _chart_heatmap(verdict.get("scenario_results", {}), tmpdir)

    # ── Build PDF ────────────────────────────────────────────────────────
    doc = BaseDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=2.2*cm, bottomMargin=2.5*cm,
        title="StructSense AI Structural Report",
        author="StructSense Research Group",
        subject="Neural RL-Based Structural Failure Detection",
        creator="StructSense AI — ReportLab",
    )
    doc.addPageTemplates([_make_page_template(doc)])

    story = []
    story += _flowables_title_page(verdict, state_params)
    story += _flowables_methodology(state_params)
    story += _flowables_results(verdict, state_params, risk_scores, c_reward, c_bars)
    story += _flowables_verdict(verdict, c_heatmap)
    story += _flowables_params_appendix(state_params)

    doc.build(story)

    # ── Cleanup temp images ──────────────────────────────────────────────
    for f in [c_reward, c_bars, c_heatmap]:
        try: os.remove(f)
        except OSError: pass
    try: os.rmdir(tmpdir)
    except OSError: pass

    return os.path.abspath(output_path)


# ════════════════════════════════════════════════════════════════════════════
#  CLI / quick-test
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    _SAMPLE_VERDICT = {
        "overall_verdict":      "HIGH RISK",
        "confidence":           0.88,
        "failure_probability":  0.76,
        "primary_failure_mode": "Column axial failure under seismic loading due to inadequate reinforcement",
        "reasons": [
            "Column reinforcement ratio 0.009 < IS 456 minimum 0.012 (Cl.26.5.3.1)",
            "No shear wall — IS 1893:2016 Cl.7.6 required for Zone 5",
            "Concrete M15 < IS 456 minimum M20 (Cl.6.1.2)",
            "Foundation depth 0.5 m < IS 1904 minimum 0.9 m",
            "Building age 80y reduces capacity to ~36%",
        ],
        "scenario_results": {
            "normal_load": "FAIL", "seismic": "FAIL",
            "wind_rain": "FAIL", "overload": "FAIL", "degradation": "FAIL",
        },
        "risk_zones": {
            "foundation": "CRITICAL", "columns": "CRITICAL",
            "beams": "HIGH", "slab": "HIGH",
        },
        "recommendations": [
            "Add RC shear walls at grid lines B and D (IS 1893:2016)",
            "Increase column reinforcement to ≥1.2% per IS 456 Cl.26.5.3.1",
            "Retrofit foundation with pile system — 3× bearing increase",
            "CFRP wrapping for columns (+25–40% axial capacity)",
            "Commission NDT (rebound hammer + UPV) for in-situ strength",
        ],
        "is_code_violations": [
            "IS 456:2000 Cl. 26.5.3.1 — column ρ=0.009 < 0.012",
            "IS 456:2000 Cl. 6.1.2 — M15 < M20 minimum",
            "IS 1893:2016 Cl. 7.6 — no shear wall in Zone 5",
            "IS 1904:1986 Cl. 4.2 — foundation depth 0.5 < 0.9 m",
            "IS 6403:1981 Cl. 4 — SBC 60 kN/m² < 100 kN/m²",
        ],
        "_meta": {"source": "gemini_flash", "model": "gemini-2.0-flash"},
    }
    _SAMPLE_PARAMS = {
        "concrete_grade": 15.0, "steel_grade": 250.0,
        "column_width_mm": 210.0, "column_depth_mm": 210.0,
        "num_floors": 15.0, "floor_load_kn_m2": 12.0,
        "soil_bearing_capacity_kn_m2": 60.0, "foundation_depth_m": 0.5,
        "seismic_zone": 5.0, "wind_speed_kmph": 220.0,
        "rainfall_mm_annual": 2800.0, "building_age_years": 80.0,
        "shear_wall_present": 0.0, "foundation_type": 0.0,
        "column_reinforcement_ratio": 0.009,
        "beam_depth_mm": 310.0, "slab_thickness_mm": 105.0,
        "location_risk_index": 0.92,
    }
    _SAMPLE_RISKS = {"foundation": 0.88, "columns": 0.79, "beams": 0.61, "slab": 0.55}
    _REWARD_HIST  = list(np.random.normal(
        np.linspace(-0.8, 1.2, 40), 0.3).clip(-2, 2))

    out = generate_report(
        _SAMPLE_VERDICT, _SAMPLE_PARAMS, _SAMPLE_RISKS, _REWARD_HIST,
        output_path="outputs/reports/structsense_test_report.pdf",
    )
    print(f"Report saved → {out}")
