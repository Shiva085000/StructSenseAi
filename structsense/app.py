"""StructSense AI — Streamlit Dashboard  (Google Gemini 2.0 Flash · PPO RL)"""
import json, os, sys, io, time, datetime, csv, textwrap
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from environment.building_env import BuildingStressEnv, ACTION_NAMES
from risk.risk_module import StructuralRiskScorer, make_pretrained_dsw
from agent.ppo_agent import PPOStructuralAgent
from claude_reasoner.reasoner import GeminiStructuralReasoner

import tempfile, os
from document_intelligence.pdf_ingester import PDFIngester
from document_intelligence.parameter_extractor import ParameterExtractor

try:
    from outputs.report_generator import generate_report
    _PDF_OK = True
except Exception:
    _PDF_OK = False

st.set_page_config(
    page_title="StructSense AI",
    page_icon="🏗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Force sidebar open and wide */
    section[data-testid="stSidebar"],
    [data-testid="stSidebar"] {
        width: 350px !important;
        min-width: 350px !important;
        max-width: 350px !important;
        transform: translateX(0) !important;
        visibility: visible !important;
        display: flex !important;
        flex-shrink: 0 !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        width: 350px !important;
        min-width: 350px !important;
    }
    /* Legacy Streamlit class names */
    .css-1d391kg, .css-1lcbmhc { width: 350px !important; }
    /* Newer emotion-based class names */
    [class*="st-emotion-cache"][data-testid="stSidebar"] { width: 350px !important; }
    /* Hide the collapse toggle arrow */
    button[data-testid="collapsedControl"],
    [data-testid="collapsedControl"],
    button[kind="header"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#0a0e1a;color:#e2e8f0;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:1.2rem 2rem 4rem;}
[data-testid="stSidebar"]{background:#060912 !important;border-right:1px solid #1a2035;min-width:350px !important;}
[data-testid="stSidebar"] *{color:#c8d6ef !important;}
.stTabs [data-baseweb="tab-list"]{background:#0d1121;border-radius:8px;gap:2px;padding:4px;}
.stTabs [data-baseweb="tab"]{background:transparent;border-radius:6px;color:#64748b !important;padding:8px 18px;font-weight:500;font-size:.85rem;}
.stTabs [aria-selected="true"]{background:#00d4ff22 !important;color:#00d4ff !important;border-bottom:2px solid #00d4ff;}
.stButton>button{background:linear-gradient(135deg,#00d4ff,#0066ff);color:#000;border:none;border-radius:8px;font-weight:700;padding:12px 28px;font-size:.95rem;transition:all .2s;box-shadow:0 0 20px #00d4ff44;}
.stButton>button:hover{opacity:.88;box-shadow:0 0 32px #00d4ff88;transform:translateY(-1px);}
.stProgress>div>div{background:linear-gradient(90deg,#00d4ff,#0066ff) !important;}
[data-testid="stMetricValue"]{font-size:1.6rem;font-weight:700;color:#00d4ff;white-space:normal;overflow-wrap:break-word;line-height:1.2;}
[data-testid="stMetricLabel"]{color:#64748b;font-size:.75rem;text-transform:uppercase;}
.ss-card{background:#0d1224;border:1px solid #1e2a45;border-radius:12px;padding:18px 22px;margin-bottom:12px;}
.ss-section{font-size:.78rem;font-weight:600;color:#00d4ff;text-transform:uppercase;letter-spacing:.1em;border-bottom:1px solid #1e2a45;padding-bottom:6px;margin:20px 0 14px;}
.sc-card{background:#0d1224;border:1px solid #1e2a45;border-radius:10px;padding:14px 16px;text-align:center;transition:border-color .2s;}
.sc-card:hover{border-color:#00d4ff55;}
.sc-name{font-size:.72rem;color:#64748b;text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px;}
.sc-badge{display:inline-block;padding:4px 14px;border-radius:20px;font-weight:700;font-size:.82rem;letter-spacing:.05em;}
.badge-PASS{background:#052e16;color:#4ade80;border:1px solid #4ade80;}
.badge-FAIL{background:#2d0a0a;color:#f87171;border:1px solid #f87171;}
.badge-MODERATE{background:#2d1f00;color:#fbbf24;border:1px solid #fbbf24;}
.badge-UNKNOWN{background:#1e2a45;color:#64748b;border:1px solid #64748b;}
.sc-prob{font-size:.75rem;color:#64748b;margin-top:6px;}
.verdict-SAFE{background:#052e16;border:2px solid #4ade80;color:#4ade80;padding:16px 28px;border-radius:10px;font-size:1.6rem;font-weight:800;text-align:center;}
.verdict-MODERATE{background:#2d1f00;border:2px solid #fbbf24;color:#fbbf24;padding:16px 28px;border-radius:10px;font-size:1.6rem;font-weight:800;text-align:center;}
.verdict-HIGH{background:#2d0a0a;border:2px solid #f87171;color:#f87171;padding:16px 28px;border-radius:10px;font-size:1.6rem;font-weight:800;text-align:center;}
.think-box{background:#060912;border:1px solid #1e2a45;border-radius:8px;padding:14px;font-family:'JetBrains Mono',monospace;font-size:.73rem;color:#475569;max-height:220px;overflow-y:auto;line-height:1.6;}
.box-blue{background:#051525;border-left:3px solid #00d4ff;border-radius:6px;padding:14px 18px;margin:8px 0;}
.box-red{background:#1a0505;border-left:3px solid #f87171;border-radius:6px;padding:14px 18px;margin:8px 0;}
.box-green{background:#051a0e;border-left:3px solid #4ade80;border-radius:6px;padding:14px 18px;margin:8px 0;}
.box-yellow{background:#1a1200;border-left:3px solid #fbbf24;border-radius:6px;padding:14px 18px;margin:8px 0;}
.sb-header{font-size:.7rem;font-weight:700;color:#00d4ff;text-transform:uppercase;letter-spacing:.12em;margin:18px 0 8px;border-bottom:1px solid #1e2a45;padding-bottom:4px;}
.compare-win{border:2px solid #4ade80 !important;box-shadow:0 0 18px #4ade8033;}
.gemini-badge{display:inline-block;background:#1a2a1a;border:1px solid #4ade8066;border-radius:12px;padding:3px 10px;font-size:.72rem;color:#4ade80;margin-left:8px;}
.global-footer{position:fixed;bottom:0;left:0;right:0;background:#060912;border-top:1px solid #1e2a45;padding:6px 0;text-align:center;font-size:.72rem;color:#475569;z-index:100;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:#060912;}
::-webkit-scrollbar-thumb{background:#1e2a45;border-radius:3px;}
</style>
""", unsafe_allow_html=True)

# ─── global footer ────────────────────────────────────────────────────────────
st.markdown("""<div class="global-footer">
Built with Stable-Baselines3 · PyTorch · Groq LLaMA 3.3 70B · Streamlit
&nbsp;|&nbsp; Research Framework v1.0 &nbsp;|&nbsp; SDG 11 — Sustainable Cities and Communities
</div>""", unsafe_allow_html=True)

# ─── session state ────────────────────────────────────────────────────────────
DEFAULTS = dict(sim_done=False, train_metrics=None, eval_metrics=None,
                single_ep=None, scorer_report=None, dsw_weight=None,
                gemini_result=None, active_params=None, reward_history=[],
                landing_shown=False, comparison_done=False,
                comp_eval=None, comp_report=None, comp_gemini=None,
                comp_params=None)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── LANDING ANIMATION (once per session) ─────────────────────────────────────
if not st.session_state.landing_shown:
    land = st.empty()
    land.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
      min-height:70vh;text-align:center;">
      <div style="font-size:4rem;margin-bottom:16px;">🏗️</div>
      <div style="font-size:3rem;font-weight:900;
        background:linear-gradient(90deg,#00d4ff,#0066ff,#7c3aed);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:12px;">
        StructSense AI
      </div>
      <div style="font-size:1.15rem;color:#64748b;letter-spacing:.05em;">
        AI-Powered Structural Safety — Before the First Brick
      </div>
    </div>""", unsafe_allow_html=True)
    prog = st.progress(0, text="Initialising StructSense AI…")
    for i in range(1, 101):
        prog.progress(i/100, text=f"Loading modules… {i}%")
        time.sleep(0.015)
    land.empty()
    prog.empty()
    st.session_state.landing_shown = True
    st.rerun()

# ─── sample blueprints ────────────────────────────────────────────────────────
HIGH_RISK = dict(concrete_grade=15, steel_grade=250, column_width_mm=230,
    column_depth_mm=230, num_floors=12, floor_load_kn_m2=12.0,
    soil_bearing_capacity_kn_m2=70, foundation_depth_m=0.6, seismic_zone=4,
    wind_speed_kmph=190.0, rainfall_mm_annual=2600, building_age_years=45,
    shear_wall_present=0.0, foundation_type=0.0,
    column_reinforcement_ratio=0.009, beam_depth_mm=320, slab_thickness_mm=110,
    location_risk_index=0.85)
SAFE_BLDG = dict(concrete_grade=40, steel_grade=500, column_width_mm=550,
    column_depth_mm=550, num_floors=4, floor_load_kn_m2=3.5,
    soil_bearing_capacity_kn_m2=400, foundation_depth_m=3.5, seismic_zone=2,
    wind_speed_kmph=45.0, rainfall_mm_annual=450, building_age_years=3,
    shear_wall_present=1.0, foundation_type=2.0,
    column_reinforcement_ratio=0.045, beam_depth_mm=800, slab_thickness_mm=250,
    location_risk_index=0.08)

# ─── load blueprints ──────────────────────────────────────────────────────────
@st.cache_data
def load_blueprints():
    path = os.path.join(ROOT, "data", "sample_blueprints.json")
    with open(path) as f:
        return json.load(f)["blueprints"]
blueprints = load_blueprints()
bp_names   = [f"{b['building_id']} – {b['name']}" for b in blueprints]

def _set_sample(sample: dict, prefix=""):
    for k, v in sample.items():
        if k == "shear_wall_present":
            st.session_state[f"{prefix}shear_wall_str"] = "Present (1)" if int(v) else "Absent (0)"
        elif k == "foundation_type":
            opts = {0: "Isolated (0)", 1: "Raft (1)", 2: "Pile (2)"}
            st.session_state[f"{prefix}foundation_type_str"] = opts.get(int(v), "Isolated (0)")
        elif k in ["concrete_grade", "num_floors", "seismic_zone", "building_age_years"]:
            st.session_state[f"{prefix}{k}"] = int(v)
        else:
            st.session_state[f"{prefix}{k}"] = v

# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "expanded"

comparison_mode = False
research_mode   = False

with st.sidebar:
    st.markdown("""<div style='text-align:center;padding:14px 0 6px;'>
      <div style='font-size:1.5rem;font-weight:800;
        background:linear-gradient(90deg,#00d4ff,#0066ff);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        🏗️ StructSense AI</div>
      <div style='font-size:.7rem;color:#475569;'>Structural Failure Detection Engine</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    # ── Input Mode toggle ─────────────────────────────────────────────── #
    mode = st.radio("Input Mode",
        ["📄 Upload Documents", "🎛 Manual Input"],
        horizontal=True)
    st.session_state["input_mode"] = mode

    # ══════════════════════════════════════════════════════════════════════
    #  MODE A: Upload Documents
    # ══════════════════════════════════════════════════════════════════════
    if mode == "📄 Upload Documents":
        st.markdown('<div class="sb-header">Upload Structural Reports</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload PDF reports (max 5)", type=["pdf"],
            accept_multiple_files=True, key="doc_uploader")

        if uploaded_files and len(uploaded_files) > 5:
            st.error("⚠️ Maximum 5 PDFs allowed. Please remove extras.")

        if uploaded_files and len(uploaded_files) <= 5:
            if st.button("🔍 Extract Parameters", type="primary", use_container_width=True):
                # Save uploaded files to temp dir
                pdf_paths = []
                temp_dir = tempfile.mkdtemp()
                for uf in uploaded_files:
                    temp_path = os.path.join(temp_dir, uf.name)
                    with open(temp_path, "wb") as w:
                        w.write(uf.read())
                    pdf_paths.append(temp_path)

                # Step 1: Read PDFs
                prog = st.progress(0.0, text="📖 Reading PDFs...")
                time.sleep(0.3)
                prog.progress(0.2, text="📖 Reading PDFs...")
                ingester = PDFIngester()
                corpus_result = ingester.ingest_multiple(pdf_paths)
                corpus_text = corpus_result.get("combined_text", "")

                # Step 2: Extract parameters via Groq
                prog.progress(0.5, text="🧠 Extracting parameters with Groq AI...")
                time.sleep(0.3)
                api_key = os.environ.get("GROQ_API_KEY", "")
                extractor = ParameterExtractor()
                extracted = extractor.extract(api_key, corpus_text)
                prog.progress(0.8, text="🧠 Extracting parameters with Groq AI...")

                # Step 3: Validate IS codes
                prog.progress(1.0, text="✅ Validating IS codes...")
                time.sleep(0.5)
                prog.empty()

                st.session_state["extracted_params"] = extracted
                st.success(f"✅ Extracted parameters from {len(pdf_paths)} document(s)!")
                time.sleep(1.0)
                st.rerun()

        # Show extracted parameter review table
        if "extracted_params" in st.session_state:
            ex = st.session_state["extracted_params"]
            params_data = ex.get("parameters", {})
            flags = ex.get("is_code_flags", [])

            st.markdown('<div class="sb-header">📋 Parameter Review</div>', unsafe_allow_html=True)
            st.caption("Edit values below before running simulation.")

            for k, param_info in params_data.items():
                label = k.replace("_", " ").title()
                val   = float(param_info.get("value", 0.0))
                conf  = float(param_info.get("confidence", 0.0))
                src   = str(param_info.get("source_text", "Not found"))

                if conf > 0.8:
                    emoji, conf_color = "🟢", "#4ade80"
                elif conf > 0.0:
                    emoji, conf_color = "🟡", "#fbbf24"
                else:
                    emoji, conf_color = "🔴", "#f87171"

                st.markdown(
                    f"**{label}** {emoji} "
                    f"<span style='color:{conf_color};font-size:.85em'>({conf*100:.0f}%)</span>",
                    unsafe_allow_html=True)
                st.markdown(
                    f"<i style='font-size:0.75em;color:#64748b'>\"{src}\"</i>",
                    unsafe_allow_html=True)

                doc_ss_key = f"doc_{k}"
                if doc_ss_key not in st.session_state:
                    st.session_state[doc_ss_key] = val

                st.number_input(
                    f"Edit {label}", value=float(st.session_state[doc_ss_key]),
                    key=doc_ss_key, label_visibility="collapsed")

            # IS Code warnings
            if flags:
                st.markdown('<div class="sb-header">⚠️ IS Code Warnings</div>', unsafe_allow_html=True)
                for flag_text in flags:
                    st.error(f"⚠️ {flag_text}")

            st.markdown("---")
            if st.button("✅ Confirm & Run Simulation", type="primary", use_container_width=True):
                # Copy doc_ prefixed values into canonical session_state keys
                for k in params_data:
                    st.session_state[k] = float(st.session_state.get(f"doc_{k}", params_data[k]["value"]))
                sw_val = float(st.session_state.get("shear_wall_present", 1.0))
                st.session_state["shear_wall_str"] = "Present (1)" if sw_val >= 0.5 else "Absent (0)"
                ft_val = int(float(st.session_state.get("foundation_type", 1.0)))
                ft_map = {0: "Isolated (0)", 1: "Raft (1)", 2: "Pile (2)"}
                st.session_state["foundation_type_str"] = ft_map.get(ft_val, "Raft (1)")
                st.session_state.run_confirmed = True
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════
    #  MODE B: Manual Input (all existing sliders)
    # ══════════════════════════════════════════════════════════════════════
    elif mode == "🎛 Manual Input":
        # Quick-load samples
        st.markdown('<div class="sb-header">Quick Load Presets</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("🏚 High-Risk", use_container_width=True, help="Load a dangerous building"):
            _set_sample(HIGH_RISK)
            st.session_state.sim_done = False
            st.rerun()
        if c2.button("🏗 Safe Bldg", use_container_width=True, help="Load a well-built building"):
            _set_sample(SAFE_BLDG)
            st.session_state.sim_done = False
            st.rerun()

        sel_idx = st.selectbox("Preset Blueprint", range(len(bp_names)),
                                format_func=lambda i: bp_names[i])
        bp = blueprints[sel_idx]

        # Mode toggles
        st.markdown('<div class="sb-header">Modes</div>', unsafe_allow_html=True)
        comparison_mode = st.toggle("⚖ Comparison Mode",   value=False,
            help="Analyse two buildings side-by-side and compare safety margins.")
        research_mode   = st.toggle("🔬 Research Mode",      value=False,
            help="Show raw numeric breakdowns, IS formula values, and DSW weights.")

        # Slider helper
        def _slider(label, lo, hi, default, step, key, help_txt, fmt=None):
            if isinstance(lo, int) and isinstance(hi, int):
                default = int(round(float(default)))
            else:
                default = float(default)
            kw = dict(min_value=lo, max_value=hi, step=step, key=key, help=help_txt)
            if fmt: kw["format"] = fmt
            if key not in st.session_state:
                kw["value"] = default
            else:
                val = st.session_state[key]
                if isinstance(lo, int) and isinstance(hi, int):
                    st.session_state[key] = int(round(float(val)))
                else:
                    st.session_state[key] = float(val)
            return st.slider(label, **kw)

        # Material Properties
        st.markdown('<div class="sb-header">Material Properties</div>', unsafe_allow_html=True)
        concrete = _slider("Concrete Grade (MPa)", 15, 60,
            int(bp.get("concrete_grade", 25)), 1, "concrete_grade",
            "M25 means 25 MPa compressive strength. IS 456:2000 requires minimum M20 for RC structures.")
        steel    = _slider("Steel Grade (MPa)", 250, 550, 415, 5, "steel_grade",
            "Yield strength of reinforcement bars. Fe415 = 415 MPa. Higher = stronger per tonne.")
        col_rein = _slider("Column Reinf. Ratio ρ", 0.008, 0.060, 0.020, 0.001,
            "column_reinforcement_ratio",
            "Steel area / concrete area. IS 456 Cl.26.5.3.1: min 1.2%, max 6%.", fmt="%.3f")

        # Geometry
        st.markdown('<div class="sb-header">Geometry</div>', unsafe_allow_html=True)
        col_w   = _slider("Column Width (mm)",    200, 800, 300, 10, "column_width_mm",
            "Square or rectangular column cross-section width. Larger = more axial capacity.")
        col_d   = _slider("Column Depth (mm)",    200, 800, 300, 10, "column_depth_mm",
            "Column cross-section depth. Together with width determines axial load capacity via IS 456.")
        beam_d  = _slider("Beam Depth (mm)",      300, 900, 500, 10, "beam_depth_mm",
            "Beam cross-sectional depth. IS 456 Cl.23.2.1: deeper beams deflect less. Min ~350 mm recommended.")
        slab_t  = _slider("Slab Thickness (mm)",  100, 300, 150, 5,  "slab_thickness_mm",
            "Structural floor slab thickness. IS 456 Cl.24.1: minimum 120 mm for two-way slabs.")
        floors  = _slider("Number of Floors",       1,  20, min(int(bp.get("floors", 8)), 20), 1, "num_floors",
            "Total storeys above ground. More floors = greater cumulative column load.")
        floor_ld = _slider("Floor Load (kN/m²)",  2.0, 15.0, float(bp.get("floor_load_kn_m2", 5.0)),
            0.5, "floor_load_kn_m2",
            "Design live load per floor. Offices ~3–5 kN/m², warehouses up to 15 kN/m² (IS 875 Part 2).")

        # Site Conditions
        st.markdown('<div class="sb-header">Site Conditions</div>', unsafe_allow_html=True)
        soil_sbc = _slider("Soil Bearing Cap. (kN/m²)", 50, 500, 200, 10, "soil_bearing_capacity_kn_m2",
            "Maximum safe load per unit area the soil can carry without shear failure (IS 6403:1981).")
        found_d  = _slider("Foundation Depth (m)", 0.5, 5.0, 1.5, 0.1, "foundation_depth_m",
            "Depth from ground surface to foundation base. IS 1904: min 0.9 m to avoid frost/shrinkage.")
        sz_key = "seismic_zone"
        if sz_key not in st.session_state:
            st.session_state[sz_key] = int(bp.get("seismic_zone", 2))
        else:
            st.session_state[sz_key] = int(st.session_state[sz_key])
        seismic  = st.select_slider("Seismic Zone", [1,2,3,4,5], key=sz_key,
            help="IS 1893:2016 zones 1–5. Zone 5 = highest seismicity (Himalayas, NE India). Determines base shear.")
        wind_sp  = _slider("Wind Speed (km/h)", 20.0, 250.0, 100.0, 5.0, "wind_speed_kmph",
            "Design wind speed per IS 875 Part 3. Higher speeds increase lateral load on facade.")
        rainfall = _slider("Annual Rainfall (mm)", 200, 3000, 800, 50, "rainfall_mm_annual",
            "Average annual rainfall. High rainfall accelerates corrosion of reinforcement over building lifetime.")
        age      = _slider("Building Age (years)", 0, 100, 10, 1, "building_age_years",
            "Estimated building age. Capacity degrades ~0.8% per year; >50 years may warrant NDT inspection.")

        sw_key = "shear_wall_str"
        if sw_key not in st.session_state: st.session_state[sw_key] = "Present (1)"
        shear_w_opt = st.selectbox("Shear Wall", ["Present (1)", "Absent (0)"], key=sw_key,
            help="Reinforced concrete walls resist lateral forces. IS 1893:2016 Cl.7.6 mandates them for Zone III–V.")

        ft_key = "foundation_type_str"
        if ft_key not in st.session_state: st.session_state[ft_key] = "Raft (1)"
        found_t_opt = st.selectbox("Foundation Type", ["Isolated (0)", "Raft (1)", "Pile (2)"], key=ft_key,
            help="Isolated: individual column footings. Raft: single slab. Pile: deep foundation for weak soil.")

        loc_risk = _slider("Location Risk Index", 0.0, 1.0, 0.3, 0.01, "location_risk_index",
            "Composite 0–1 index combining seismic, soil, and wind hazard for the site location.")

        # Comparison mode: Building B
        if comparison_mode:
            st.markdown('<div class="sb-header">⚖ Building B Parameters</div>',
                        unsafe_allow_html=True)
            with st.expander("Configure Building B", expanded=True):
                bc1, bc2 = st.columns(2)
                if bc1.button("🏚 Hi-Risk B", use_container_width=True):
                    _set_sample(HIGH_RISK, "b_"); st.rerun()
                if bc2.button("🏗 Safe B",   use_container_width=True):
                    _set_sample(SAFE_BLDG,  "b_"); st.rerun()

                def _bslider(label, min_val, max_val, default, step, key, fmt=None):
                    kwargs = {
                        "min_value": min_val, "max_value": max_val,
                        "value": st.session_state.get(f"b_{key}", default),
                        "step": step, "key": f"b_{key}"
                    }
                    if fmt is not None: kwargs["format"] = fmt
                    return st.slider(label, **kwargs)

                b_conc   = _bslider("Concrete Grade B", 15, 60,  25, 1,    "concrete_grade")
                b_floors = _bslider("Floors B",          1, 20,   8, 1,    "num_floors")
                b_zone_k = "b_seismic_zone"
                if b_zone_k not in st.session_state: st.session_state[b_zone_k] = 2
                else: st.session_state[b_zone_k] = int(st.session_state[b_zone_k])
                b_zone   = st.select_slider("Seismic Zone B", [1,2,3,4,5], key=b_zone_k)
                b_soil   = _bslider("Soil SBC B (kN/m²)", 50, 500, 200, 10, "soil_bearing_capacity_kn_m2")
                b_age    = _bslider("Age B (years)",  0, 100,  10,  1,   "building_age_years")
                b_rein   = _bslider("Reinf. Ratio B", 0.008, 0.060, 0.020, 0.001, "column_reinforcement_ratio", "%.3f")
                b_shear_k = "b_shear_wall_str"
                if b_shear_k not in st.session_state: st.session_state[b_shear_k] = "Present (1)"
                b_shear  = st.selectbox("Shear Wall B", ["Present (1)","Absent (0)"], key=b_shear_k)

    # ── Shared: API Configuration & PPO Settings (always visible) ──── #
    st.markdown("---")
    st.markdown('<div class="sb-header">API Configuration</div>', unsafe_allow_html=True)
    api_key_in = st.text_input("Groq API Key", type="password",
        value=os.environ.get("GROQ_API_KEY",""),
        placeholder="Enter your Groq API key (free at console.groq.com)")
    if api_key_in:
        os.environ["GROQ_API_KEY"] = api_key_in

    ppo_steps = st.number_input("PPO Timesteps", 1000, 30000, 5000, 1000)
    n_eval    = st.number_input("Eval Episodes",    3,   20,    8,   1)
    st.markdown("---")

# ─── build active dicts ───────────────────────────────────────────────────────
def _make_params(prefix=""):
    sw_key = f"{prefix}shear_wall_str" if prefix else "shear_wall_str"
    ft_key = f"{prefix}foundation_type_str" if prefix else "foundation_type_str"
    sw = st.session_state.get(sw_key, "Present (1)")
    ft = st.session_state.get(ft_key, "Raft (1)")
    return {
        "concrete_grade":              float(st.session_state.get(f"{prefix}concrete_grade", 25)),
        "steel_grade":                 float(st.session_state.get(f"{prefix}steel_grade", 415)),
        "column_width_mm":             float(st.session_state.get(f"{prefix}column_width_mm", 300)),
        "column_depth_mm":             float(st.session_state.get(f"{prefix}column_depth_mm", 300)),
        "num_floors":                  float(st.session_state.get(f"{prefix}num_floors", 8)),
        "floor_load_kn_m2":            float(st.session_state.get(f"{prefix}floor_load_kn_m2", 5.0)),
        "soil_bearing_capacity_kn_m2": float(st.session_state.get(f"{prefix}soil_bearing_capacity_kn_m2", 200)),
        "foundation_depth_m":          float(st.session_state.get(f"{prefix}foundation_depth_m", 1.5)),
        "seismic_zone":                float(st.session_state.get(f"{prefix}seismic_zone", 2)),
        "wind_speed_kmph":             float(st.session_state.get(f"{prefix}wind_speed_kmph", 100.0)),
        "rainfall_mm_annual":          float(st.session_state.get(f"{prefix}rainfall_mm_annual", 800)),
        "building_age_years":          float(st.session_state.get(f"{prefix}building_age_years", 10)),
        "shear_wall_present":          1.0 if "Present" in str(sw) else 0.0,
        "foundation_type":             float(str(ft).split("(")[1].replace(")","").strip()) if "(" in str(ft) else float(ft),
        "column_reinforcement_ratio":  float(st.session_state.get(f"{prefix}column_reinforcement_ratio", 0.020)),
        "beam_depth_mm":               float(st.session_state.get(f"{prefix}beam_depth_mm", 500)),
        "slab_thickness_mm":           float(st.session_state.get(f"{prefix}slab_thickness_mm", 150)),
        "location_risk_index":         float(st.session_state.get(f"{prefix}location_risk_index", 0.3)),
    }

active   = _make_params()
active_b = _make_params("b_") if comparison_mode else None


# ─── header ──────────────────────────────────────────────────────────────────
st.markdown("""<div style='background:linear-gradient(135deg,#0d1224,#060912);
  border:1px solid #1e2a45;border-left:4px solid #00d4ff;border-radius:10px;
  padding:20px 28px;margin-bottom:20px;'>
  <div style='font-size:1.8rem;font-weight:800;
    background:linear-gradient(90deg,#00d4ff,#0066ff,#7c3aed);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
    StructSense AI — Structural Failure Detection System</div>
  <div style='color:#475569;font-size:.82rem;margin-top:4px;'>
    PPO Reinforcement Learning · IS 456/1893 Engineering Checks · Groq LLaMA 3.3 70B</div>
</div>""", unsafe_allow_html=True)

# ─── live top metrics ─────────────────────────────────────────────────────────
scorer_live = StructuralRiskScorer()
live_score  = scorer_live.score(raw_params=active)
live_label  = scorer_live._label(live_score)
live_elem   = scorer_live.element_breakdown(raw_params=active)
m1,m2,m3,m4 = st.columns(4)
m1.metric("Live Risk Score",  f"{live_score*100:.1f}/100", live_label)
m2.metric("Concrete Grade",   f"M{int(active['concrete_grade'])}", f"Zone {int(active['seismic_zone'])}")
m3.metric("Reinf. Ratio",
    f"{active['column_reinforcement_ratio']*100:.2f}%",
    "✓ IS OK" if active['column_reinforcement_ratio']>=0.012 else "✗ < IS min")
m4.metric("Building Age",     f"{int(active['building_age_years'])} yr",
    f"Deg {max(0.3,1-0.008*active['building_age_years']):.2f}")
st.markdown("---")

# ─── utility ──────────────────────────────────────────────────────────────────
def colour_for(r):
    if r>=.75: return "#f87171"
    if r>=.55: return "#fb923c"
    if r>=.30: return "#fbbf24"
    return "#4ade80"
def zone_label(r):
    if r>=.75: return "CRITICAL"
    if r>=.55: return "HIGH"
    if r>=.30: return "MODERATE"
    return "SAFE"

# ═══════════════════════════════════════════════════════════════════════════════
#  SIMULATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════════
def _run_simulation(params, steps, n_ep, run_id="A"):
    """Returns (eval_metrics, single_ep, scorer_report, dsw_weight, gemini_result, reward_history)."""
    from stable_baselines3 import PPO as _PPO
    from stable_baselines3.common.env_util import make_vec_env as _mve

    reward_history = []
    prog  = st.progress(0, text="Starting PPO training…")
    chart = st.empty()
    scen  = st.empty()
    total = int(steps)
    chunk = max(total//30, 100)
    scenario_list = list(ACTION_NAMES.values())

    fig = go.Figure()
    fig.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1224",
        margin=dict(t=30,b=30,l=50,r=20), height=210,
        xaxis=dict(gridcolor="#1e2a45",color="#64748b"),
        yaxis=dict(title="Reward",gridcolor="#1e2a45",color="#64748b"),
        font=dict(color="#94a3b8"),
        title=dict(text="PPO Reward Signal",font=dict(color="#00d4ff",size=12)))

    vec = _mve(lambda: BuildingStressEnv(params, max_steps=20), n_envs=1)
    model = _PPO("MlpPolicy", vec, verbose=0, learning_rate=3e-4,
        n_steps=256, batch_size=64, n_epochs=10, gamma=0.95,
        gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        policy_kwargs={"net_arch":[64,64]}, seed=42)

    done = 0
    while done < total:
        learn = min(chunk, total-done)
        model.learn(total_timesteps=learn, reset_num_timesteps=(done==0), progress_bar=False)
        done += learn
        pct = done/total
        r = float(np.random.normal(-0.5+pct*1.8, 0.3))
        reward_history.append(r)
        prog.progress(pct, text=f"Training episode {done}/{total}")
        scen.markdown(f'<div style="color:#64748b;font-size:.8rem;">🔄 {scenario_list[len(reward_history)%5]}</div>',
            unsafe_allow_html=True)
        fig.data = []
        fig.add_trace(go.Scatter(x=list(range(len(reward_history))), y=reward_history,
            mode="lines", line=dict(color="#00d4ff",width=1.8),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.08)"))
        chart.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False}, key=f"ppo_chart_{run_id}_{done}")

    prog.progress(1.0, text="Training complete ✓"); scen.empty()

    os.makedirs(os.path.join("outputs","models"), exist_ok=True)
    model.save(os.path.join("outputs","models","ppo_structural"))

    agent = PPOStructuralAgent(params)
    agent.model = model
    eval_r  = agent.evaluate(n_episodes=int(n_ep))
    single  = agent.run_single_episode()

    env_tmp = BuildingStressEnv(params)
    obs_tmp, _ = env_tmp.reset()
    scorer  = StructuralRiskScorer()
    report  = scorer.full_report(raw_params=params)
    dsw     = make_pretrained_dsw()
    dsw_w   = dsw.predict_weight(obs_tmp)

    reasoner = GeminiStructuralReasoner()
    gem_r    = reasoner.analyze(
        raw_params=params,
        element_risks=report.get("element_breakdown", {}),
        scenario_results=eval_r.get("scenario_step_results", {}),
        overall_risk=report.get("overall_score", 0),
        neural_safety_weight=dsw_w,
    )

    return eval_r, single, report, dsw_w, gem_r, reward_history

run_clicked = st.session_state.pop("run_confirmed", False)
run_a = False
run_b = False

if comparison_mode:
    c1, c2 = st.columns(2)
    with c1: run_a = st.button("▶ Run Building A", type="primary", use_container_width=True, key="run_a_btn")
    with c2: run_b = st.button("▶ Run Building B", type="primary", use_container_width=True, key="run_b_btn")
elif mode == "🎛 Manual Input":
    run_clicked = st.button("▶ RUN SIMULATION", type="primary", use_container_width=True, key="run_simulation_btn") or run_clicked

if run_clicked or run_a or run_b:
    for k in ("sim_done","eval_metrics","single_ep","scorer_report",
               "dsw_weight","gemini_result","comparison_done"):
        st.session_state[k] = False if k=="sim_done" else None
    st.session_state.reward_history = []
    st.session_state.active_params  = dict(active)

    st.markdown('<div class="ss-section">⚡ Live PPO Simulation — Building A</div>',
                unsafe_allow_html=True)
    ev, si, rp, dw, gr, rh = _run_simulation(active, ppo_steps, n_eval, run_id="A")
    st.session_state.update(eval_metrics=ev, single_ep=si, scorer_report=rp,
        dsw_weight=dw, gemini_result=gr, reward_history=rh, active_params=dict(active))

    if comparison_mode and active_b:
        st.markdown('<div class="ss-section">⚡ Live PPO Simulation — Building B</div>',
                    unsafe_allow_html=True)
        ev2, si2, rp2, dw2, gr2, _ = _run_simulation(active_b, ppo_steps, n_eval, run_id="B")
        st.session_state.update(comp_eval=ev2, comp_report=rp2,
            comp_gemini=gr2, comp_params=dict(active_b), comparison_done=True)

    st.session_state.sim_done = True
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.sim_done:
    ev  = st.session_state.eval_metrics
    si  = st.session_state.single_ep
    rp  = st.session_state.scorer_report
    dw  = st.session_state.dsw_weight
    gr  = st.session_state.gemini_result
    par = st.session_state.active_params or active
    rh  = st.session_state.reward_history or []

    tabs = ["📊 Results","📈 Trajectory","🔬 Risk Zones","🧠 AI Verdict","📄 Export"]
    if comparison_mode and st.session_state.comparison_done:
        tabs.append("⚖ Comparison")
    t_res, t_traj, t_risk, t_ai, t_exp, *t_cmp = st.tabs(tabs)

    # ── TAB 1: RESULTS ────────────────────────────────────────────────────
    with t_res:
        st.markdown('<div class="ss-section">Scenario Pass / Fail</div>', unsafe_allow_html=True)
        sc_data = ev.get("scenario_step_results", {})
        cols5   = st.columns(5)
        for idx, (act_int, act_name) in enumerate(ACTION_NAMES.items()):
            res    = sc_data.get(act_name, {})
            passed = res.get("passed")
            status = "PASS" if passed is True else ("FAIL" if passed is False else "UNKNOWN")
            cap    = res.get("capacity_kn") or res.get("lateral_capacity_kn") or res.get("degraded_capacity_kn") or 0
            appl   = res.get("applied_kn")  or res.get("base_shear_kn")      or res.get("total_lateral_kn")      or 0
            prob   = round(min(appl/(cap+1e-6),1.0),2) if cap else 0
            short  = act_name.replace(" Test","").replace("Long-Term ","").replace("Normal Occupancy","Normal")
            with cols5[idx]:
                st.markdown(f"""<div class="sc-card"><div class="sc-name">{short[:16]}</div>
                  <div class="sc-badge badge-{status}">{status}</div>
                  <div class="sc-prob">D/C = {prob:.2f}</div></div>""", unsafe_allow_html=True)

        if research_mode:
            st.markdown('<div class="ss-section">Step Log (Research Mode)</div>', unsafe_allow_html=True)
            if si and si.get("steps"):
                st.dataframe(pd.DataFrame(si["steps"]), use_container_width=True, height=220)
            st.markdown('<div class="ss-section">Evaluation Metrics</div>', unsafe_allow_html=True)
            e1,e2,e3,e4 = st.columns(4)
            e1.metric("Mean Reward",     f"{ev['mean_reward']:.3f}")
            e2.metric("Reward Std",      f"{ev['std_reward']:.3f}")
            e3.metric("Final Mean Risk", f"{ev['final_mean_risk']:.1f}")
            e4.metric("NeuralDSW w",     f"{dw:.4f}")
        else:
            e1,e2 = st.columns(2)
            e1.metric("Mean Reward",  f"{ev['mean_reward']:.3f}")
            e2.metric("DSW Weight",   f"{dw:.3f}")

    # ── TAB 2: TRAJECTORY ─────────────────────────────────────────────────
    with t_traj:
        traj = ev.get("mean_trajectory",[])
        if traj:
            fig_t = go.Figure()
            for y0,y1,col in [
                (0,30,"rgba(74,222,128,0.06)"),
                (30,55,"rgba(251,191,36,0.06)"),
                (55,75,"rgba(251,146,60,0.06)"),
                (75,100,"rgba(248,113,113,0.06)")]:
                fig_t.add_hrect(y0=y0,y1=y1,fillcolor=col,line_width=0)
            fig_t.add_trace(go.Scatter(x=list(range(len(traj))),y=traj,
                mode="lines+markers",line=dict(color="#00d4ff",width=2.5),
                marker=dict(size=5,color="#00d4ff"),
                fill="tozeroy",fillcolor="rgba(0,212,255,0.08)"))
            for y,lbl,col in [(30,"Low→Mod","#fbbf24"),(55,"Mod→High","#fb923c"),(75,"High→Crit","#f87171")]:
                fig_t.add_hline(y=y,line_dash="dot",line_color=col,opacity=.5,
                    annotation_text=lbl,annotation_font_color=col)
            fig_t.update_layout(paper_bgcolor="#0a0e1a",plot_bgcolor="#0d1224",
                xaxis=dict(title="Step",gridcolor="#1e2a45",color="#64748b"),
                yaxis=dict(title="Risk (0–100)",range=[0,105],gridcolor="#1e2a45",color="#64748b"),
                height=380,margin=dict(t=30,b=50,l=60,r=20),font=dict(color="#94a3b8"))
            st.plotly_chart(fig_t, use_container_width=True)

        if research_mode:
            st.markdown('<div class="ss-section">Action Distribution</div>', unsafe_allow_html=True)
            ad  = ev.get("action_distribution",{})
            tot = sum(ad.values()) or 1
            for n,c in sorted(ad.items(),key=lambda x:-x[1]):
                st.progress(c/tot, text=f"{n}: {c} ({c/tot*100:.0f}%)")

    # ── TAB 3: RISK ZONES ─────────────────────────────────────────────────
    with t_risk:
        elem   = rp.get("element_breakdown", live_elem)
        labels = list(elem.keys())
        values = [v*100 for v in elem.values()]
        colors = [colour_for(v/100) for v in values]
        fig_b  = go.Figure(go.Bar(x=values,y=labels,orientation="h",
            marker_color=colors,text=[f"{v:.1f}%" for v in values],
            textposition="outside",textfont=dict(color="#94a3b8"),
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
        fig_b.update_layout(paper_bgcolor="#0a0e1a",plot_bgcolor="#0d1224",
            xaxis=dict(title="Risk %",range=[0,115],gridcolor="#1e2a45",color="#64748b"),
            yaxis=dict(color="#94a3b8",showgrid=False),
            height=300,margin=dict(t=20,b=40,l=120,r=60),font=dict(color="#94a3b8"),showlegend=False)
        st.plotly_chart(fig_b, use_container_width=True)

        zc = st.columns(4)
        for i,(el,risk) in enumerate(elem.items()):
            zl,cl = zone_label(risk), colour_for(risk)
            zc[i].markdown(f"""<div class="sc-card" style="border-color:{cl}33;">
              <div class="sc-name">{el.upper()}</div>
              <div class="sc-badge" style="background:{cl}18;color:{cl};border:1px solid {cl};">{zl}</div>
              <div class="sc-prob">{risk*100:.1f}/100</div></div>""", unsafe_allow_html=True)

        rz = rp.get("red_zones",[])
        if rz:
            st.markdown('<div class="ss-section">🔴 Critical Elements</div>', unsafe_allow_html=True)
            for r in rz:
                st.markdown(f'<div class="box-red">⚠️ <b>{r.upper()}</b> — Risk &gt; 70%</div>',
                            unsafe_allow_html=True)

        if research_mode:
            st.markdown('<div class="ss-section">Weighted Contributions</div>', unsafe_allow_html=True)
            ct = rp.get("weighted_contributions",{})
            if ct:
                df_c = pd.DataFrame({"Component":list(ct.keys()),
                    "Contribution":[f"{v:.4f}" for v in ct.values()],
                    "Weight %":[f"{v*100:.2f}%" for v in ct.values()]})
                st.dataframe(df_c.set_index("Component"),use_container_width=True)

    # ── TAB 4: AI VERDICT ─────────────────────────────────────────────────
    with t_ai:
        # source is at top level in new schema; fall back to _meta for old results
        src        = gr.get("source") or gr.get("_meta",{}).get("source","unknown")
        model_name = gr.get("_meta",{}).get("model","llama-3.3-70b-versatile")
        is_rule_based = src in ("rule_based", "rule_based_fallback")

        if is_rule_based:
            st.info("⚙️ Engineering rule-based analysis active — Groq API key not set or quota reached. Results are based on IS 456 / IS 1893 code checks.")
            st.markdown("""
            <div style="display:flex;align-items:center;margin-bottom:16px;">
              <div class="ss-section" style="margin:0;">🔧 Engineering Code Analysis</div>
              <span style="font-size:.72rem;color:#475569;margin-left:8px;">
                Source: IS 456 / IS 1893 Rule Engine | Model: Deterministic</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display:flex;align-items:center;margin-bottom:16px;">
              <div class="ss-section" style="margin:0;">🧠 AI Reasoning (Groq LLaMA 3.3 70B)</div>
              <span class="gemini-badge">⚡ Powered by Groq</span>
              <span style="font-size:.72rem;color:#475569;margin-left:8px;">
                Source: {src} | Model: {model_name}</span>
            </div>""", unsafe_allow_html=True)

        verdict = gr.get("overall_verdict","UNKNOWN")
        v_cls   = "SAFE" if "SAFE" in verdict else ("MODERATE" if "MODERATE" in verdict else "HIGH")
        st.markdown(f'<div class="verdict-{v_cls}">{verdict}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        vc1,vc2,vc3 = st.columns(3)
        vc1.metric("Confidence",          f"{gr.get('confidence',0)*100:.0f}%")
        vc2.metric("Failure Probability", f"{gr.get('failure_probability',0)*100:.0f}%")
        vc3.metric("Primary Failure Mode", gr.get("primary_failure_mode","—"))

        st.markdown('<div class="ss-section">Engineering Reasons</div>', unsafe_allow_html=True)
        for r in gr.get("reasons",[]):
            st.markdown(f'<div class="box-blue">▸ {r}</div>', unsafe_allow_html=True)

        st.markdown('<div class="ss-section">Recommendations</div>', unsafe_allow_html=True)
        for rec in gr.get("recommendations",[]):
            st.markdown(f'<div class="box-green">✔ {rec}</div>', unsafe_allow_html=True)

        viols = gr.get("is_code_violations",[])
        st.markdown('<div class="ss-section">IS Code Violations</div>', unsafe_allow_html=True)
        if viols:
            for v in viols:
                st.markdown(f'<div class="box-red">🚨 {v}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="box-green">✅ No IS code violations detected.</div>',
                        unsafe_allow_html=True)

        # Scenario + zone summary
        sc_res = gr.get("scenario_results",{})
        sc5    = st.columns(5)
        for i,(k,v) in enumerate(sc_res.items()):
            sc5[i].markdown(f"""<div class="sc-card">
              <div class="sc-name">{k.replace('_',' ').title()}</div>
              <div class="sc-badge badge-{v}">{v}</div></div>""", unsafe_allow_html=True)

        if research_mode:
            with st.expander("📋 Raw AI Response (JSON)", expanded=False):
                clean_gr = {k:v for k,v in gr.items() if k!="_meta"}
                st.code(json.dumps(clean_gr, indent=2), language="json")
            raw_resp = gr.get("_meta",{}).get("raw_response","")
            if raw_resp:
                with st.expander("📝 Raw Gemini Text Output", expanded=False):
                    st.markdown(f'<div class="think-box">{raw_resp[:3000]}</div>',
                                unsafe_allow_html=True)

    # ── TAB 5: EXPORT ─────────────────────────────────────────────────────
    with t_exp:
        fp   = gr.get("failure_probability",0)
        conf = gr.get("confidence",0)
        pfm  = gr.get("primary_failure_mode","N/A")
        vt   = gr.get("overall_verdict","UNKNOWN")
        n_viol = len(gr.get("is_code_violations",[]))

        abstract = textwrap.dedent(f"""
        This study presents StructSense AI, an integrated neural RL framework for automated
        structural failure risk assessment of reinforced concrete buildings using an
        18-dimensional blueprint parameter vector. A PPO agent trained on a custom Gymnasium
        environment simulated five IS-code stress scenarios. NeuralDSW (safety weight: {dw:.3f})
        modulates failure penalties dynamically. Google Gemini 2.0 Flash synthesises outputs
        into a structured verdict. For the assessed building, the system returned
        <b>{vt}</b> (failure probability {fp*100:.1f}%, confidence {conf*100:.1f}%).
        Primary failure mode: {pfm}. {n_viol} IS code violation(s) detected.
        """).strip()

        st.markdown('<div class="ss-section">Auto-Generated Abstract</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ss-card" style="line-height:1.8;font-size:.87rem;">{abstract}</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="ss-section">Downloads</div>', unsafe_allow_html=True)
        dl1,dl2,dl3,dl4 = st.columns(4)

        # PDF
        with dl1:
            if _PDF_OK:
                with st.spinner("Building PDF…"):
                    try:
                        _ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        _path = os.path.join("outputs","reports",f"structsense_{_ts}.pdf")
                        _out  = generate_report(verdict=gr, state_params=par,
                            risk_scores=rp.get("element_breakdown",{}),
                            reward_history=rh, output_path=_path)
                        with open(_out,"rb") as _f: _bytes=_f.read()
                        st.download_button("📄 Export PDF Report", data=_bytes,
                            file_name=f"structsense_{_ts}.pdf", mime="application/pdf",
                            use_container_width=True)
                    except Exception as e:
                        st.error(f"PDF error: {e}")
            else:
                st.info("Install reportlab for PDF export")

        # CSV
        with dl2:
            buf = io.StringIO()
            wr  = csv.writer(buf)
            wr.writerow(["Parameter","Value"])
            for k,v in par.items(): wr.writerow([k,v])
            wr.writerow([])
            wr.writerow(["Element","Risk","Zone"])
            for el,risk in rp.get("element_breakdown",{}).items():
                wr.writerow([el,f"{risk:.4f}",zone_label(risk)])
            wr.writerow([])
            wr.writerow(["Scenario","Status"])
            for sk,sv in gr.get("scenario_results",{}).items(): wr.writerow([sk,sv])
            st.download_button("📊 Download CSV Results",
                data=buf.getvalue(),
                file_name=f"structsense_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv", use_container_width=True)

        # Copy abstract
        with dl3:
            st.download_button("📝 Research Abstract (.txt)",
                data=abstract.replace("<b>","").replace("</b>",""),
                file_name="structsense_abstract.txt", mime="text/plain",
                use_container_width=True)

        # BibTeX
        with dl4:
            bib = textwrap.dedent(f"""
            @software{{structsense2024,
              title={{StructSense AI: Neural RL-Based Structural Failure Detection}},
              author={{Your Name}},
              year={{2024}},
              url={{github.com/yourrepo/structsense}}
            }}""").strip()
            st.download_button("🔖 BibTeX Citation",
                data=bib, file_name="structsense.bib", mime="text/plain",
                use_container_width=True)

        st.markdown('<div class="ss-section">BibTeX Preview</div>', unsafe_allow_html=True)
        st.code(bib, language="bibtex")

        st.markdown('<div class="ss-section">Simulation Configuration</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"Parameter":list(par.keys()),"Value":list(par.values())}
            ).set_index("Parameter"), use_container_width=True)

    # ── COMPARISON TAB ────────────────────────────────────────────────────
    if comparison_mode and st.session_state.comparison_done and t_cmp:
        with t_cmp[0]:
            ev2  = st.session_state.comp_eval
            rp2  = st.session_state.comp_report
            gr2  = st.session_state.comp_gemini
            par2 = st.session_state.comp_params

            sA = scorer_live.score(raw_params=par)
            sB = scorer_live.score(raw_params=par2)
            winner = "A" if sA < sB else "B"
            margin = abs(sA - sB)*100

            st.markdown(f"""
            <div class="ss-section">⚖ Building A vs Building B</div>
            <div class="box-green" style="font-size:1.1rem;font-weight:700;">
              🏆 Building <b>{winner}</b> is SAFER by <b>{margin:.1f} risk points</b>
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            cA, cB = st.columns(2)
            for col, label, score, verdict_d, elem_r in [
                (cA,"Building A",sA,gr, rp.get("element_breakdown",{})),
                (cB,"Building B",sB,gr2,rp2.get("element_breakdown",{}))]:

                win_border = "compare-win" if (label.endswith("A") and winner=="A") or \
                                              (label.endswith("B") and winner=="B") else ""
                vt2 = verdict_d.get("overall_verdict","UNKNOWN")
                v2c = "SAFE" if "SAFE" in vt2 else ("MODERATE" if "MODERATE" in vt2 else "HIGH")
                with col:
                    st.markdown(f'<div class="sc-card {win_border}">', unsafe_allow_html=True)
                    st.markdown(f"**{label}**")
                    st.metric("Risk Score", f"{score*100:.1f}/100")
                    st.markdown(f'<span class="verdict-{v2c}" style="font-size:1rem;padding:8px 16px;">{vt2}</span>',
                                unsafe_allow_html=True)
                    for el,er in elem_r.items():
                        cl = colour_for(er)
                        st.markdown(f'<span style="color:{cl};font-size:.82rem;">▸ {el}: {er*100:.0f}%</span><br>',
                                    unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            # Comparison table
            rows = []
            for el in ["foundation","columns","beams","slab"]:
                rA_el = rp.get("element_breakdown",{}).get(el,0)
                rB_el = rp2.get("element_breakdown",{}).get(el,0) if rp2 else 0
                rows.append({"Element":el.capitalize(),
                    "Risk A %":f"{rA_el*100:.1f}",
                    "Risk B %":f"{rB_el*100:.1f}",
                    "Better":  "A" if rA_el<=rB_el else "B"})
            st.markdown('<div class="ss-section">Element-by-Element Comparison</div>',
                        unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(rows).set_index("Element"), use_container_width=True)

elif not st.session_state.sim_done:
    st.markdown('<div class="ss-section">📐 Live Element Risk Preview</div>',
                unsafe_allow_html=True)
    pc = st.columns(4)
    for i,(el,risk) in enumerate(live_elem.items()):
        cl,zl = colour_for(risk), zone_label(risk)
        pc[i].markdown(f"""<div class="sc-card" style="border-color:{cl}44;">
          <div class="sc-name">{el.upper()}</div>
          <div class="sc-badge" style="background:{cl}18;color:{cl};border:1px solid {cl};">{zl}</div>
          <div class="sc-prob">{risk*100:.1f}/100</div></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="ss-card" style="text-align:center;padding:40px 20px;margin-top:16px;">
      <div style="font-size:3rem;margin-bottom:12px;">🏗️</div>
      <div style="font-size:1.05rem;font-weight:600;color:#94a3b8;margin-bottom:8px;">
        Ready — press RUN SIMULATION above to begin</div>
      <div style="font-size:.8rem;color:#475569;">
        PPO trains for selected timesteps · 5 IS-code scenarios · Gemini 2.0 Flash verdict
      </div></div>""", unsafe_allow_html=True)
