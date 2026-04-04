"""
claude_reasoner/reasoner.py – StructSense AI · Groq LLaMA 3.3 70B Reasoner
============================================================================
Provides:
  - get_structural_verdict()   : standalone function
  - GeminiStructuralReasoner   : class wrapper keeping the same .analyze() interface
    used by app.py, with a deterministic rule-based fallback when no API key is set.

Output schema:
{
  "overall_verdict":     "HIGH RISK | MODERATE RISK | SAFE",
  "confidence":          float 0-1,
  "failure_probability": float 0-1,
  "primary_failure_mode": str,
  "reasons":             [str, ...],
  "scenario_results":    {normal_load, seismic, wind_rain, overload, degradation},
  "risk_zones":          {foundation, columns, beams, slab},
  "recommendations":     [str, ...],
  "is_code_violations":  [str, ...],
  "source":              "groq" | "rule_based"
}
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

try:
    from groq import Groq
    _GROQ_OK = True
except ImportError:
    _GROQ_OK = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
#  IS code violation table
# ═══════════════════════════════════════════════════════════════════════════════

_IS_CHECKS: List[Dict[str, Any]] = [
    {"param": "column_reinforcement_ratio", "limit": 0.012, "bad_when": "below",
     "clause": "IS 456:2000 Cl. 26.5.3.1",
     "text": "Minimum longitudinal reinforcement for columns (ρ_min = 1.2%)"},
    {"param": "column_reinforcement_ratio", "limit": 0.06,  "bad_when": "above",
     "clause": "IS 456:2000 Cl. 26.5.3.1",
     "text": "Maximum longitudinal reinforcement for columns (ρ_max = 6%)"},
    {"param": "concrete_grade",             "limit": 20.0,  "bad_when": "below",
     "clause": "IS 456:2000 Cl. 6.1.2",
     "text": "Minimum M20 concrete for reinforced concrete structures"},
    {"param": "slab_thickness_mm",          "limit": 120.0, "bad_when": "below",
     "clause": "IS 456:2000 Cl. 24.1",
     "text": "Minimum slab thickness for two-way slabs"},
    {"param": "foundation_depth_m",         "limit": 0.9,   "bad_when": "below",
     "clause": "IS 1904:1986 Cl. 4.2",
     "text": "Minimum foundation depth to avoid frost/shrink damage"},
    {"param": "soil_bearing_capacity_kn_m2","limit": 100.0, "bad_when": "below",
     "clause": "IS 6403:1981 Cl. 4",
     "text": "Allowable bearing capacity below recommended minimum for multi-storey"},
    {"param": "beam_depth_mm",              "limit": 350.0, "bad_when": "below",
     "clause": "IS 456:2000 Cl. 23.2.1",
     "text": "Deflection-controlled minimum beam depth"},
]


def _detect_violations(raw_params: Dict[str, Any]) -> List[str]:
    violations = []
    for chk in _IS_CHECKS:
        v     = float(raw_params.get(chk["param"], 0))
        limit = chk["limit"]
        hit   = (chk["bad_when"] == "below" and v < limit) or \
                (chk["bad_when"] == "above"  and v > limit)
        if hit:
            violations.append(
                f"{chk['clause']} — {chk['text']} (actual: {v}, limit: {limit})"
            )
    return violations


def _zone_label(risk: float) -> str:
    if risk >= 0.75: return "CRITICAL"
    if risk >= 0.55: return "HIGH"
    if risk >= 0.30: return "MODERATE"
    return "SAFE"


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule-based fallback
# ═══════════════════════════════════════════════════════════════════════════════

def rule_based_fallback(state_params: dict, risk_scores: dict,
                        simulation_results: dict) -> dict:
    """Deterministic IS-code verdict — no API key required."""

    risk_avg = sum(risk_scores.values()) / len(risk_scores) if risk_scores else 0.5
    failures = [k for k, v in simulation_results.items() if v == "FAIL"]

    if risk_avg > 0.7:
        verdict, fp, conf = "HIGH RISK", round(risk_avg, 2), 0.85
    elif risk_avg > 0.4:
        verdict, fp, conf = "MODERATE RISK", round(risk_avg, 2), 0.78
    else:
        verdict, fp, conf = "SAFE", round(risk_avg, 2), 0.90

    # Reasons from actual params
    reasons: List[str] = []
    conc  = state_params.get("concrete_grade", 25)
    rho   = state_params.get("column_reinforcement_ratio", 0.02)
    sw    = state_params.get("shear_wall_present", 1)
    zone  = state_params.get("seismic_zone", 2)
    age   = state_params.get("building_age_years", 0)
    soil  = state_params.get("soil_bearing_capacity_kn_m2", 200)

    if conc < 20:
        reasons.append(f"Concrete grade M{int(conc)} is below IS 456 minimum M20")
    if rho < 0.012:
        reasons.append("Column reinforcement ratio below IS 456 Clause 26.5.3 minimum of 0.012")
    if sw == 0 and zone >= 3:
        reasons.append(f"No shear walls — mandatory for Zone {int(zone)} per IS 1893")
    if age > 30:
        reasons.append(f"Building age {int(age)} years indicates material degradation")
    if soil < 100:
        reasons.append("Soil bearing capacity insufficient for structural load")
    if not reasons:
        reasons.append("Parameters are within acceptable IS code limits")

    # IS code violations (full detail from checker)
    violations = _detect_violations(state_params)
    if not violations:
        violations = ["No IS code violations detected"]

    # Recommendations
    recs: List[str] = []
    if verdict == "HIGH RISK":
        recs = ["Immediate structural audit by licensed engineer required",
                "Retrofit foundation with pile support system",
                "Install shear walls at primary grid lines before occupancy"]
    elif verdict == "MODERATE RISK":
        recs = ["Schedule structural inspection every 2 years",
                "Monitor foundation settlement and crack propagation",
                "Review load distribution on critical columns"]
    else:
        recs = ["Maintain regular maintenance schedule",
                "Re-evaluate after 10 years or major seismic event"]

    return {
        "overall_verdict":      verdict,
        "confidence":           conf,
        "failure_probability":  fp,
        "primary_failure_mode": "Seismic and overload failure" if failures else "No critical failure mode",
        "reasons":              reasons,
        "scenario_results":     simulation_results,
        "risk_zones":           {k: _zone_label(v) for k, v in risk_scores.items()},
        "recommendations":      recs,
        "is_code_violations":   violations,
        "source":               "rule_based",
        "_meta": {
            "source": "rule_based",
            "model":  "deterministic",
            "raw_response": "",
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Standalone function
# ═══════════════════════════════════════════════════════════════════════════════

def get_structural_verdict(api_key: str, state_params: dict,
                           risk_scores: dict, simulation_results: dict) -> dict:
    """
    Groq LLaMA 3.3 70B structural verdict.
    Falls back silently to rule_based_fallback on any error or missing key.
    """
    if not api_key or not _GROQ_OK:
        return rule_based_fallback(state_params, risk_scores, simulation_results)

    try:
        client = Groq(api_key=api_key)

        prompt = f"""
You are a structural safety expert AI. Analyze this building and return
ONLY valid JSON with no markdown, no explanation, no extra text.

BUILDING PARAMETERS:
{json.dumps(state_params, indent=2)}

RISK SCORES PER ELEMENT:
{json.dumps(risk_scores, indent=2)}

SIMULATION RESULTS:
{json.dumps(simulation_results, indent=2)}

Return EXACTLY this JSON structure:
{{
  "overall_verdict": "HIGH RISK or MODERATE RISK or SAFE",
  "confidence": 0.0,
  "failure_probability": 0.0,
  "primary_failure_mode": "one line description",
  "reasons": ["reason 1", "reason 2", "reason 3"],
  "scenario_results": {{
    "normal_load": "PASS or FAIL",
    "seismic": "PASS or FAIL",
    "wind_rain": "PASS or FAIL",
    "overload": "PASS or FAIL",
    "degradation": "PASS or FAIL"
  }},
  "risk_zones": {{
    "foundation": "CRITICAL or HIGH or MODERATE or SAFE",
    "columns": "CRITICAL or HIGH or MODERATE or SAFE",
    "beams": "CRITICAL or HIGH or MODERATE or SAFE",
    "slab": "CRITICAL or HIGH or MODERATE or SAFE"
  }},
  "recommendations": ["rec 1", "rec 2", "rec 3"],
  "is_code_violations": ["violation 1", "violation 2"],
  "source": "groq"
}}
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a structural engineering AI. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=2048
        )

        raw   = response.choices[0].message.content.strip()
        clean = re.sub(r"```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        clean = re.sub(r"```\s*$", "", clean).strip()
        result = json.loads(clean)
        # Attach _meta for compatibility with app.py
        result.setdefault("_meta", {
            "source":       "groq",
            "model":        "llama-3.3-70b-versatile",
            "raw_response": raw,
        })
        result.setdefault("source", "groq")
        return result

    except Exception:
        return rule_based_fallback(state_params, risk_scores, simulation_results)


# ═══════════════════════════════════════════════════════════════════════════════
#  Class wrapper — keeps app.py's .analyze() interface intact
# ═══════════════════════════════════════════════════════════════════════════════

class GeminiStructuralReasoner:
    """
    Drop-in replacement — now backed by Groq LLaMA 3.3 70B.
    Falls back to deterministic IS-code rule engine when key is absent.
    """

    def __init__(self) -> None:
        self._api_key = os.environ.get("GROQ_API_KEY", "").strip()

    @property
    def is_available(self) -> bool:
        return bool(_GROQ_OK and self._api_key)

    def analyze(
        self,
        raw_params: Dict[str, Any],
        element_risks: Dict[str, float],
        scenario_results: Dict[str, Any],
        overall_risk: float = 0.0,
        neural_safety_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Flatten scenario_results then call Groq (or fallback)."""

        # Flatten detailed scenario dicts to simple pass/fail strings
        flat: Dict[str, str] = {}
        _KEY_MAP = {
            "Normal Occupancy Load Test": "normal_load",
            "Seismic Stress Test":        "seismic",
            "Wind + Rain Combined Test":  "wind_rain",
            "Overload Test (150 %)":      "overload",
            "Long-Term Degradation Test": "degradation",
        }
        for name, res in (scenario_results or {}).items():
            key = _KEY_MAP.get(name, name)
            if isinstance(res, dict):
                passed = res.get("passed")
                cap  = res.get("capacity_kn") or res.get("lateral_capacity_kn") or res.get("degraded_capacity_kn")
                appl = res.get("applied_kn")  or res.get("base_shear_kn")      or res.get("total_lateral_kn")
                if passed is True:
                    flat[key] = "PASS"
                elif cap and appl and abs(cap - appl) / (cap + 1e-6) < 0.15:
                    flat[key] = "MODERATE"
                else:
                    flat[key] = "FAIL"
            else:
                flat[key] = str(res)
        for k in ("normal_load", "seismic", "wind_rain", "overload", "degradation"):
            flat.setdefault(k, "PASS")

        return get_structural_verdict(self._api_key, raw_params, element_risks, flat)


# Backward-compat alias
ClaudeStructuralReasoner = GeminiStructuralReasoner
