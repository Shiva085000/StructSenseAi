"""Quick fallback test — uses a fake API key so API is never called."""
import sys, os
sys.path.insert(0, ".")
os.environ.pop("GEMINI_API_KEY", None)  # clear
os.environ["GEMINI_API_KEY"] = "fake-key-for-fallback-test"  # invalid = no network call attempted

# Override _GENAI_OK to False to force fallback without network
import claude_reasoner.reasoner as _mod
_orig = _mod._GENAI_OK
_mod._GENAI_OK = False

from claude_reasoner.reasoner import GeminiStructuralReasoner, ClaudeStructuralReasoner

r = GeminiStructuralReasoner()
print(f"is_available (fake key, GENAI disabled): {r.is_available}")

params = {
    "concrete_grade": 15, "steel_grade": 250,
    "column_width_mm": 210, "column_depth_mm": 210,
    "num_floors": 15, "floor_load_kn_m2": 12,
    "soil_bearing_capacity_kn_m2": 60, "foundation_depth_m": 0.5,
    "seismic_zone": 5, "wind_speed_kmph": 220,
    "rainfall_mm_annual": 2800, "building_age_years": 80,
    "shear_wall_present": 0, "foundation_type": 0,
    "column_reinforcement_ratio": 0.009,
    "beam_depth_mm": 310, "slab_thickness_mm": 105,
    "location_risk_index": 0.92,
}
risks = {"foundation": 0.88, "columns": 0.79, "beams": 0.61, "slab": 0.55}
sim   = {
    "Normal Occupancy Load Test": {"passed": False},
    "Seismic Stress Test":        {"passed": False},
    "Wind + Rain Combined Test":  {"passed": False},
    "Overload Test (150 %)":      {"passed": False},
    "Long-Term Degradation Test": {"passed": False},
}

verdict = r.analyze(raw_params=params, element_risks=risks,
                    scenario_results=sim, overall_risk=0.88)
print(f"verdict          : {verdict['overall_verdict']}")
print(f"source           : {verdict['_meta']['source']}")
print(f"failure_prob     : {verdict['failure_probability']}")
print(f"reasons          : {len(verdict['reasons'])}")
print(f"IS violations    : {len(verdict['is_code_violations'])}")
print(f"scenario_results : {verdict['scenario_results']}")
print(f"risk_zones       : {verdict['risk_zones']}")

assert verdict["overall_verdict"]         == "HIGH RISK"
assert verdict["_meta"]["source"]         == "rule_based_fallback"
assert len(verdict["is_code_violations"]) >= 4
assert set(verdict["scenario_results"])   == {"normal_load","seismic","wind_rain","overload","degradation"}
assert set(verdict["risk_zones"])         == {"foundation","columns","beams","slab"}

# Alias check
assert ClaudeStructuralReasoner is GeminiStructuralReasoner

_mod._GENAI_OK = _orig  # restore
print("\nALL CHECKS PASSED")
