"""
ParameterExtractor – Uses Groq LLaMA 3.3 to extract 18 structural parameters
from combined PDF text corpus for StructSense AI.
"""
import json
import logging
from typing import Dict, Any

from groq import Groq

logger = logging.getLogger(__name__)

# The 18 parameters StructSense needs, with human labels and IS code limits
PARAM_SPEC = {
    "concrete_grade":              {"label": "Concrete Grade (MPa)",       "min": 15,    "max": 60,   "default": 25.0,   "unit": "MPa"},
    "steel_grade":                 {"label": "Steel Grade (MPa)",          "min": 250,   "max": 550,  "default": 415.0,  "unit": "MPa"},
    "column_width_mm":             {"label": "Column Width (mm)",          "min": 200,   "max": 800,  "default": 300.0,  "unit": "mm"},
    "column_depth_mm":             {"label": "Column Depth (mm)",          "min": 200,   "max": 800,  "default": 300.0,  "unit": "mm"},
    "num_floors":                  {"label": "Number of Floors",           "min": 1,     "max": 20,   "default": 5.0,    "unit": ""},
    "floor_load_kn_m2":            {"label": "Floor Load (kN/m²)",         "min": 2.0,   "max": 15.0, "default": 5.0,    "unit": "kN/m²"},
    "soil_bearing_capacity_kn_m2": {"label": "Soil Bearing Cap. (kN/m²)",  "min": 50,    "max": 500,  "default": 200.0,  "unit": "kN/m²"},
    "foundation_depth_m":          {"label": "Foundation Depth (m)",       "min": 0.5,   "max": 5.0,  "default": 1.5,    "unit": "m"},
    "seismic_zone":                {"label": "Seismic Zone",               "min": 1,     "max": 5,    "default": 2.0,    "unit": ""},
    "wind_speed_kmph":             {"label": "Wind Speed (km/h)",          "min": 20.0,  "max": 250.0,"default": 100.0,  "unit": "km/h"},
    "rainfall_mm_annual":          {"label": "Annual Rainfall (mm)",       "min": 200,   "max": 3000, "default": 800.0,  "unit": "mm"},
    "building_age_years":          {"label": "Building Age (years)",       "min": 0,     "max": 100,  "default": 10.0,   "unit": "years"},
    "shear_wall_present":          {"label": "Shear Wall Present",         "min": 0.0,   "max": 1.0,  "default": 1.0,    "unit": "0/1"},
    "foundation_type":             {"label": "Foundation Type",            "min": 0.0,   "max": 2.0,  "default": 1.0,    "unit": "0=Isolated,1=Raft,2=Pile"},
    "column_reinforcement_ratio":  {"label": "Column Reinf. Ratio",       "min": 0.008, "max": 0.06, "default": 0.02,   "unit": ""},
    "beam_depth_mm":               {"label": "Beam Depth (mm)",           "min": 300,   "max": 900,  "default": 500.0,  "unit": "mm"},
    "slab_thickness_mm":           {"label": "Slab Thickness (mm)",       "min": 100,   "max": 300,  "default": 150.0,  "unit": "mm"},
    "location_risk_index":         {"label": "Location Risk Index",       "min": 0.0,   "max": 1.0,  "default": 0.3,    "unit": "0-1"},
}


class ParameterExtractor:
    """
    Extracts 18 structural building parameters from raw document text
    using Groq LLaMA 3.3, with confidence scores and source snippets.
    """

    def _build_prompt(self, corpus: str) -> str:
        key_list = "\n".join(
            f'  - "{k}" ({s["label"]}, range {s["min"]}–{s["max"]} {s["unit"]})'
            for k, s in PARAM_SPEC.items()
        )
        return f"""You are an expert structural engineering data extractor.

From the document text below, extract ALL 18 building parameters into a JSON object.

### Required keys:
{key_list}

### Output format (strict JSON, no markdown):
{{
  "parameters": {{
    "<key>": {{
      "value": <float>,
      "confidence": <float 0.0-1.0>,
      "source_text": "<short verbatim quote from document>"
    }},
    ...
  }},
  "is_code_flags": [
    "<string describing any IS 456/IS 1893 code violation or concern>"
  ]
}}

### Rules:
1. Return ONLY raw JSON. No markdown, no explanation.
2. "value" must be a float within the specified range.
3. "confidence" = 1.0 if explicitly stated, 0.5-0.8 if inferred, 0.0 if guessed/defaulted.
4. "source_text" = short quote from document where the value was found, or "Not found in document" if missing.
5. If a parameter cannot be found, use a reasonable engineering default and set confidence to 0.0.
6. For "shear_wall_present": 1.0 = present, 0.0 = absent.
7. For "foundation_type": 0.0 = Isolated, 1.0 = Raft, 2.0 = Pile.
8. For "column_reinforcement_ratio": convert percentages to decimal (e.g. 2% → 0.02).
9. For "concrete_grade": M25 → 25.0, M40 → 40.0, etc.
10. For "steel_grade": Fe415 → 415.0, Fe500 → 500.0, etc.
11. In "is_code_flags", list any IS code violations you detect, such as:
    - Concrete grade below M20 (IS 456 minimum for RC)
    - Reinforcement ratio below 0.8% or above 6% (IS 456 Cl.26.5.3.1)
    - No shear walls in seismic zone ≥ 3 (IS 1893 Cl.7.6)
    - Foundation depth < 0.9m (IS 1904)

### Document text:
\"\"\"
{corpus[:8000]}
\"\"\"

Return ONLY the JSON object."""

    def extract(self, api_key: str, corpus: str) -> Dict[str, Any]:
        """
        Extract parameters from document corpus text.

        :param api_key: Groq API key
        :param corpus: Combined text from all PDFs (string)
        :returns: dict with "parameters" and "is_code_flags"
        """
        if not api_key:
            logger.warning("No API key provided, returning defaults")
            return self._default_result()

        if not corpus or not corpus.strip():
            logger.warning("Empty corpus, returning defaults")
            return self._default_result()

        try:
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert JSON structured data extractor for structural engineering documents. Return only raw JSON."},
                    {"role": "user", "content": self._build_prompt(corpus)},
                ],
                temperature=0.0,
                max_completion_tokens=2000,
            )

            content = response.choices[0].message.content.strip()

            # Strip markdown fences if the model adds them
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            result = json.loads(content.strip())

            # Validate and clamp values
            params = result.get("parameters", {})
            for key, spec in PARAM_SPEC.items():
                if key not in params:
                    params[key] = {
                        "value": spec["default"],
                        "confidence": 0.0,
                        "source_text": "Not found in document",
                    }
                else:
                    p = params[key]
                    val = float(p.get("value", spec["default"]))
                    val = max(spec["min"], min(spec["max"], val))
                    p["value"] = val
                    p["confidence"] = float(p.get("confidence", 0.0))
                    p["source_text"] = str(p.get("source_text", "Not found in document"))

            result["parameters"] = params
            if "is_code_flags" not in result:
                result["is_code_flags"] = []

            return result

        except Exception as e:
            logger.error(f"Parameter extraction failed: {e}")
            return self._default_result()

    def _default_result(self) -> Dict[str, Any]:
        """Return all defaults with zero confidence."""
        params = {}
        for key, spec in PARAM_SPEC.items():
            params[key] = {
                "value": spec["default"],
                "confidence": 0.0,
                "source_text": "Not found in document",
            }
        return {"parameters": params, "is_code_flags": []}


# Standalone test
if __name__ == "__main__":
    ext = ParameterExtractor()
    print("ParameterExtractor ready")
    print(f"Default result keys: {list(ext._default_result()['parameters'].keys())}")
