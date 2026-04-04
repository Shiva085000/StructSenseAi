import os
import json
import logging
import PyPDF2
from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The 18 parameters needed for StructSense
REQUIRED_KEYS = [
    "concrete_grade", "steel_grade", "column_width_mm", "column_depth_mm",
    "num_floors", "floor_load_kn_m2", "soil_bearing_capacity_kn_m2",
    "foundation_depth_m", "seismic_zone", "wind_speed_kmph",
    "rainfall_mm_annual", "building_age_years", "shear_wall_present",
    "foundation_type", "column_reinforcement_ratio", "beam_depth_mm",
    "slab_thickness_mm", "location_risk_index"
]

class DocumentParser:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("Groq API Key not found. Parsing will be disabled.")

    def extract_text(self, file_uploader_obj) -> str:
        """Extract text from a Streamlit UploadedFile (PDF or TXT)."""
        if not file_uploader_obj:
            return ""
        
        filename = file_uploader_obj.name.lower()
        text = ""
        
        try:
            if filename.endswith(".pdf"):
                reader = PyPDF2.PdfReader(file_uploader_obj)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            elif filename.endswith(".txt"):
                text = file_uploader_obj.read().decode("utf-8")
            else:
                logger.error(f"Unsupported file format: {filename}")
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            
        return text

    def parse_parameters(self, document_text: str) -> dict:
        """Use Groq LLaMA 3.3 to extract the 18 parameters into a strict JSON dictionary."""
        if not self.client:
            return {}
            
        if not document_text.strip():
            return {}

        prompt = f"""
You are an expert structural engineering AI assistant.
Your task is to extract exact numerical values for 18 specific building parameters from the text below.

### Output Constraints:
1. You MUST return ONLY a valid JSON object. No markdown formatting, no explanations, no text before or after the JSON.
2. The JSON keys MUST EXACTLY match the following 18 keys.
3. If a value is not found or cannot be reasonably inferred from the text, use `null`.
4. Pay attention to the REQUIRED UNITS in the key names (e.g., _mm, _m, _kn_m2). Convert if necessary.

### The 18 Keys & Expected Types:
- "concrete_grade" (float, e.g., M25 -> 25.0)
- "steel_grade" (float, e.g., Fe415 -> 415.0)
- "column_width_mm" (float, in millimeters)
- "column_depth_mm" (float, in millimeters)
- "num_floors" (float, number of stories)
- "floor_load_kn_m2" (float, in kN/m2)
- "soil_bearing_capacity_kn_m2" (float, safe bearing capacity in kN/m2)
- "foundation_depth_m" (float, in meters)
- "seismic_zone" (float, IS code zone 1 to 5)
- "wind_speed_kmph" (float, in km/h)
- "rainfall_mm_annual" (float, in mm)
- "building_age_years" (float, in years)
- "shear_wall_present" (float, 1.0 for yes/present, 0.0 for no/absent)
- "foundation_type" (float, 0.0 for Isolated, 1.0 for Raft, 2.0 for Pile)
- "column_reinforcement_ratio" (float, a decimal between 0.008 to 0.06, e.g., 2% -> 0.02)
- "beam_depth_mm" (float, in millimeters)
- "slab_thickness_mm" (float, in millimeters)
- "location_risk_index" (float, between 0.0 and 1.0)

### Document Text:
\"\"\"
{document_text[:6000]}
\"\"\"

Return ONLY the JSON object.
"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert JSON structured data extractor. You must only return raw JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_completion_tokens=800
            )
            
            content = response.choices[0].message.content.strip()
            # Clean up potential markdown formatting if LLaMA still returns it
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
                
            parsed_data = json.loads(content.strip())
            
            # Filter to only the keys we actually want, and drop nulls
            filtered_data = {
                k: v for k, v in parsed_data.items() 
                if k in REQUIRED_KEYS and v is not None
            }
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error calling Groq API or parsing JSON: {e}")
            return {}
