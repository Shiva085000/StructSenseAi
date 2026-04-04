import google.generativeai as genai
import warnings, pathlib
warnings.filterwarnings("ignore")

genai.configure(api_key="AIzaSyAZNV_FcAyHX2Imus1xQRV2Sw37NX4krRo")
models = [m.name for m in genai.list_models()
          if "generateContent" in m.supported_generation_methods]
out = "\n".join(sorted(models))
pathlib.Path("_models.txt").write_text(out)
print(f"Wrote {len(models)} models to _models.txt")
print(out)
