# StructSense AI — Structural Failure Detection System

AI-powered structural safety analysis using PPO Reinforcement Learning, IS 456/1893 engineering checks, and Groq LLaMA 3.3 70B reasoning.

---

## Getting Started

### Prerequisites

- Python 3.14+ (installed via [uv](https://docs.astral.sh/uv/) / Astral CPython)
- Dependencies listed in `requirements.txt`

### Run the App

From inside the `structsense/` directory, paste this command in your terminal:

```bash
py -m streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## Usage

1. Choose **Upload Documents** to extract parameters from PDF structural reports, or switch to **Manual Input** to adjust sliders directly.
2. Enter your **Groq API Key** in the sidebar (free at [console.groq.com](https://console.groq.com)).
3. Click **RUN SIMULATION** to train the PPO agent and get an AI safety verdict.
4. Explore results across the **Results**, **Trajectory**, **Risk Zones**, and **AI Verdict** tabs.
5. Export a PDF report, CSV data, or research abstract from the **Export** tab.

---

## Tech Stack

- **Streamlit** — Dashboard UI
- **Stable-Baselines3 + PyTorch** — PPO Reinforcement Learning
- **Groq LLaMA 3.3 70B** — AI structural reasoning
- **IS 456 / IS 1893** — Indian Standard code checks
- **Plotly** — Interactive visualizations
