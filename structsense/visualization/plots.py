"""
StructurePlotter – Interactive and publication-quality visualizations for StructSense AI.
Uses Plotly for interactive charts and Matplotlib for export-quality figures.
"""
import os
import io
import base64
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Any, Optional


# ────────────────────────── Colour palette ────────────────────────────── #
RISK_PALETTE = {
    "LOW":      "#00c853",
    "MODERATE": "#ffab00",
    "HIGH":     "#ff6d00",
    "CRITICAL": "#d50000",
}

DARK_BG = "#0e1117"
CARD_BG = "#1a1d26"
TEXT    = "#e0e0e0"
GRID    = "#2a2d3a"
ACCENT  = "#7c4dff"


class StructurePlotter:
    """Collection of visualizations for structural risk analysis."""

    # ------------------------------------------------------------------ #
    #  1. Risk Gauge                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def risk_gauge(score: float, label: str = "") -> go.Figure:
        """Plotly gauge chart showing overall risk score 0–100."""
        color = RISK_PALETTE.get(label, "#ff6d00")
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            delta={"reference": 50, "increasing": {"color": "#d50000"},
                   "decreasing": {"color": "#00c853"}},
            title={"text": f"Risk Score<br><span style='font-size:0.7em;color:{color}'>{label}</span>",
                   "font": {"color": TEXT, "size": 18}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": TEXT,
                         "tickfont": {"color": TEXT}},
                "bar":  {"color": color, "thickness": 0.35},
                "bgcolor": CARD_BG,
                "bordercolor": GRID,
                "steps": [
                    {"range": [0,  30], "color": "#0a2e1a"},
                    {"range": [30, 55], "color": "#2e2000"},
                    {"range": [55, 75], "color": "#2e1600"},
                    {"range": [75, 100],"color": "#2e0000"},
                ],
                "threshold": {
                    "line":  {"color": "white", "width": 3},
                    "thickness": 0.85,
                    "value": score,
                },
            },
            number={"suffix": "/100", "font": {"color": TEXT}},
        ))
        fig.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            height=300, margin=dict(t=60, b=20, l=30, r=30),
            font={"color": TEXT},
        )
        return fig

    # ------------------------------------------------------------------ #
    #  2. Stress Trajectory (PPO)                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def stress_trajectory(
        steps: List[int],
        risks: List[float],
        title: str = "PPO Stress Trajectory",
    ) -> go.Figure:
        """Animated line chart of risk score over PPO episode steps."""
        if not steps or not risks:
            steps = list(range(5))
            risks = [40, 45, 50, 48, 52]

        fig = go.Figure()

        # Risk zones
        fig.add_hrect(y0=0,  y1=30,  fillcolor="#00c853", opacity=0.07, line_width=0)
        fig.add_hrect(y0=30, y1=55,  fillcolor="#ffab00", opacity=0.07, line_width=0)
        fig.add_hrect(y0=55, y1=75,  fillcolor="#ff6d00", opacity=0.07, line_width=0)
        fig.add_hrect(y0=75, y1=100, fillcolor="#d50000", opacity=0.07, line_width=0)

        # Gradient fill under curve
        fig.add_trace(go.Scatter(
            x=steps, y=risks,
            fill="tozeroy",
            fillcolor="rgba(124, 77, 255, 0.15)",
            line=dict(color=ACCENT, width=2.5, dash="solid"),
            mode="lines+markers",
            marker=dict(size=6, color=ACCENT, symbol="circle"),
            name="Risk Score",
            hovertemplate="Step %{x}<br>Risk: %{y:.1f}<extra></extra>",
        ))

        # Threshold lines
        for y, label, color in [(30, "Low→Moderate", "#ffab00"),
                                 (55, "Moderate→High", "#ff6d00"),
                                 (75, "High→Critical", "#d50000")]:
            fig.add_hline(y=y, line_dash="dot", line_color=color,
                          opacity=0.6, annotation_text=label,
                          annotation_font_color=color,
                          annotation_position="bottom right")

        fig.update_layout(
            title=dict(text=title, font=dict(color=TEXT, size=16)),
            xaxis=dict(title="Episode Step", gridcolor=GRID, color=TEXT,
                       showgrid=True, zeroline=False),
            yaxis=dict(title="Risk Score", range=[0, 105], gridcolor=GRID,
                       color=TEXT, showgrid=True),
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            legend=dict(bgcolor=CARD_BG, bordercolor=GRID, font=dict(color=TEXT)),
            height=400,
            margin=dict(t=60, b=50, l=60, r=30),
            font=dict(color=TEXT),
            hovermode="x unified",
        )
        return fig

    # ------------------------------------------------------------------ #
    #  3. Scenario Radar Chart                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def scenario_radar(scenario_scores: Dict[str, float]) -> go.Figure:
        """Polar radar chart of risk across 5 stress scenarios."""
        categories = list(scenario_scores.keys())
        values     = list(scenario_scores.values())
        # Close the polygon
        categories_closed = categories + [categories[0]]
        values_closed     = values + [values[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor="rgba(124, 77, 255, 0.25)",
            line=dict(color=ACCENT, width=2),
            marker=dict(size=8, color=ACCENT),
            name="Risk per Scenario",
            hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
        ))

        fig.update_layout(
            polar=dict(
                bgcolor=CARD_BG,
                radialaxis=dict(
                    visible=True, range=[0, 100],
                    gridcolor=GRID, tickcolor=TEXT,
                    tickfont=dict(color=TEXT, size=10),
                    linecolor=GRID,
                ),
                angularaxis=dict(
                    gridcolor=GRID, linecolor=GRID,
                    tickfont=dict(color=TEXT, size=11),
                ),
            ),
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            title=dict(text="Stress Scenario Risk Profile", font=dict(color=TEXT, size=16)),
            font=dict(color=TEXT),
            showlegend=False,
            height=400,
            margin=dict(t=70, b=30, l=50, r=50),
        )
        return fig

    # ------------------------------------------------------------------ #
    #  4. Monte Carlo Histogram                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def monte_carlo_histogram(mc_data: Dict[str, Any]) -> go.Figure:
        """Plotly histogram of Monte Carlo risk distribution with KDE overlay."""
        samples      = mc_data.get("samples", [])
        kde_x        = mc_data.get("kde_x", [])
        kde_y        = mc_data.get("kde_y", [])
        mean         = mc_data.get("mean", 0)
        p5           = mc_data.get("p5", 0)
        p95          = mc_data.get("p95", 0)
        fail_prob    = mc_data.get("failure_prob", 0)

        fig = go.Figure()

        # Risk band fills
        fig.add_vrect(x0=0,  x1=30,  fillcolor="#00c853", opacity=0.06, line_width=0)
        fig.add_vrect(x0=30, x1=55,  fillcolor="#ffab00", opacity=0.06, line_width=0)
        fig.add_vrect(x0=55, x1=75,  fillcolor="#ff6d00", opacity=0.06, line_width=0)
        fig.add_vrect(x0=75, x1=100, fillcolor="#d50000", opacity=0.06, line_width=0)

        # Histogram
        fig.add_trace(go.Histogram(
            x=samples,
            nbinsx=40,
            marker_color=ACCENT,
            opacity=0.65,
            name="Simulation Samples",
            hovertemplate="Risk: %{x:.1f}<br>Count: %{y}<extra></extra>",
        ))

        # KDE curve (scaled to histogram)
        if kde_x and kde_y:
            n_samples = len(samples)
            bin_width  = (max(samples) - min(samples)) / 40 if samples else 1
            kde_scaled = [y * n_samples * bin_width for y in kde_y]
            fig.add_trace(go.Scatter(
                x=kde_x, y=kde_scaled,
                mode="lines",
                line=dict(color="#ff6d00", width=2.5),
                name="KDE",
                hovertemplate="Risk: %{x:.1f}<extra></extra>",
            ))

        # Annotation lines
        fig.add_vline(x=mean,  line_dash="dash",  line_color="white",
                      annotation_text=f"μ={mean:.1f}", annotation_font_color="white")
        fig.add_vline(x=p5,    line_dash="dot",   line_color="#00c853",
                      annotation_text=f"P5={p5:.1f}", annotation_font_color="#00c853")
        fig.add_vline(x=p95,   line_dash="dot",   line_color="#d50000",
                      annotation_text=f"P95={p95:.1f}", annotation_font_color="#d50000")

        fig.update_layout(
            title=dict(
                text=f"Monte Carlo Risk Distribution (n=1000) | P(Critical)={fail_prob*100:.1f}%",
                font=dict(color=TEXT, size=15),
            ),
            xaxis=dict(title="Risk Score", gridcolor=GRID, color=TEXT, range=[0, 100]),
            yaxis=dict(title="Frequency", gridcolor=GRID, color=TEXT),
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            legend=dict(bgcolor=CARD_BG, bordercolor=GRID, font=dict(color=TEXT)),
            barmode="overlay",
            height=420,
            margin=dict(t=70, b=50, l=60, r=30),
            font=dict(color=TEXT),
        )
        return fig

    # ------------------------------------------------------------------ #
    #  5. Component Breakdown Bar                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def component_breakdown_bar(breakdown: Dict[str, float]) -> go.Figure:
        """Horizontal bar chart of risk component contributions."""
        if not breakdown:
            return go.Figure()

        components = list(breakdown.keys())
        values     = list(breakdown.values())
        max_vals   = {"Seismic Vulnerability": 25, "Age & Maintenance": 20,
                      "Load Capacity": 20, "Foundation & Soil": 15,
                      "Wind Exposure": 10, "Concrete Quality": 10}

        bar_colors = []
        for comp, val in zip(components, values):
            max_v = max_vals.get(comp, 25)
            ratio = val / max_v
            if ratio < 0.4:
                bar_colors.append("#00c853")
            elif ratio < 0.65:
                bar_colors.append("#ffab00")
            elif ratio < 0.85:
                bar_colors.append("#ff6d00")
            else:
                bar_colors.append("#d50000")

        fig = go.Figure(go.Bar(
            x=values, y=components,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.1f}" for v in values],
            textposition="outside",
            textfont=dict(color=TEXT),
            hovertemplate="%{y}: %{x:.2f} pts<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text="Risk Component Breakdown", font=dict(color=TEXT, size=16)),
            xaxis=dict(title="Risk Points", gridcolor=GRID, color=TEXT, showgrid=True),
            yaxis=dict(color=TEXT, showgrid=False),
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            height=380, margin=dict(t=60, b=50, l=200, r=60),
            font=dict(color=TEXT),
            showlegend=False,
        )
        return fig

    # ------------------------------------------------------------------ #
    #  6. Parameter Heatmap (Multi-blueprint)                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def parameter_heatmap(blueprints: List[Dict[str, Any]]) -> go.Figure:
        """Normalised heatmap of key parameters for all blueprints."""
        keys    = ["floors", "age_years", "concrete_strength_mpa",
                   "maintenance_score", "seismic_zone", "wind_load_kn_m2"]
        labels  = ["Floors", "Age (y)", "Concrete (MPa)",
                   "Maintenance", "Seismic Zone", "Wind (kN/m²)"]
        names   = [b.get("name", b.get("building_id", f"BP{i}")) for i, b in enumerate(blueprints)]

        # Build matrix
        matrix = []
        for k in keys:
            row = [float(b.get(k, 0)) for b in blueprints]
            row_min, row_max = min(row), max(row)
            span = row_max - row_min if row_max != row_min else 1
            row_norm = [(v - row_min) / span for v in row]
            matrix.append(row_norm)

        fig = go.Figure(go.Heatmap(
            z=matrix,
            x=names,
            y=labels,
            colorscale="RdYlGn_r",
            showscale=True,
            colorbar=dict(title="Normalised", tickfont=dict(color=TEXT),
                          titlefont=dict(color=TEXT)),
            hovertemplate="Building: %{x}<br>Parameter: %{y}<br>Normalised: %{z:.2f}<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text="Blueprint Parameter Comparison Heatmap", font=dict(color=TEXT, size=16)),
            xaxis=dict(color=TEXT, tickangle=-45),
            yaxis=dict(color=TEXT),
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            height=400,
            margin=dict(t=60, b=120, l=150, r=30),
            font=dict(color=TEXT),
        )
        return fig

    # ------------------------------------------------------------------ #
    #  7. Publication Plot (Matplotlib)                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def publication_plot(
        mc_data: Dict[str, Any],
        scenario_scores: Dict[str, float],
        breakdown: Dict[str, float],
        building_name: str = "Structure",
        save_path: Optional[str] = None,
    ) -> str:
        """
        Generates a 3-panel publication-quality Matplotlib figure.
        Returns the figure as a base64 PNG string (for Streamlit display).
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#0e1117")
        fig.suptitle(
            f"StructSense AI – {building_name}",
            color="white", fontsize=14, fontweight="bold", y=1.02,
        )

        # Panel 1: Monte Carlo histogram
        ax1 = axes[0]
        ax1.set_facecolor("#1a1d26")
        samples = mc_data.get("samples", [50] * 100)
        ax1.hist(samples, bins=30, color="#7c4dff", alpha=0.75, edgecolor="none")
        ax1.axvline(mc_data.get("mean", 50), color="white", lw=1.5, linestyle="--",
                    label=f"μ={mc_data.get('mean', 50):.1f}")
        ax1.axvline(mc_data.get("p95", 70), color="#d50000", lw=1.5,
                    linestyle=":", label=f"P95={mc_data.get('p95', 70):.1f}")
        ax1.set_xlabel("Risk Score", color="white")
        ax1.set_ylabel("Frequency", color="white")
        ax1.set_title("Monte Carlo Distribution", color="white")
        ax1.tick_params(colors="white")
        ax1.legend(framealpha=0, labelcolor="white", fontsize=8)
        for spine in ax1.spines.values():
            spine.set_edgecolor("#2a2d3a")

        # Panel 2: Scenario bar chart
        ax2 = axes[1]
        ax2.set_facecolor("#1a1d26")
        sc_labels = list(scenario_scores.keys())
        sc_vals   = list(scenario_scores.values())
        colors    = [("#d50000" if v >= 75 else
                      "#ff6d00" if v >= 55 else
                      "#ffab00" if v >= 30 else "#00c853") for v in sc_vals]
        bars = ax2.barh(sc_labels, sc_vals, color=colors, edgecolor="none", height=0.55)
        for bar, val in zip(bars, sc_vals):
            ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f"{val:.0f}", va="center", color="white", fontsize=9)
        ax2.set_xlim(0, 110)
        ax2.set_xlabel("Risk Score", color="white")
        ax2.set_title("Scenario Risk Profile", color="white")
        ax2.tick_params(colors="white")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#2a2d3a")

        # Panel 3: Component breakdown
        ax3 = axes[2]
        ax3.set_facecolor("#1a1d26")
        bd_labels = list(breakdown.keys())
        bd_vals   = list(breakdown.values())
        bd_colors = ["#7c4dff", "#536dfe", "#4dd0e1", "#69f0ae", "#ffd740", "#ff6e40"]
        wedges, texts, autotexts = ax3.pie(
            bd_vals,
            labels=None,
            autopct="%1.0f%%",
            startangle=140,
            colors=bd_colors[:len(bd_vals)],
            pctdistance=0.75,
            wedgeprops=dict(edgecolor="#0e1117", linewidth=2),
        )
        for at in autotexts:
            at.set_color("white")
            at.set_fontsize(8)
        legend_patches = [mpatches.Patch(facecolor=c, label=l)
                          for c, l in zip(bd_colors, bd_labels)]
        ax3.legend(handles=legend_patches, loc="lower center",
                   bbox_to_anchor=(0.5, -0.3), ncol=2, framealpha=0,
                   labelcolor="white", fontsize=7)
        ax3.set_title("Risk Component Share", color="white")

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight",
                        facecolor="#0e1117", edgecolor="none")

        # Encode to base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor="#0e1117", edgecolor="none")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return b64
