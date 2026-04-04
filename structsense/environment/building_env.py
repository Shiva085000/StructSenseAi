"""
BuildingStressEnv – Upgraded Custom Gymnasium Environment (18-dimensional state space)
======================================================================================
Simulates 5 structural stress scenarios on a building using simplified IS 456 / IS 1893
engineering formulas to determine pass/fail and compute rewards.

State Space (18 continuous/encoded features):
    0  concrete_grade               MPa  [15 – 60]
    1  steel_grade                  MPa  [250 – 550]
    2  column_width_mm              mm   [200 – 800]
    3  column_depth_mm              mm   [200 – 800]
    4  num_floors                   int  [1 – 20]
    5  floor_load_kn_m2             kN/m²[2 – 15]
    6  soil_bearing_capacity_kn_m2  kN/m²[50 – 500]
    7  foundation_depth_m           m    [0.5 – 5.0]
    8  seismic_zone                 int  [1 – 5]
    9  wind_speed_kmph              km/h [20 – 250]
   10  rainfall_mm_annual           mm   [200 – 3000]
   11  building_age_years           yr   [0 – 100]
   12  shear_wall_present           bin  {0, 1}
   13  foundation_type              enc  {0=isolated, 1=raft, 2=pile}
   14  column_reinforcement_ratio   —    [0.008 – 0.06]
   15  beam_depth_mm                mm   [300 – 900]
   16  slab_thickness_mm            mm   [100 – 300]
   17  location_risk_index          —    [0 – 1]

Action Space (5 discrete stress scenarios):
    0  Normal occupancy load test
    1  Seismic stress test          (IS 1893 base-shear method)
    2  Wind + rain combined test
    3  Overload test                (150 % capacity)
    4  Long-term degradation test   (age-factor degradation)

Reward:
    +1.0           if structure passes the scenario
    -1.0 × rw      if structure fails  (rw = risk_weight ∈ [0.5, 2.0])
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ─────────────────────────── physical constants ──────────────────────────── #
IS_AH_ZONE = {1: 0.02, 2: 0.04, 3: 0.08, 4: 0.12, 5: 0.16}   # IS 1893 Ah coefficients
SAFETY_FACTOR_COLUMN = 1.0          # already factored inside IS 456 formula
DEGRADATION_RATE     = 0.008        # 0.8 % capacity loss per year
MIN_CAPACITY_RATIO   = 0.30         # below this fraction → catastrophic failure


# ─────────────────────────── dimension bounds ────────────────────────────── #
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "concrete_grade":                (15.0, 60.0),
    "steel_grade":                   (250.0, 550.0),
    "column_width_mm":               (200.0, 800.0),
    "column_depth_mm":               (200.0, 800.0),
    "num_floors":                    (1.0,  20.0),
    "floor_load_kn_m2":              (2.0,  15.0),
    "soil_bearing_capacity_kn_m2":   (50.0, 500.0),
    "foundation_depth_m":            (0.5,  5.0),
    "seismic_zone":                  (1.0,  5.0),
    "wind_speed_kmph":               (20.0, 250.0),
    "rainfall_mm_annual":            (200.0, 3000.0),
    "building_age_years":            (0.0,  100.0),
    "shear_wall_present":            (0.0,  1.0),
    "foundation_type":               (0.0,  2.0),
    "column_reinforcement_ratio":    (0.008, 0.06),
    "beam_depth_mm":                 (300.0, 900.0),
    "slab_thickness_mm":             (100.0, 300.0),
    "location_risk_index":           (0.0,  1.0),
}

PARAM_KEYS: List[str] = list(PARAM_BOUNDS.keys())   # ordered; index == obs position

ACTION_NAMES: Dict[int, str] = {
    0: "Normal Occupancy Load Test",
    1: "Seismic Stress Test",
    2: "Wind + Rain Combined Test",
    3: "Overload Test (150 %)",
    4: "Long-Term Degradation Test",
}


# ══════════════════════════════════════════════════════════════════════════════
class BuildingStressEnv(gym.Env):
    """Custom Gymnasium environment for structural stress simulation."""

    metadata = {"render_modes": ["human"]}

    # ------------------------------------------------------------------ init #
    def __init__(
        self,
        blueprint_params: Dict[str, Any],
        max_steps: int = 20,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.blueprint_params = blueprint_params
        self.max_steps        = max_steps
        self.render_mode      = render_mode

        # Spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(18,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        # Episode state
        self._step_count:       int         = 0
        self._state:            np.ndarray  = np.zeros(18, dtype=np.float32)
        self._raw_params:       Dict        = {}
        self._risk_trajectory:  List[float] = []
        self._damage_log:       List[Dict]  = []
        self._action_history:   List[int]   = []
        self._cumulative_damage: float      = 0.0

    # ─────────────────────────────────────────────── helpers ─────────────── #
    @staticmethod
    def _normalize(key: str, value: float) -> float:
        lo, hi = PARAM_BOUNDS[key]
        return float(np.clip((value - lo) / (hi - lo + 1e-9), 0.0, 1.0))

    def _params_to_state(self, params: Dict[str, Any]) -> np.ndarray:
        arr = np.array(
            [self._normalize(k, float(params.get(k, (PARAM_BOUNDS[k][0] + PARAM_BOUNDS[k][1]) / 2)))
             for k in PARAM_KEYS],
            dtype=np.float32,
        )
        return arr

    @staticmethod
    def _denorm(key: str, norm_val: float) -> float:
        lo, hi = PARAM_BOUNDS[key]
        return norm_val * (hi - lo) + lo

    def _raw(self, key: str) -> float:
        """Return the current raw (un-normalised) value for a parameter."""
        return self._denorm(key, float(self._state[PARAM_KEYS.index(key)]))

    # ───────────────────────────── default params ────────────────────────── #
    @staticmethod
    def _default_params() -> Dict[str, float]:
        return {
            "concrete_grade":               25.0,
            "steel_grade":                  415.0,
            "column_width_mm":              300.0,
            "column_depth_mm":              300.0,
            "num_floors":                   5.0,
            "floor_load_kn_m2":             5.0,
            "soil_bearing_capacity_kn_m2":  200.0,
            "foundation_depth_m":           1.5,
            "seismic_zone":                 2.0,
            "wind_speed_kmph":              100.0,
            "rainfall_mm_annual":           800.0,
            "building_age_years":           10.0,
            "shear_wall_present":           1.0,
            "foundation_type":              1.0,
            "column_reinforcement_ratio":   0.02,
            "beam_depth_mm":                450.0,
            "slab_thickness_mm":            150.0,
            "location_risk_index":          0.3,
        }

    # ──────────────────────── IS 456 column capacity (kN) ───────────────── #
    def _column_capacity_kn(self) -> float:
        """
        IS 456 Cl. 39.3 – Short column axial capacity
            Pu = 0.4 × fck × Ac + 0.67 × fy × Asc
        fck  = concrete characteristic compressive strength (MPa = N/mm²)
        fy   = yield strength of steel (MPa = N/mm²)
        Ac   = net concrete area (mm²) = b×d − Asc
        Asc  = steel area = ratio × (b × d)
        Returns capacity in kN.
        """
        fck  = self._raw("concrete_grade")          # MPa
        fy   = self._raw("steel_grade")             # MPa
        b    = self._raw("column_width_mm")         # mm
        d    = self._raw("column_depth_mm")         # mm
        rho  = self._raw("column_reinforcement_ratio")

        gross_area = b * d                           # mm²
        Asc        = rho * gross_area                # mm²
        Ac         = gross_area - Asc                # mm²

        Pu_kN = (0.4 * fck * Ac + 0.67 * fy * Asc) / 1000.0  # N → kN
        return max(Pu_kN, 1.0)

    # ───────────────────── applied gravity load per column (kN) ─────────── #
    def _applied_gravity_load_kn(self, overload_factor: float = 1.0) -> float:
        """
        Simplified tributary-area-based gravity load.
        Assumes each internal column carries ~25 m² floor area (5 m × 5 m bay).
        """
        q         = self._raw("floor_load_kn_m2")   # kN/m²
        floors    = self._raw("num_floors")
        trib_area = 25.0                             # m² per column (typical bay)
        dead_load_factor = 1.25                      # DL + LL combined factor

        applied = q * trib_area * floors * dead_load_factor * overload_factor
        return applied

    # ─────────────────────── IS 1893 seismic base shear (kN) ────────────── #
    def _seismic_base_shear_kn(self) -> float:
        """
        IS 1893 (Part 1) – Simplified lateral base shear
            VB = Ah × W
        Ah = seismic coefficient (zone-based + response reduction)
        W  = seismic weight ≈ DL + 0.25 LL  (simplified as total load)
        """
        zone = int(round(self._raw("seismic_zone")))
        Ah   = IS_AH_ZONE.get(zone, 0.04)

        # Reduce Ah if shear walls present (response reduction factor R ~ 5 vs 3)
        if self._raw("shear_wall_present") >= 0.5:
            Ah *= 0.60          # shear walls halve the effective Ah

        # Seismic weight (kN) per column (25 m² tributary, all floors)
        q      = self._raw("floor_load_kn_m2")
        floors = self._raw("num_floors")
        W_col  = q * 25.0 * floors * 1.0   # 1.0 = no LL reduction for simplicity

        VB = Ah * W_col
        return VB

    # ─────────────────────── wind pressure (kN on column) ───────────────── #
    def _wind_lateral_load_kn(self) -> float:
        """
        IS 875 (Part 3) simplified – wind pressure on tributary column face.
            qz = 0.6 × Vz² / 1000   (kN/m²)  where Vz in m/s
        Applied over column height per floor (assume 3 m storey height).
        """
        v_kmph = self._raw("wind_speed_kmph")
        v_ms   = v_kmph / 3.6                        # convert to m/s
        qz     = 0.6 * (v_ms ** 2) / 1000.0          # kN/m²

        rain_factor = 1.0 + self._raw("rainfall_mm_annual") / 3000.0 * 0.1   # ≤ 10 % amplification

        floors         = self._raw("num_floors")
        storey_height  = 3.0                          # m
        column_face    = 0.3                          # m (assumed column width exposed)
        wind_on_column = qz * rain_factor * column_face * storey_height * floors
        return wind_on_column

    # ──────────────────── IS 456 soil-bearing check (kN) ────────────────── #
    def _soil_pressure_check(self, overload_factor: float = 1.0) -> Tuple[bool, str]:
        """
        Check if the footing pressure exceeds allowable soil bearing capacity.
        Simplified: assumes isolated footing 1.5 m × 1.5 m.
        """
        allow_sbc = self._raw("soil_bearing_capacity_kn_m2")
        load      = self._applied_gravity_load_kn(overload_factor)
        foot_area = 1.5 * 1.5                         # m² – simplified footing
        applied_pressure = load / foot_area           # kN/m²

        depth_bonus = min(self._raw("foundation_depth_m") / 5.0, 0.4) * allow_sbc
        eff_sbc = allow_sbc + depth_bonus

        # Pile foundation allows 3× the SBC
        if int(round(self._raw("foundation_type"))) == 2:
            eff_sbc *= 3.0
        # Raft allows 1.5×
        elif int(round(self._raw("foundation_type"))) == 1:
            eff_sbc *= 1.5

        if applied_pressure > eff_sbc:
            return False, (f"Soil bearing failure: applied {applied_pressure:.1f} kN/m² "
                           f"> allowable {eff_sbc:.1f} kN/m²")
        return True, ""

    # ─────────────────── age-based degradation factor ────────────────────── #
    def _degradation_factor(self) -> float:
        """Returns a multiplier in [0.30, 1.0] reflecting age-driven capacity loss."""
        age = self._raw("building_age_years")
        lost = DEGRADATION_RATE * age          # 0.8 % per year
        return float(max(MIN_CAPACITY_RATIO, 1.0 - lost))

    # ──────────────────────── risk weight ────────────────────────────────── #
    def _risk_weight(self) -> float:
        """
        Scalar risk weight ∈ [0.5, 2.0].
        Derived from: seismic zone, location_risk_index, age, and maintenance proxy.
        Analogous to a DSW (Dynamic Structural Weight) output.
        """
        zone_w   = self._raw("seismic_zone") / 5.0
        loc_w    = self._raw("location_risk_index")
        age_w    = self._raw("building_age_years") / 100.0
        maint_w  = 1.0 - float(self._raw("shear_wall_present"))   # no shear wall ↑ risk

        raw = 0.35 * zone_w + 0.30 * loc_w + 0.20 * age_w + 0.15 * maint_w
        return float(np.clip(0.5 + raw * 1.5, 0.5, 2.0))

    # ──────────────────────── overall risk score 0-100 ───────────────────── #
    def _compute_risk_score(self) -> float:
        cap   = self._column_capacity_kn()
        load  = self._applied_gravity_load_kn()
        ratio = load / (cap + 1e-6)               # demand / capacity
        seismic_contrib = (self._raw("seismic_zone") / 5.0) * 25
        age_contrib     = (self._raw("building_age_years") / 100.0) * 20
        wind_contrib    = (self._raw("wind_speed_kmph") / 250.0) * 15
        soil_contrib    = (1.0 - self._raw("soil_bearing_capacity_kn_m2") / 500.0) * 15
        load_contrib    = min(ratio, 1.0) * 25

        raw = seismic_contrib + age_contrib + wind_contrib + soil_contrib + load_contrib
        return float(np.clip(raw, 0, 100))

    # ══════════════════════════════════════════════════════════════════════ #
    #  RESET                                                                 #
    # ══════════════════════════════════════════════════════════════════════ #
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self._step_count        = 0
        self._cumulative_damage = 0.0
        self._risk_trajectory   = []
        self._action_history    = []
        self._damage_log        = []

        # Merge blueprint params over defaults; apply small RNG noise for exploration
        base = self._default_params()
        base.update({k: v for k, v in self.blueprint_params.items() if k in PARAM_BOUNDS})
        self._raw_params = base

        if seed is not None:
            rng = np.random.default_rng(seed)
            # ±2 % Gaussian jitter on most params for robustness
            for k in PARAM_KEYS:
                if k not in ("shear_wall_present", "foundation_type", "seismic_zone"):
                    lo, hi = PARAM_BOUNDS[k]
                    span = hi - lo
                    base[k] = float(np.clip(
                        base.get(k, (lo + hi) / 2) + rng.normal(0, span * 0.02),
                        lo, hi
                    ))

        self._state = self._params_to_state(base)
        base_risk   = self._compute_risk_score()
        self._risk_trajectory.append(base_risk)

        info = {
            "base_risk":              base_risk,
            "column_capacity_kn":     round(self._column_capacity_kn(), 1),
            "applied_load_kn":        round(self._applied_gravity_load_kn(), 1),
            "step":                   0,
            "degradation_factor":     round(self._degradation_factor(), 3),
        }
        return self._state.copy(), info

    # ══════════════════════════════════════════════════════════════════════ #
    #  STEP                                                                  #
    # ══════════════════════════════════════════════════════════════════════ #
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Apply one stress scenario and return (obs, reward, terminated, truncated, info).

        Engineering checks performed per action:
            0 – Gravity load vs IS 456 column capacity
            1 – Seismic base shear vs lateral capacity (function of shear walls / columns)
            2 – Wind + rain amplified lateral pressure vs column capacity
            3 – 150 % overload applied to gravity + soil checks
            4 – Age-degraded capacity vs gravity load
        """
        assert self.action_space.contains(action), f"Invalid action {action}"
        self._step_count += 1

        passed         = False
        failure_reason = ""
        extra_info: Dict[str, Any] = {}

        # ── 1. Degrade state with cumulative damage already accumulated ── #
        deg_factor = max(self._degradation_factor() - self._cumulative_damage * 0.05, 0.1)

        # ── 2. Effective capacity after degradation ─────────────────────── #
        eff_cap_kn = self._column_capacity_kn() * deg_factor

        # ══════════════════════════════════════════════════════════════════
        # ACTION 0: Normal occupancy load test
        # ══════════════════════════════════════════════════════════════════
        if action == 0:
            applied = self._applied_gravity_load_kn(overload_factor=1.0)
            soil_ok, soil_msg = self._soil_pressure_check(overload_factor=1.0)

            if not soil_ok:
                failure_reason = f"[Normal Load] {soil_msg}"
            elif applied > eff_cap_kn:
                failure_reason = (
                    f"[Normal Load] Column axial failure: "
                    f"applied {applied:.1f} kN > capacity {eff_cap_kn:.1f} kN "
                    f"(IS 456 Cl.39.3)"
                )
            else:
                passed = True

            extra_info = {"applied_kn": round(applied, 1),
                          "capacity_kn": round(eff_cap_kn, 1),
                          "demand_ratio": round(applied / (eff_cap_kn + 1e-6), 3)}

        # ══════════════════════════════════════════════════════════════════
        # ACTION 1: Seismic stress test (IS 1893)
        # ══════════════════════════════════════════════════════════════════
        elif action == 1:
            VB = self._seismic_base_shear_kn()

            # Lateral column capacity estimate: 20 % of axial capacity
            # (shear walls increase lateral capacity significantly)
            shear_wall_bonus = 2.5 if self._raw("shear_wall_present") >= 0.5 else 1.0
            lateral_cap_kn   = eff_cap_kn * 0.20 * shear_wall_bonus

            # Height amplification: taller buildings see higher overturning
            floors        = self._raw("num_floors")
            height_factor = 1.0 + (floors - 1) * 0.05   # 5 % amplification per storey above 1

            effective_VB = VB * height_factor

            if effective_VB > lateral_cap_kn:
                failure_reason = (
                    f"[Seismic] IS 1893 base shear {effective_VB:.1f} kN "
                    f"> lateral capacity {lateral_cap_kn:.1f} kN "
                    f"(Zone {int(round(self._raw('seismic_zone')))}, "
                    f"Ah={IS_AH_ZONE.get(int(round(self._raw('seismic_zone'))), 0.04):.2f})"
                )
            else:
                passed = True

            extra_info = {"base_shear_kn": round(effective_VB, 1),
                          "lateral_capacity_kn": round(lateral_cap_kn, 1),
                          "seismic_zone": int(round(self._raw("seismic_zone"))),
                          "shear_wall": bool(self._raw("shear_wall_present") >= 0.5)}

        # ══════════════════════════════════════════════════════════════════
        # ACTION 2: Wind + rain combined test (IS 875 Part 3)
        # ══════════════════════════════════════════════════════════════════
        elif action == 2:
            wind_load = self._wind_lateral_load_kn()

            # Rainfall causes additional hydrostatic pressure on foundation
            rain_mm   = self._raw("rainfall_mm_annual")
            hydro_kn  = (rain_mm / 3000.0) * 50.0    # up to 50 kN extra

            total_lateral = wind_load + hydro_kn

            # Lateral capacity – same logic as seismic
            shear_wall_bonus = 2.0 if self._raw("shear_wall_present") >= 0.5 else 1.0
            lateral_cap_kn   = eff_cap_kn * 0.15 * shear_wall_bonus

            # Also check foundation erosion due to high rainfall
            foundation_erode = rain_mm > 2000 and self._raw("foundation_depth_m") < 1.0

            if foundation_erode:
                failure_reason = (
                    f"[Wind+Rain] Foundation erosion risk: "
                    f"annual rainfall {rain_mm:.0f} mm with shallow foundation "
                    f"{self._raw('foundation_depth_m'):.1f} m"
                )
            elif total_lateral > lateral_cap_kn:
                failure_reason = (
                    f"[Wind+Rain] Lateral overload: "
                    f"wind {wind_load:.1f} kN + hydro {hydro_kn:.1f} kN = {total_lateral:.1f} kN "
                    f"> capacity {lateral_cap_kn:.1f} kN "
                    f"(IS 875 Part 3)"
                )
            else:
                passed = True

            extra_info = {"wind_load_kn": round(wind_load, 1),
                          "hydrostatic_kn": round(hydro_kn, 1),
                          "total_lateral_kn": round(total_lateral, 1),
                          "lateral_capacity_kn": round(lateral_cap_kn, 1),
                          "foundation_erosion_risk": foundation_erode}

        # ══════════════════════════════════════════════════════════════════
        # ACTION 3: Overload test (150 % of design load)
        # ══════════════════════════════════════════════════════════════════
        elif action == 3:
            overload_factor = 1.50
            applied_ol   = self._applied_gravity_load_kn(overload_factor=overload_factor)
            soil_ok, soil_msg = self._soil_pressure_check(overload_factor=overload_factor)

            if not soil_ok:
                failure_reason = f"[Overload 150%] {soil_msg}"
            elif applied_ol > eff_cap_kn:
                failure_reason = (
                    f"[Overload 150%] Column failure at 150% load: "
                    f"applied {applied_ol:.1f} kN > capacity {eff_cap_kn:.1f} kN "
                    f"(demand/capacity = {applied_ol / (eff_cap_kn + 1e-6):.2f})"
                )
            else:
                passed = True

            extra_info = {"overload_factor": overload_factor,
                          "applied_kn": round(applied_ol, 1),
                          "capacity_kn": round(eff_cap_kn, 1),
                          "demand_ratio": round(applied_ol / (eff_cap_kn + 1e-6), 3)}

        # ══════════════════════════════════════════════════════════════════
        # ACTION 4: Long-term degradation test
        # ══════════════════════════════════════════════════════════════════
        elif action == 4:
            age       = self._raw("building_age_years")
            raw_deg   = self._degradation_factor()   # capacity multiplier

            # Additional corrosion risk for high-rainfall / coastal areas
            rain_mm   = self._raw("rainfall_mm_annual")
            corrosion_penalty = 1.0
            if rain_mm > 2000:
                corrosion_penalty = 1.0 - (rain_mm - 2000) / 10000.0  # up to −10 %

            final_deg_factor  = raw_deg * corrosion_penalty
            degraded_cap_kn   = self._column_capacity_kn() * final_deg_factor
            applied           = self._applied_gravity_load_kn()

            if final_deg_factor <= MIN_CAPACITY_RATIO:
                failure_reason = (
                    f"[Degradation] Structure critically degraded: "
                    f"capacity factor {final_deg_factor:.2f} ≤ threshold {MIN_CAPACITY_RATIO} "
                    f"(age {age:.0f} y, corrosion penalty {corrosion_penalty:.3f})"
                )
            elif applied > degraded_cap_kn:
                failure_reason = (
                    f"[Degradation] Age-degraded column failure: "
                    f"applied {applied:.1f} kN > degraded capacity {degraded_cap_kn:.1f} kN "
                    f"(age factor {final_deg_factor:.2f})"
                )
            else:
                passed = True

            extra_info = {"age_years": age,
                          "degradation_factor": round(final_deg_factor, 3),
                          "corrosion_penalty": round(corrosion_penalty, 3),
                          "degraded_capacity_kn": round(degraded_cap_kn, 1),
                          "applied_kn": round(applied, 1)}

        # ══════════════════════════════════════════════════════════════════
        # 3. Compute reward
        # ══════════════════════════════════════════════════════════════════
        # Prevent the agent from "camping" on the safest action by severely 
        # penalizing it for repeating an action it did in the last 4 steps.
        recent_penalty = 2.0 if action in self._action_history[-4:] else 0.0
        self._action_history.append(action)

        rw = self._risk_weight()
        if passed:
            reward = 1.0 - recent_penalty
            damage_increment = 0.005   # slight wear per step even on pass
        else:
            reward = -1.0 * rw - recent_penalty
            damage_increment = 0.05 * rw   # bigger damage on failure

        self._cumulative_damage = min(self._cumulative_damage + damage_increment, 1.0)

        # ── 4. Degrade state slightly on failure ────────────────────────── #
        if not passed:
            idx_rein = PARAM_KEYS.index("column_reinforcement_ratio")
            self._state[idx_rein] = max(
                0.0,
                self._state[idx_rein] - 0.01  # gradual steel loss
            )
            idx_age = PARAM_KEYS.index("building_age_years")
            self._state[idx_age]  = min(
                1.0,
                self._state[idx_age] + 0.02   # accelerated ageing on failure
            )

        # ── 5. Risk score for this step ─────────────────────────────────── #
        risk_score = self._compute_risk_score()
        self._risk_trajectory.append(risk_score)

        # ── 6. Episode termination conditions ────────────────────────────── #
        terminated = (
            self._cumulative_damage >= 0.95
            or risk_score >= 95.0
        )
        truncated = self._step_count >= self.max_steps

        # ── 7. Build info dict ───────────────────────────────────────────── #
        info: Dict[str, Any] = {
            "step":               self._step_count,
            "action_name":        ACTION_NAMES[action],
            "passed":             passed,
            "failure_reason":     failure_reason,
            "risk_score":         round(risk_score, 2),
            "risk_weight":        round(rw, 3),
            "cumulative_damage":  round(self._cumulative_damage, 3),
            "degradation_factor": round(deg_factor, 3),
            **extra_info,
        }
        self._damage_log.append(info)

        return self._state.copy(), float(reward), terminated, truncated, info

    # ═══════════════════════════════════════════════════════════════════════ #
    #  Accessors                                                              #
    # ═══════════════════════════════════════════════════════════════════════ #
    def get_risk_trajectory(self) -> List[float]:
        return list(self._risk_trajectory)

    def get_damage_log(self) -> List[Dict[str, Any]]:
        return list(self._damage_log)

    def get_column_capacity(self) -> float:
        return self._column_capacity_kn()

    def get_state_dict(self) -> Dict[str, float]:
        """Return current normalised state as a labelled dict."""
        return {k: float(self._state[i]) for i, k in enumerate(PARAM_KEYS)}

    def get_raw_state_dict(self) -> Dict[str, float]:
        """Return current raw (un-normalised) state values."""
        return {k: round(self._denorm(k, float(self._state[i])), 4)
                for i, k in enumerate(PARAM_KEYS)}

    # ═══════════════════════════════════════════════════════════════════════ #
    #  Render                                                                 #
    # ═══════════════════════════════════════════════════════════════════════ #
    def render(self) -> None:
        risk  = self._risk_trajectory[-1] if self._risk_trajectory else 0.0
        cap   = self._column_capacity_kn()
        load  = self._applied_gravity_load_kn()
        print(
            f"Step {self._step_count:>3} | "
            f"Risk: {risk:5.1f}/100 | "
            f"Cap: {cap:7.1f} kN | "
            f"Load: {load:7.1f} kN | "
            f"D/C: {load/(cap+1e-6):.3f} | "
            f"CumDmg: {self._cumulative_damage:.3f}"
        )

    def close(self) -> None:
        pass
