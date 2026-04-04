"""
risk/risk_module.py – StructSense AI Risk Engine
=================================================
Two components:

1. StructuralRiskScorer
   - Accepts a raw 18-dim parameter dict OR normalised state vector
   - Weighted-sum risk score (0–1) across 6 critical parameters
   - Per-element breakdown: {foundation, columns, beams, slab}
   - Red-zone flagging for elements with risk > 0.70

2. NeuralDSW  (Dynamic Safety Weight – PyTorch)
   - 3-layer MLP: 18 → 64 → 32 → 1  (Sigmoid output)
   - Produces a safety weight w ∈ (0, 1)
   - Adaptive reward: structural_efficiency − (w × failure_penalty)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── import env metadata for normalization bounds ──────────────────────────── #
from environment.building_env import PARAM_KEYS, PARAM_BOUNDS


# ══════════════════════════════════════════════════════════════════════════════
#  1. StructuralRiskScorer
# ══════════════════════════════════════════════════════════════════════════════

class StructuralRiskScorer:
    """
    Rule-based structural risk scorer operating on the 18-dimensional state.

    Weighted-sum overall risk (0–1):
        seismic_zone                  0.25
        soil_bearing_capacity_kn_m2   0.20  (inverted: low SBC = high risk)
        column_reinforcement_ratio     0.20  (inverted: low rho = high risk)
        foundation_depth_m            0.15  (inverted: shallow = high risk)
        building_age_years            0.10
        shear_wall_present            0.10  (inverted: absent = high risk)

    Per-element breakdown keys:
        foundation, columns, beams, slab

    Red zones: elements with element_risk > 0.70
    """

    # Weighted parameters (positive weight = higher raw value → more risk)
    # Tuple: (param_key, weight, invert)
    # invert=True  → risk ∝ (1 − normalised_value)   i.e. low value is dangerous
    # invert=False → risk ∝ normalised_value           i.e. high value is dangerous
    WEIGHTED_PARAMS: List[Tuple[str, float, bool]] = [
        ("seismic_zone",                 0.25, False),   # zone 5 = worst
        ("soil_bearing_capacity_kn_m2",  0.20, True),    # low SBC = bad
        ("column_reinforcement_ratio",   0.20, True),    # low rho = bad
        ("foundation_depth_m",           0.15, True),    # shallow = bad
        ("building_age_years",           0.10, False),   # old = bad
        ("shear_wall_present",           0.10, True),    # absent = bad
    ]

    RED_ZONE_THRESHOLD = 0.70

    def __init__(self) -> None:
        assert abs(sum(w for _, w, _ in self.WEIGHTED_PARAMS) - 1.0) < 1e-6, \
            "Weights must sum to 1.0"

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalise(key: str, value: float) -> float:
        """Normalize raw value → [0, 1] using PARAM_BOUNDS."""
        lo, hi = PARAM_BOUNDS[key]
        return float(np.clip((value - lo) / (hi - lo + 1e-9), 0.0, 1.0))

    def _get_norm(
        self,
        key: str,
        state_vec: Optional[np.ndarray],
        raw_params: Optional[Dict],
    ) -> float:
        """Retrieve normalised value from either a state vector or raw dict."""
        if state_vec is not None:
            idx = PARAM_KEYS.index(key)
            return float(np.clip(state_vec[idx], 0.0, 1.0))
        if raw_params is not None:
            return self._normalise(key, float(raw_params.get(key, 0.0)))
        raise ValueError("Provide either state_vec or raw_params")

    # ------------------------------------------------------------------ #
    #  Overall Risk Score (0–1)                                            #
    # ------------------------------------------------------------------ #
    def score(
        self,
        state_vec: Optional[np.ndarray] = None,
        raw_params: Optional[Dict] = None,
    ) -> float:
        """
        Compute overall structural risk score ∈ [0, 1].

        Args:
            state_vec : 18-dim normalised numpy array from BuildingStressEnv
            raw_params: dict with raw parameter values (alternative to state_vec)

        Returns:
            risk ∈ [0.0, 1.0]  — higher = more dangerous
        """
        total = 0.0
        for key, weight, invert in self.WEIGHTED_PARAMS:
            norm_val = self._get_norm(key, state_vec, raw_params)
            contribution = (1.0 - norm_val) if invert else norm_val
            total += weight * contribution
        return float(np.clip(total, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    #  Weighted-component breakdown                                        #
    # ------------------------------------------------------------------ #
    def component_contributions(
        self,
        state_vec: Optional[np.ndarray] = None,
        raw_params: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Return each weighted contribution (un-normalised to weight space).
        Keys: seismic_zone, soil_bearing, reinforcement, foundation,
              building_age, shear_wall
        """
        out = {}
        for key, weight, invert in self.WEIGHTED_PARAMS:
            norm_val = self._get_norm(key, state_vec, raw_params)
            contribution = (1.0 - norm_val) if invert else norm_val
            short = {
                "seismic_zone":                "seismic_zone",
                "soil_bearing_capacity_kn_m2": "soil_bearing",
                "column_reinforcement_ratio":  "reinforcement",
                "foundation_depth_m":          "foundation_depth",
                "building_age_years":          "building_age",
                "shear_wall_present":          "shear_wall",
            }.get(key, key)
            out[short] = round(weight * contribution, 4)
        return out

    # ------------------------------------------------------------------ #
    #  Per-element breakdown: foundation, columns, beams, slab             #
    # ------------------------------------------------------------------ #
    def element_breakdown(
        self,
        state_vec: Optional[np.ndarray] = None,
        raw_params: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Return per-structural-element risk scores ∈ [0, 1].

        Foundation risk  ← soil SBC (inv), foundation depth (inv), foundation type
        Column risk      ← reinforcement ratio (inv), concrete grade (inv),
                           column dimensions, seismic zone
        Beam risk        ← beam depth (inv), concrete grade (inv), floor load
        Slab risk        ← slab thickness (inv), floor load, num floors
        """
        def gn(k: str) -> float:
            return self._get_norm(k, state_vec, raw_params)

        # ── Foundation ─────────────────────────────────────────────────── #
        soil_risk  = 1.0 - gn("soil_bearing_capacity_kn_m2")   # low SBC = bad
        depth_risk = 1.0 - gn("foundation_depth_m")            # shallow = bad
        ftype_norm = gn("foundation_type")                       # 0=isolated worst
        ftype_risk = 1.0 - ftype_norm / 2.0                     # pile=0, isolated=1
        shear_risk = 1.0 - gn("shear_wall_present")

        foundation_risk = (
            0.35 * soil_risk +
            0.30 * depth_risk +
            0.20 * ftype_risk +
            0.15 * shear_risk
        )

        # ── Columns ────────────────────────────────────────────────────── #
        rein_risk  = 1.0 - gn("column_reinforcement_ratio")    # low rho = bad
        conc_risk  = 1.0 - gn("concrete_grade")                # low fck = bad
        # narrow column = higher risk (inverted)
        col_dim    = (gn("column_width_mm") + gn("column_depth_mm")) / 2.0
        col_risk   = 1.0 - col_dim
        seis_risk  = gn("seismic_zone")

        column_risk = (
            0.35 * rein_risk +
            0.25 * conc_risk +
            0.20 * col_risk +
            0.20 * seis_risk
        )

        # ── Beams ──────────────────────────────────────────────────────── #
        beam_risk_depth = 1.0 - gn("beam_depth_mm")            # shallow beam = bad
        fload_risk      = gn("floor_load_kn_m2")               # heavy load = bad
        rein_beam_risk  = 1.0 - gn("column_reinforcement_ratio")  # proxy

        beam_risk = (
            0.40 * beam_risk_depth +
            0.35 * fload_risk +
            0.25 * rein_beam_risk
        )

        # ── Slab ───────────────────────────────────────────────────────── #
        slab_thin_risk = 1.0 - gn("slab_thickness_mm")         # thin slab = bad
        floor_risk     = gn("floor_load_kn_m2")
        floors_risk    = gn("num_floors")                       # many floors = more risk

        slab_risk = (
            0.40 * slab_thin_risk +
            0.35 * floor_risk +
            0.25 * floors_risk
        )

        return {
            "foundation": round(float(np.clip(foundation_risk, 0, 1)), 4),
            "columns":    round(float(np.clip(column_risk,     0, 1)), 4),
            "beams":      round(float(np.clip(beam_risk,       0, 1)), 4),
            "slab":       round(float(np.clip(slab_risk,       0, 1)), 4),
        }

    # ------------------------------------------------------------------ #
    #  Red-zone flagging                                                   #
    # ------------------------------------------------------------------ #
    def red_zones(
        self,
        state_vec: Optional[np.ndarray] = None,
        raw_params: Optional[Dict] = None,
    ) -> List[str]:
        """
        Return list of element names whose risk exceeds RED_ZONE_THRESHOLD (0.70).
        Empty list means no critical elements.
        """
        breakdown = self.element_breakdown(state_vec=state_vec, raw_params=raw_params)
        return [elem for elem, risk in breakdown.items()
                if risk > self.RED_ZONE_THRESHOLD]

    # ------------------------------------------------------------------ #
    #  Full report                                                         #
    # ------------------------------------------------------------------ #
    def full_report(
        self,
        state_vec: Optional[np.ndarray] = None,
        raw_params: Optional[Dict] = None,
    ) -> Dict:
        """
        Convenience: returns overall score, element breakdown, weighted
        contributions, and red-zone list in a single call.
        """
        overall   = self.score(state_vec=state_vec, raw_params=raw_params)
        elements  = self.element_breakdown(state_vec=state_vec, raw_params=raw_params)
        contribs  = self.component_contributions(state_vec=state_vec, raw_params=raw_params)
        zones     = self.red_zones(state_vec=state_vec, raw_params=raw_params)

        return {
            "overall_risk":         round(overall, 4),
            "risk_label":           self._label(overall),
            "element_breakdown":    elements,
            "weighted_contributions": contribs,
            "red_zones":            zones,
            "any_critical":         len(zones) > 0,
        }

    @staticmethod
    def _label(score: float) -> str:
        if score < 0.30: return "LOW"
        if score < 0.55: return "MODERATE"
        if score < 0.75: return "HIGH"
        return "CRITICAL"


# ══════════════════════════════════════════════════════════════════════════════
#  2. NeuralDSW – Dynamic Safety Weight (PyTorch MLP)
# ══════════════════════════════════════════════════════════════════════════════

class NeuralDSW(nn.Module):
    """
    Neural Dynamic Safety Weight network.

    Architecture:   Linear(18→64) → ReLU → Dropout(0.2)
                    Linear(64→32) → ReLU → Dropout(0.1)
                    Linear(32→1)  → Sigmoid
    Output:         safety_weight w ∈ (0, 1)

    Adaptive reward formula:
        R = structural_efficiency − (w × failure_penalty)

    Semantics:
        w → 0: model trusts the structure is safe → failure penalty is discounted
        w → 1: model sees high risk → failure penalty is amplified

    Training:
        Supervised on (state, true_risk_label) pairs; uses BCELoss.
        Alternatively trained end-to-end with RL rewards via policy gradient.
    """

    def __init__(
        self,
        input_dim: int = 18,
        hidden1: int = 64,
        hidden2: int = 32,
        dropout1: float = 0.20,
        dropout2: float = 0.10,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(p=dropout1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(p=dropout2),
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------ #
    #  Forward pass                                                        #
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: float tensor of shape (batch, 18) – normalised state vectors

        Returns:
            safety_weight: float tensor of shape (batch, 1) ∈ (0, 1)
        """
        return self.net(x)

    # ------------------------------------------------------------------ #
    #  Convenience: single-state inference (no grad)                      #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_weight(
        self,
        state_vec: Union[np.ndarray, torch.Tensor],
    ) -> float:
        """
        Return scalar safety weight for a single 18-dim state.

        Args:
            state_vec: numpy array (18,) or torch Tensor (18,)

        Returns:
            w ∈ (0, 1)  — higher = higher perceived risk
        """
        if isinstance(state_vec, np.ndarray):
            x = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
        else:
            x = state_vec.float().unsqueeze(0) if state_vec.dim() == 1 else state_vec.float()
        self.eval()
        return float(self.net(x).squeeze())

    # ------------------------------------------------------------------ #
    #  Adaptive reward                                                     #
    # ------------------------------------------------------------------ #
    def adaptive_reward(
        self,
        state_vec: Union[np.ndarray, torch.Tensor],
        structural_efficiency: float,
        failure_penalty: float,
        passed: bool = True,
    ) -> Dict[str, float]:
        """
        Compute adaptive reward for an RL transition.

            R = structural_efficiency − (w × failure_penalty)

        If the structure passed the stress test, failure_penalty is 0.
        If it failed, failure_penalty is amplified by w.

        Args:
            state_vec              : 18-dim normalised state
            structural_efficiency  : base reward signal (typically in [0, 1])
            failure_penalty        : raw penalty magnitude (positive scalar)
            passed                 : whether the scenario was passed

        Returns:
            dict with keys: safety_weight, reward, components
        """
        w = self.predict_weight(state_vec)

        if passed:
            effective_penalty = 0.0
        else:
            effective_penalty = w * failure_penalty

        reward = structural_efficiency - effective_penalty

        return {
            "safety_weight":        round(w, 4),
            "structural_efficiency": round(structural_efficiency, 4),
            "failure_penalty_raw":   round(failure_penalty, 4),
            "effective_penalty":     round(effective_penalty, 4),
            "reward":                round(reward, 4),
            "passed":                passed,
        }

    # ------------------------------------------------------------------ #
    #  Supervised training step                                           #
    # ------------------------------------------------------------------ #
    def train_step(
        self,
        state_batch: torch.Tensor,
        risk_labels: torch.Tensor,
        optimizer: optim.Optimizer,
    ) -> float:
        """
        Single supervised training step using BCELoss.

        Args:
            state_batch : (N, 18) float tensor
            risk_labels : (N, 1)  float tensor  ∈ {0, 1}  (1 = high-risk)
            optimizer   : any torch optimizer

        Returns:
            loss value (float)
        """
        self.train()
        optimizer.zero_grad()
        preds = self.net(state_batch)
        loss  = nn.functional.binary_cross_entropy(preds, risk_labels)
        loss.backward()
        optimizer.step()
        return float(loss.item())

    # ------------------------------------------------------------------ #
    #  Quick supervised fit                                               #
    # ------------------------------------------------------------------ #
    def fit(
        self,
        states: np.ndarray,
        labels: np.ndarray,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> List[float]:
        """
        Train NeuralDSW on (state, risk_label) pairs.

        Args:
            states    : (N, 18) float array – normalised state vectors
            labels    : (N,)    float array – 1=risky, 0=safe
            epochs    : training epochs
            lr        : learning rate
            batch_size: mini-batch size
            verbose   : print loss every 10 epochs

        Returns:
            loss history (list of floats)
        """
        optimizer  = optim.Adam(self.parameters(), lr=lr)
        scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        X = torch.tensor(states, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

        N = X.shape[0]
        loss_history: List[float] = []

        for epoch in range(epochs):
            perm   = torch.randperm(N)
            X, y   = X[perm], y[perm]
            ep_loss = 0.0
            batches = 0

            for i in range(0, N, batch_size):
                xb = X[i: i + batch_size]
                yb = y[i: i + batch_size]
                ep_loss += self.train_step(xb, yb, optimizer)
                batches += 1

            scheduler.step()
            avg_loss = ep_loss / max(batches, 1)
            loss_history.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:>3}/{epochs} | loss: {avg_loss:.4f}")

        self.eval()
        return loss_history

    # ------------------------------------------------------------------ #
    #  Generate synthetic training data from StructuralRiskScorer        #
    # ------------------------------------------------------------------ #
    @staticmethod
    def generate_training_data(
        n_samples: int = 2000,
        risk_threshold: float = 0.55,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic (state, label) pairs for supervised pre-training.
        Labels: 1 if StructuralRiskScorer.score() ≥ risk_threshold else 0.

        Args:
            n_samples      : number of random states to generate
            risk_threshold : boundary between safe (0) and risky (1)
            seed           : random seed

        Returns:
            (states, labels)  shapes: (N, 18), (N,)
        """
        rng    = np.random.default_rng(seed)
        scorer = StructuralRiskScorer()

        states = rng.uniform(0.0, 1.0, size=(n_samples, 18)).astype(np.float32)
        labels = np.array([
            1.0 if scorer.score(state_vec=states[i]) >= risk_threshold else 0.0
            for i in range(n_samples)
        ], dtype=np.float32)

        return states, labels

    # ------------------------------------------------------------------ #
    #  Save / load                                                        #
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: str = "cpu") -> None:
        self.load_state_dict(torch.load(path, map_location=map_location))
        self.eval()


# ══════════════════════════════════════════════════════════════════════════════
#  Module-level convenience factories
# ══════════════════════════════════════════════════════════════════════════════

def make_pretrained_dsw(
    n_samples: int = 2000,
    epochs: int = 50,
    verbose: bool = False,
) -> NeuralDSW:
    """
    Instantiate and pre-train a NeuralDSW on synthetic risk data.
    Ready to use as a drop-in safety-weight provider.
    """
    dsw    = NeuralDSW()
    states, labels = NeuralDSW.generate_training_data(n_samples=n_samples)
    dsw.fit(states, labels, epochs=epochs, verbose=verbose)
    return dsw
