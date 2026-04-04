"""
PPOStructuralAgent – Wraps Stable-Baselines3 PPO for the 18D BuildingStressEnv.
Trains a policy and evaluates risk trajectories per stress scenario.
"""
import os
import numpy as np
from typing import Dict, Any, List, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from environment.building_env import BuildingStressEnv, ACTION_NAMES


class _RiskTrajectoryCallback(BaseCallback):
    """Collects per-step risk scores and scenario pass/fail during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_risks: List[float] = []
        self.episode_rewards: List[float] = []
        self.scenario_pass_counts: Dict[str, int] = {n: 0 for n in ACTION_NAMES.values()}
        self.scenario_fail_counts: Dict[str, int] = {n: 0 for n in ACTION_NAMES.values()}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            risk = info.get("risk_score", info.get("current_risk", 0))
            self.episode_risks.append(risk)
            name = info.get("action_name", "")
            if info.get("passed") is True and name in self.scenario_pass_counts:
                self.scenario_pass_counts[name] += 1
            elif info.get("passed") is False and name in self.scenario_fail_counts:
                self.scenario_fail_counts[name] += 1
        return True


class PPOStructuralAgent:
    """PPO-based RL agent for structural stress scenario simulation."""

    MODEL_DIR = os.path.join("outputs", "models")

    def __init__(self, blueprint_params: Dict[str, Any]):
        self.blueprint_params   = blueprint_params
        self.model: Optional[PPO] = None
        self._train_risks:    List[float] = []
        self._eval_trajectory: List[float] = []
        self._training_metrics: Dict[str, Any] = {}
        self._callback: Optional[_RiskTrajectoryCallback] = None
        os.makedirs(self.MODEL_DIR, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #
    def train(self, timesteps: int = 5000) -> Dict[str, Any]:
        def _make_env():
            return BuildingStressEnv(self.blueprint_params, max_steps=20)

        vec_env        = make_vec_env(_make_env, n_envs=1)
        self._callback = _RiskTrajectoryCallback()

        self.model = PPO(
            "MlpPolicy", vec_env,
            verbose=0,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.95,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs={"net_arch": [64, 64]},
            seed=42,
        )
        self.model.learn(
            total_timesteps=timesteps,
            callback=self._callback,
            progress_bar=False,
        )
        self._train_risks = self._callback.episode_risks

        model_path = os.path.join(self.MODEL_DIR, "ppo_structural")
        self.model.save(model_path)

        risks = self._train_risks
        self._training_metrics = {
            "timesteps":           timesteps,
            "model_path":          model_path + ".zip",
            "mean_risk":           round(float(np.mean(risks)) if risks else 0, 2),
            "max_risk":            round(float(np.max(risks))  if risks else 0, 2),
            "min_risk":            round(float(np.min(risks))  if risks else 0, 2),
            "risk_samples":        len(risks),
            "scenario_pass_counts":self._callback.scenario_pass_counts,
            "scenario_fail_counts":self._callback.scenario_fail_counts,
        }
        return self._training_metrics

    # ------------------------------------------------------------------ #
    #  Evaluation — multi-episode
    # ------------------------------------------------------------------ #
    def evaluate(self, n_episodes: int = 10) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Call train() first.")

        env = BuildingStressEnv(self.blueprint_params, max_steps=20)
        episode_rewards:  List[float]       = []
        episode_risks:    List[List[float]] = []
        action_counts:    Dict[int, int]    = {i: 0 for i in range(5)}
        # Per-scenario detailed results (last episode)
        scenario_step_results: Dict[str, Dict] = {}

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep)
            done   = False
            total_r = 0.0
            ep_risks: List[float] = []

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                action_int = int(action)
                action_counts[action_int] += 1
                obs, reward, terminated, truncated, info = env.step(action_int)
                total_r += float(reward)
                ep_risks.append(info.get("risk_score", info.get("current_risk", 0)))

                # Collect last-episode pass/fail per scenario for app display
                if ep == n_episodes - 1:
                    aname = info.get("action_name", f"Action {action_int}")
                    scenario_step_results[aname] = {
                        "passed":         info.get("passed"),
                        "failure_reason": info.get("failure_reason", ""),
                        **{k: v for k, v in info.items()
                           if k not in ("passed", "failure_reason", "action_name",
                                        "step", "risk_score", "risk_weight",
                                        "cumulative_damage", "degradation_factor")}
                    }
                done = terminated or truncated

            episode_rewards.append(total_r)
            episode_risks.append(ep_risks)

        # Uniform-length trajectory
        max_len = max((len(r) for r in episode_risks), default=0)
        padded  = [r + [r[-1]] * (max_len - len(r)) for r in episode_risks if r]
        mean_traj = np.mean(padded, axis=0).tolist() if padded else []
        self._eval_trajectory = mean_traj

        action_dist = {ACTION_NAMES[k]: v for k, v in action_counts.items()}

        return {
            "n_episodes":          n_episodes,
            "mean_reward":         round(float(np.mean(episode_rewards)), 3),
            "std_reward":          round(float(np.std(episode_rewards)), 3),
            "episode_rewards":     [round(r, 3) for r in episode_rewards],
            "mean_trajectory":     [round(r, 2) for r in mean_traj],
            "action_distribution": action_dist,
            "final_mean_risk":     round(mean_traj[-1], 2) if mean_traj else 0,
            "scenario_step_results": scenario_step_results,
        }

    # ------------------------------------------------------------------ #
    #  Single-episode inference with step log
    # ------------------------------------------------------------------ #
    def run_single_episode(self) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Call train() first.")

        env = BuildingStressEnv(self.blueprint_params, max_steps=20)
        obs, _ = env.reset(seed=0)
        done = False
        steps: List[Dict] = []
        total_reward = 0.0

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            action_int = int(action)
            obs, reward, terminated, truncated, info = env.step(action_int)
            total_reward += float(reward)
            steps.append({
                "step":    info["step"],
                "action":  info["action_name"],
                "risk":    round(info.get("risk_score", info.get("current_risk", 0)), 2),
                "damage":  round(info["cumulative_damage"], 3),
                "passed":  info.get("passed"),
                "reward":  round(float(reward), 3),
                "failure_reason": info.get("failure_reason", ""),
            })
            done = terminated or truncated

        return {
            "steps":           steps,
            "total_reward":    round(total_reward, 3),
            "final_risk":      steps[-1]["risk"]   if steps else 0,
            "final_damage":    steps[-1]["damage"]  if steps else 0,
            "risk_trajectory": [s["risk"] for s in steps],
        }

    def get_training_risks(self)   -> List[float]:      return self._train_risks
    def get_eval_trajectory(self)  -> List[float]:      return self._eval_trajectory
    def get_training_metrics(self) -> Dict[str, Any]:   return self._training_metrics
