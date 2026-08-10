"""
dapi_module_corrected.py

Corrected Dynamic Adaptive Privacy Intensity (DAPI) module.
This version restores key suppression behavior from the earlier DAPI algorithm,
including trust shaping, hard low-trust cutoffs, capped aggregation, and
trust-aware noise scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
import math

import numpy as np
import torch


class PrivacyAdaptationStrategy(Enum):
    PERFORMANCE_BASED = "performance_based"
    ROUND_BASED = "round_based"
    CONVERGENCE_BASED = "convergence_based"
    HYBRID = "hybrid"


@dataclass
class DAPIConfig:
    initial_clip_norm: float = 1.0
    initial_noise_multiplier: float = 0.1
    target_epsilon: float = 2.0
    adaptation_window: int = 5
    adaptation_strategy: PrivacyAdaptationStrategy = PrivacyAdaptationStrategy.HYBRID
    delta: float = 1e-5
    min_clip_norm: float = 0.1
    max_clip_norm: float = 5.0
    min_noise_multiplier: float = 0.0
    max_noise_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if isinstance(self.adaptation_strategy, PrivacyAdaptationStrategy):
            self.adaptation_strategy = self.adaptation_strategy.value


class DifferentialPrivacyAdaptiveInterface:
    """Compatibility wrapper used by the shared FRL runner."""

    def __init__(self, config: DAPIConfig | None = None) -> None:
        self.config = config or DAPIConfig()
        self.clip_norm = float(self.config.initial_clip_norm)
        self.noise_multiplier = float(self.config.initial_noise_multiplier)
        self.round_idx = 0
        self.adaptation_history: List[Dict[str, float]] = []

    def apply_differential_privacy(
        self,
        delta_state: Dict[str, torch.Tensor],
        seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        flat, spec = self._flatten_state_dict(delta_state)
        if flat.numel() == 0:
            return delta_state

        norm = torch.linalg.norm(flat, ord=2).item()
        if norm > self.clip_norm and norm > 0:
            flat = flat * (self.clip_norm / norm)

        if self.noise_multiplier > 0:
            generator = None
            if seed is not None:
                generator = torch.Generator(device=flat.device)
                generator.manual_seed(seed)
            noise = torch.normal(
                mean=0.0,
                std=self.clip_norm * self.noise_multiplier,
                size=flat.shape,
                generator=generator,
                device=flat.device,
            )
            flat = flat + noise

        return self._reconstruct_state_dict(flat, spec)

    def adapt_privacy_parameters(
        self,
        round_metrics: Dict[str, float],
        client_deltas: List[Dict[str, torch.Tensor]],
        global_state: Dict[str, torch.Tensor],
    ) -> Tuple[float, float]:
        self.round_idx += 1
        accuracy = float(round_metrics.get("accuracy", 0.0))
        loss = float(round_metrics.get("loss", 0.0))

        strategy = self.config.adaptation_strategy
        if isinstance(strategy, PrivacyAdaptationStrategy):
            strategy_name = strategy.value
        else:
            strategy_name = str(strategy)

        if strategy_name in {"performance_based", "hybrid"}:
            if accuracy < 0.70:
                self.noise_multiplier *= 0.90
                self.clip_norm *= 1.05
            elif accuracy > 0.90:
                self.noise_multiplier *= 1.03
                self.clip_norm *= 0.98

        if strategy_name in {"round_based", "hybrid"} and self.round_idx % max(1, self.config.adaptation_window) == 0:
            self.noise_multiplier *= 0.98

        self.clip_norm = float(np.clip(self.clip_norm, self.config.min_clip_norm, self.config.max_clip_norm))
        self.noise_multiplier = float(
            np.clip(
                self.noise_multiplier,
                self.config.min_noise_multiplier,
                self.config.max_noise_multiplier,
            )
        )

        self.adaptation_history.append({
            "round": float(self.round_idx),
            "accuracy": accuracy,
            "loss": loss,
            "clip_norm": self.clip_norm,
            "noise_multiplier": self.noise_multiplier,
            "epsilon": self.get_privacy_budget()["epsilon"],
        })
        return self.clip_norm, self.noise_multiplier

    def get_privacy_budget(self) -> Dict[str, float]:
        epsilon = self.config.target_epsilon / max(self.noise_multiplier, 1e-8)
        return {"epsilon": float(epsilon), "delta": float(self.config.delta)}

    def get_adaptation_history(self) -> List[Dict[str, float]]:
        return self.adaptation_history

    @staticmethod
    def _flatten_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[Tuple[str, torch.Size, int]]]:
        flat_parts = []
        spec = []
        for name, tensor in state_dict.items():
            part = tensor.detach().cpu().reshape(-1)
            flat_parts.append(part)
            spec.append((name, tensor.shape, part.numel()))
        if not flat_parts:
            return torch.tensor([]), spec
        return torch.cat(flat_parts), spec

    @staticmethod
    def _reconstruct_state_dict(
        flat_vec: torch.Tensor,
        spec: List[Tuple[str, torch.Size, int]],
    ) -> Dict[str, torch.Tensor]:
        out = {}
        offset = 0
        for name, shape, numel in spec:
            out[name] = flat_vec[offset: offset + numel].reshape(shape).clone()
            offset += numel
        return out


@dataclass
class TrustConfig:
    alpha_stability: float = 0.18
    alpha_quality: float = 0.20
    alpha_similarity: float = 0.18
    alpha_history: float = 0.10
    alpha_f1_score: float = 0.16
    alpha_gradient_dev: float = 0.18
    # Retained as diagnostic compatibility fields; they are not trust factors.
    alpha_behavior_consistency: float = 0.0
    alpha_reward_consistency: float = 0.0
    stability_eta: float = 10.0
    similarity_kappa: float = 10.0
    smoothing: float = 0.65
    trust_sensitivity: float = 3.0
    low_trust_threshold: float = 1e-8
    trust_weight_power: float = 4.0
    trust_temperature: float = 1.0
    weight_normalization: str = "power"
    warmup_rounds: int = 0
    min_weight_eligibility_threshold: float = 0.0
    max_agg_weight: float = 0.25
    min_agg_weight: float = 0.0

    def validate(self) -> None:
        weights = [
            self.alpha_stability,
            self.alpha_quality,
            self.alpha_similarity,
            self.alpha_history,
            self.alpha_f1_score,
            self.alpha_gradient_dev,
            self.alpha_behavior_consistency,
            self.alpha_reward_consistency,
        ]
        if any(w < 0 for w in weights):
            raise ValueError("Trust weights must be non-negative.")
        total = sum(weights)
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"Trust weights must sum to 1.0, got {total:.6f}.")
        if not (0.0 <= self.smoothing < 1.0):
            raise ValueError("smoothing must be in [0, 1).")
        if self.trust_weight_power < 1.0:
            raise ValueError("trust_weight_power must be >= 1.0.")
        if self.trust_temperature <= 0.0:
            raise ValueError("trust_temperature must be positive.")
        if self.weight_normalization not in {"power", "softmax"}:
            raise ValueError("weight_normalization must be 'power' or 'softmax'.")
        if self.warmup_rounds < 0:
            raise ValueError("warmup_rounds must be non-negative.")
        if not (0.0 <= self.min_weight_eligibility_threshold <= 1.0):
            raise ValueError("min_weight_eligibility_threshold must be in [0, 1].")
        if not (0.0 <= self.low_trust_threshold <= 1.0):
            raise ValueError("low_trust_threshold must be in [0, 1].")
        if not (0.0 <= self.max_agg_weight <= 1.0):
            raise ValueError("max_agg_weight must be in [0, 1].")
        if not (0.0 <= self.min_agg_weight <= 1.0):
            raise ValueError("min_agg_weight must be in [0, 1].")
        if self.min_agg_weight > self.max_agg_weight:
            raise ValueError("min_agg_weight must be <= max_agg_weight.")


@dataclass
class PrivacyConfig:
    epsilon_min: float = 0.3
    epsilon_max: float = 4.0
    clip_norm: float = 1.0
    delta: float = 1e-5
    tiny: float = 1e-8
    sigma_min: float = 0.02
    sigma_max: float = 0.12

    def validate(self) -> None:
        if self.epsilon_min <= 0 or self.epsilon_max <= 0:
            raise ValueError("epsilon_min and epsilon_max must be positive.")
        if self.epsilon_min > self.epsilon_max:
            raise ValueError("epsilon_min must be <= epsilon_max.")
        if self.clip_norm <= 0:
            raise ValueError("clip_norm must be positive.")
        if self.delta <= 0:
            raise ValueError("delta must be positive.")
        if self.sigma_min < 0 or self.sigma_max < 0:
            raise ValueError("sigma_min and sigma_max must be non-negative.")
        if self.sigma_min > self.sigma_max:
            raise ValueError("sigma_min must be <= sigma_max.")


@dataclass
class ClientTrustState:
    prev_update: Optional[np.ndarray] = None
    smoothed_trust: float = 0.5
    rounds_seen: int = 0
    reward_history: List[float] = None

    def __post_init__(self) -> None:
        if self.reward_history is None:
            self.reward_history = []


def twa_bound_update(raw_update: np.ndarray, norm_bound: float) -> np.ndarray:
    """Return the TWA robustness-bounded update, independent of DP/privacy."""
    norm = float(np.linalg.norm(raw_update))
    scale = min(1.0, float(norm_bound) / max(norm, 1e-12))
    return (raw_update * scale).astype(np.float32, copy=False)


class DAPIController:
    """
    Dynamic Adaptive Privacy Intensity controller.
    """

    def __init__(
        self,
        trust_cfg: Optional[TrustConfig] = None,
        privacy_cfg: Optional[PrivacyConfig] = None,
    ) -> None:
        self.trust_cfg = trust_cfg or TrustConfig()
        self.privacy_cfg = privacy_cfg or PrivacyConfig()
        self.trust_cfg.validate()
        self.privacy_cfg.validate()
        self.client_states: Dict[int, ClientTrustState] = {}

    @staticmethod
    def flatten_state_dict(state_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for _, tensor in state_dict.items():
            arr = tensor.detach().cpu().numpy().astype(np.float32).ravel()
            vectors.append(arr)
        if not vectors:
            return np.array([], dtype=np.float32)
        return np.concatenate(vectors, axis=0)

    @staticmethod
    def unflatten_to_state_dict(
        flat_vector: np.ndarray,
        reference_state_dict: Dict[str, torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        restored: Dict[str, torch.Tensor] = {}
        idx = 0
        for key, tensor in reference_state_dict.items():
            numel = tensor.numel()
            chunk = flat_vector[idx: idx + numel]
            reshaped = chunk.reshape(tensor.shape)
            restored[key] = torch.tensor(
                reshaped,
                dtype=tensor.dtype,
                device=device if device is not None else tensor.device,
            )
            idx += numel
        if idx != len(flat_vector):
            raise ValueError("Flat vector length does not match reference state_dict structure.")
        return restored

    @staticmethod
    def normalize_to_unit_interval(value: float, low: float, high: float) -> float:
        if high <= low:
            return 0.5
        out = (value - low) / (high - low)
        return float(np.clip(out, 0.0, 1.0))

    def get_client_state(self, client_id: int) -> ClientTrustState:
        if client_id not in self.client_states:
            self.client_states[client_id] = ClientTrustState()
        return self.client_states[client_id]

    def compute_reward_stability(self, client_id: int, current_reward: float) -> float:
        state = self.get_client_state(client_id)
        state.reward_history.append(float(current_reward))
        window = np.asarray(state.reward_history[-5:], dtype=np.float32)
        if len(window) < 2:
            return 0.7
        mean_abs = float(np.mean(np.abs(window))) + self.privacy_cfg.tiny
        cv = float(np.std(window) / mean_abs)
        return float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))

    def compute_similarity_score(
        self,
        current_update: np.ndarray,
        mean_update: np.ndarray,
    ) -> float:
        if len(current_update) == 0 or len(mean_update) == 0:
            return 1.0
        numerator = float(np.dot(current_update, mean_update))
        denom = float(np.linalg.norm(current_update) * np.linalg.norm(mean_update)) + self.privacy_cfg.tiny
        similarity = numerator / denom
        return float(np.clip((similarity + 1.0) / 2.0, 0.0, 1.0))

    def compute_history_score(self, client_id: int) -> float:
        state = self.get_client_state(client_id)
        return float(np.clip(state.smoothed_trust, 0.0, 1.0))

    def compute_gradient_deviation_score(
        self,
        gradient_norm: float,
        median_grad_norm: float,
    ) -> float:
        if median_grad_norm <= self.privacy_cfg.tiny:
            return 0.5
        dev = abs(gradient_norm - median_grad_norm) / (median_grad_norm + self.privacy_cfg.tiny)
        return float(np.clip(1.0 - dev, 0.0, 1.0))

    def compute_trust(
        self,
        client_id: int,
        current_update: np.ndarray,
        mean_update: np.ndarray,
        local_quality_metric: float,
        current_reward: float,
        f1_score: float = 0.5,
        f1_min: float = 0.0,
        f1_max: float = 1.0,
        gradient_norm: float = 0.0,
        median_grad_norm: float = 0.0,
        metric_min: float = 0.0,
        metric_max: float = 1.0,
        behavior_consistency: float = 0.5,
        reward_consistency: float = 0.5,
    ) -> Dict[str, float]:
        cfg = self.trust_cfg

        stability = self.compute_reward_stability(client_id, current_reward)
        quality = self.normalize_to_unit_interval(local_quality_metric, metric_min, metric_max)
        similarity = self.compute_similarity_score(current_update, mean_update)
        history = self.compute_history_score(client_id)
        f1_norm = self.normalize_to_unit_interval(f1_score, f1_min, f1_max)
        grad_dev = self.compute_gradient_deviation_score(gradient_norm, median_grad_norm)
        behavior = float(np.clip(behavior_consistency, 0.0, 1.0))
        reward_check = float(np.clip(reward_consistency, 0.0, 1.0))

        raw_trust = (
            cfg.alpha_stability * stability
            + cfg.alpha_quality * quality
            + cfg.alpha_similarity * similarity
            + cfg.alpha_history * history
            + cfg.alpha_f1_score * f1_norm
            + cfg.alpha_gradient_dev * grad_dev
            + cfg.alpha_behavior_consistency * behavior
            + cfg.alpha_reward_consistency * reward_check
        )
        raw_trust = float(np.clip(raw_trust, 0.0, 1.0))

        state = self.get_client_state(client_id)
        if state.rounds_seen == 0:
            smoothed = raw_trust
        else:
            smoothed = cfg.smoothing * state.smoothed_trust + (1.0 - cfg.smoothing) * raw_trust
        smoothed = float(np.clip(smoothed, 0.0, 1.0))

        state.smoothed_trust = smoothed
        state.prev_update = current_update.copy()
        state.rounds_seen += 1

        return {
            "stability": stability,
            "quality": quality,
            "similarity": similarity,
            "history": history,
            "f1_score": f1_norm,
            "gradient_deviation": grad_dev,
            "behavior_consistency": behavior,
            "reward_consistency": reward_check,
            "raw_trust": raw_trust,
            "smoothed_trust": smoothed,
        }

    def trust_to_epsilon_sharp(self, trust_score: float) -> float:
        cfg = self.privacy_cfg
        kappa = self.trust_cfg.trust_sensitivity
        trust_score = float(np.clip(trust_score, 0.0, 1.0))
        epsilon = cfg.epsilon_min + (cfg.epsilon_max - cfg.epsilon_min) * (trust_score ** kappa)
        return float(np.clip(epsilon, cfg.epsilon_min, cfg.epsilon_max))

    def trust_to_noise_scale(self, trust_score: float) -> float:
        cfg = self.privacy_cfg
        trust_score = float(np.clip(trust_score, 0.0, 1.0))
        sigma = cfg.sigma_max - (cfg.sigma_max - cfg.sigma_min) * (trust_score ** self.trust_cfg.trust_sensitivity)
        return float(np.clip(sigma, cfg.sigma_min, cfg.sigma_max))

    def clip_update(self, flat_update: np.ndarray) -> np.ndarray:
        """Compatibility wrapper; TWA norm bounding is not a privacy operation."""
        return twa_bound_update(flat_update, self.privacy_cfg.clip_norm)

    def add_gaussian_noise(self, flat_update: np.ndarray, noise_scale: float) -> np.ndarray:
        noise = np.random.normal(
            loc=0.0,
            scale=noise_scale,
            size=flat_update.shape,
        ).astype(np.float32)
        return flat_update + noise

    def privatize_update(
        self,
        client_id: int,
        local_state_dict: Dict[str, torch.Tensor],
        mean_update: np.ndarray,
        local_quality_metric: float,
        current_reward: float,
        f1_score: float = 0.5,
        f1_min: float = 0.0,
        f1_max: float = 1.0,
        gradient_norm: float = 0.0,
        median_grad_norm: float = 0.0,
        metric_min: float = 0.0,
        metric_max: float = 1.0,
        behavior_consistency: float = 0.5,
        reward_consistency: float = 0.5,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        flat_update = self.flatten_state_dict(local_state_dict)
        clipped_update = self.clip_update(flat_update)

        trust_info = self.compute_trust(
            client_id=client_id,
            current_update=clipped_update,
            mean_update=mean_update,
            local_quality_metric=local_quality_metric,
            current_reward=current_reward,
            f1_score=f1_score,
            f1_min=f1_min,
            f1_max=f1_max,
            gradient_norm=gradient_norm,
            median_grad_norm=median_grad_norm,
            metric_min=metric_min,
            metric_max=metric_max,
            behavior_consistency=behavior_consistency,
            reward_consistency=reward_consistency,
        )

        trust_score = trust_info["smoothed_trust"]
        epsilon = self.trust_to_epsilon_sharp(trust_score)
        noise_scale = self.trust_to_noise_scale(trust_score)
        privatized = self.add_gaussian_noise(clipped_update, noise_scale)

        info = {
            **trust_info,
            "epsilon": float(epsilon),
            "noise_scale": float(noise_scale),
            "update_norm_before_clip": float(np.linalg.norm(flat_update)),
            "update_norm_after_clip": float(np.linalg.norm(clipped_update)),
        }
        return privatized, info

    def compute_round_trusts(
        self,
        client_ids: List[int],
        clipped_updates: Dict[int, np.ndarray],
        local_quality_metrics: Dict[int, float],
        current_rewards: Dict[int, float],
        f1_scores: Dict[int, float],
        gradient_norms: Dict[int, float],
        behavior_scores: Optional[Dict[int, float]] = None,
        reward_consistency_scores: Optional[Dict[int, float]] = None,
        reference_update: Optional[np.ndarray] = None,
        reward_min: float = -600.0,
        reward_max: float = 900.0,
    ) -> Dict[int, Dict[str, float]]:
        cfg = self.trust_cfg
        behavior_scores = behavior_scores or {}
        reward_consistency_scores = reward_consistency_scores or {}

        mean_update = (
            reference_update
            if reference_update is not None
            else np.mean(np.stack([clipped_updates[cid] for cid in client_ids], axis=0), axis=0)
        )
        median_grad_norm = float(np.median(np.asarray([gradient_norms.get(cid, 0.0) for cid in client_ids], dtype=np.float32)))

        stability_raw: Dict[int, float] = {}
        quality_raw: Dict[int, float] = {}
        similarity_raw: Dict[int, float] = {}
        grad_dev_raw: Dict[int, float] = {}
        history_raw: Dict[int, float] = {}
        f1_raw: Dict[int, float] = {}
        behavior_raw: Dict[int, float] = {}
        reward_consistency_raw: Dict[int, float] = {}

        for cid in client_ids:
            stability_raw[cid] = self.compute_reward_stability(cid, current_rewards.get(cid, 0.0))
            quality_raw[cid] = self.normalize_to_unit_interval(
                local_quality_metrics.get(cid, 0.0), reward_min, reward_max
            )
            similarity_raw[cid] = self.compute_similarity_score(clipped_updates[cid], mean_update)
            grad_dev_raw[cid] = abs(gradient_norms.get(cid, 0.0) - median_grad_norm) / (median_grad_norm + self.privacy_cfg.tiny)
            history_raw[cid] = self.get_client_state(cid).smoothed_trust
            f1_raw[cid] = f1_scores.get(cid, 0.5)
            behavior_raw[cid] = float(np.clip(behavior_scores.get(cid, 0.5), 0.0, 1.0))
            reward_consistency_raw[cid] = float(np.clip(reward_consistency_scores.get(cid, 0.5), 0.0, 1.0))

        stability = stability_raw
        quality = quality_raw
        similarity = similarity_raw
        history = history_raw
        f1_norm = {cid: float(np.clip(value, 0.0, 1.0)) for cid, value in f1_raw.items()}
        grad_dev = {cid: 1.0 / (1.0 + value) for cid, value in grad_dev_raw.items()}
        behavior = behavior_raw
        reward_consistency = reward_consistency_raw

        trust_logs: Dict[int, Dict[str, float]] = {}
        for cid in client_ids:
            raw_score = (
                cfg.alpha_stability * stability[cid]
                + cfg.alpha_quality * quality[cid]
                + cfg.alpha_similarity * similarity[cid]
                + cfg.alpha_history * history[cid]
                + cfg.alpha_f1_score * f1_norm[cid]
                + cfg.alpha_gradient_dev * grad_dev[cid]
            )
            consistency_gate = 1.0
            raw_score = float(np.clip(raw_score, 0.0, 1.0))

            state = self.get_client_state(cid)
            if state.rounds_seen == 0:
                smoothed = raw_score
            else:
                smoothed = cfg.smoothing * state.smoothed_trust + (1.0 - cfg.smoothing) * raw_score
            smoothed = float(np.clip(smoothed, 0.0, 1.0))

            state.smoothed_trust = smoothed
            state.prev_update = clipped_updates[cid].copy()
            state.rounds_seen += 1

            trust_logs[cid] = {
                "stability": stability[cid],
                "quality": quality[cid],
                "similarity": similarity[cid],
                "history": history[cid],
                "f1_score": f1_norm[cid],
                "gradient_deviation": grad_dev[cid],
                "behavior_consistency": behavior[cid],
                "reward_consistency": reward_consistency[cid],
                "consistency_gate": consistency_gate,
                "raw_score": raw_score,
                "smoothed_trust": smoothed,
                "median_grad_norm": median_grad_norm,
                "client_grad_norm": gradient_norms.get(cid, 0.0),
            }

        return trust_logs

    def privatize_updates_round(
        self,
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        client_sizes: Dict[int, int],
        local_quality_metrics: Dict[int, float],
        current_rewards: Dict[int, float],
        f1_scores: Optional[Dict[int, float]] = None,
        gradient_norms: Optional[Dict[int, float]] = None,
        behavior_scores: Optional[Dict[int, float]] = None,
        reward_consistency_scores: Optional[Dict[int, float]] = None,
        metric_min: float = 0.0,
        metric_max: float = 1.0,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[str, float]]]:
        client_ids = sorted(client_updates.keys())
        f1_scores = f1_scores or {}
        gradient_norms = gradient_norms or {}

        clipped_updates: Dict[int, np.ndarray] = {}
        for cid in client_ids:
            flat = self.flatten_state_dict(client_updates[cid])
            clipped_updates[cid] = self.clip_update(flat)

        trust_logs = self.compute_round_trusts(
            client_ids=client_ids,
            clipped_updates=clipped_updates,
            local_quality_metrics=local_quality_metrics,
            current_rewards=current_rewards,
            f1_scores=f1_scores,
            gradient_norms=gradient_norms,
            behavior_scores=behavior_scores,
            reward_consistency_scores=reward_consistency_scores,
        )

        privatized: Dict[int, np.ndarray] = {}
        for cid in client_ids:
            trust = trust_logs[cid]["smoothed_trust"]
            epsilon = self.trust_to_epsilon_sharp(trust)
            noise_scale = self.trust_to_noise_scale(trust)
            privatized[cid] = self.add_gaussian_noise(clipped_updates[cid], noise_scale)
            trust_logs[cid]["epsilon"] = float(epsilon)
            trust_logs[cid]["noise_scale"] = float(noise_scale)
            trust_logs[cid]["update_norm_after_clip"] = float(np.linalg.norm(clipped_updates[cid]))

        return privatized, trust_logs


def _cosine_similarity(raw_update: np.ndarray, mean_update: np.ndarray) -> float:
    if len(raw_update) == 0 or len(mean_update) == 0:
        return 1.0
    denom = np.linalg.norm(raw_update) * np.linalg.norm(mean_update) + 1e-8
    return float(np.dot(raw_update, mean_update) / denom)


def _minmax_normalize(values: Dict[int, float], invert: bool = False) -> Dict[int, float]:
    if not values:
        return {}
    raw_arr = np.asarray(list(values.values()), dtype=np.float32)
    low = float(np.min(raw_arr))
    high = float(np.max(raw_arr))
    if high - low < 1e-8:
        return {cid: 0.5 for cid in values}
    normalized = {}
    for cid, value in values.items():
        scaled = (float(value) - low) / (high - low)
        if invert:
            scaled = 1.0 - scaled
        normalized[cid] = float(np.clip(scaled, 0.0, 1.0))
    return normalized


def trust_weighted_fedavg(
    client_updates: List[np.ndarray],
    client_sizes: List[int],
    client_trusts: List[float],
    low_trust_threshold: float = 0.0,
    trust_weight_power: float = 1.0,
    max_agg_weight: float = 1.0,
    min_agg_weight: float = 0.0,
    trust_temperature: float = 1.0,
    weight_normalization: str = "power",
    min_weight_eligibility_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if not client_updates:
        raise ValueError("client_updates cannot be empty.")
    if not (len(client_updates) == len(client_sizes) == len(client_trusts)):
        raise ValueError("client_updates, client_sizes, and client_trusts must have the same length.")

    sizes = np.asarray(client_sizes, dtype=np.float32)
    trusts = np.asarray(client_trusts, dtype=np.float32)
    trusts = np.clip(trusts, 0.0, 1.0)

    active = trusts >= low_trust_threshold
    trusts = np.where(active, trusts, 0.0)
    temperature = max(float(trust_temperature), 1e-8)
    if weight_normalization == "softmax":
        logits = trusts / temperature
        if np.any(active):
            logits = logits - float(np.max(logits[active]))
        sharpened_trusts = np.where(active, np.exp(logits), 0.0)
    elif weight_normalization == "power":
        effective_power = max(float(trust_weight_power), 1.0) / temperature
        sharpened_trusts = np.power(trusts, effective_power)
    else:
        raise ValueError(f"Unknown weight_normalization: {weight_normalization}")
    raw_weights = sizes * sharpened_trusts

    if raw_weights.sum() <= 0:
        raw_weights = sizes / (sizes.sum() + 1e-8)
    else:
        raw_weights = raw_weights / (raw_weights.sum() + 1e-8)

    if min_agg_weight > 0:
        floor_eligible = active & (trusts >= min_weight_eligibility_threshold)
        if np.any(floor_eligible):
            raw_weights[floor_eligible] = np.maximum(
                raw_weights[floor_eligible], min_agg_weight
            )
            raw_weights = raw_weights / (raw_weights.sum() + 1e-8)

    if max_agg_weight < 1.0:
        for _ in range(8):
            over = raw_weights > max_agg_weight
            if not np.any(over):
                break
            excess = float(np.sum(raw_weights[over] - max_agg_weight))
            raw_weights[over] = max_agg_weight
            under = ~over & (raw_weights > 0)
            if np.any(under) and excess > 0:
                raw_weights[under] += excess * (raw_weights[under] / (raw_weights[under].sum() + 1e-8))
            raw_weights = raw_weights / (raw_weights.sum() + 1e-8)

    stacked = np.stack(client_updates, axis=0).astype(np.float32)
    global_update = np.sum(stacked * raw_weights[:, None], axis=0)
    return global_update, raw_weights


def apply_flat_update_to_model(
    model: torch.nn.Module,
    flat_update: np.ndarray,
    reference_state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.nn.Module:
    ref_state = reference_state_dict or model.state_dict()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = None
    new_state = DAPIController.unflatten_to_state_dict(
        flat_vector=flat_update,
        reference_state_dict=ref_state,
        device=device,
    )
    model.load_state_dict(new_state)
    return model


def run_dapi_round(
    round_idx: int,
    client_models: Dict[int, torch.nn.Module],
    client_sizes: Dict[int, int],
    local_quality_metrics: Dict[int, float],
    dapi: DAPIController,
    metric_min: float = 0.0,
    metric_max: float = 1.0,
    current_rewards: Optional[Dict[int, float]] = None,
    f1_scores: Optional[Dict[int, float]] = None,
    gradient_norms: Optional[Dict[int, float]] = None,
    behavior_scores: Optional[Dict[int, float]] = None,
    reward_consistency_scores: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    if not client_models:
        raise ValueError("client_models cannot be empty.")

    client_ids = sorted(client_models.keys())
    client_sizes_list = [client_sizes[cid] for cid in client_ids]
    current_rewards = current_rewards or local_quality_metrics
    f1_scores = f1_scores or {}
    gradient_norms = gradient_norms or {}
    behavior_scores = behavior_scores or {}
    reward_consistency_scores = reward_consistency_scores or {}

    processed_deltas, logs = dapi.privatize_updates_round(
        client_updates={cid: client_models[cid].state_dict() for cid in client_ids},
        client_sizes={cid: client_sizes[cid] for cid in client_ids},
        local_quality_metrics=local_quality_metrics,
        current_rewards=current_rewards,
        f1_scores=f1_scores,
        gradient_norms=gradient_norms,
        behavior_scores=behavior_scores,
        reward_consistency_scores=reward_consistency_scores,
        metric_min=metric_min,
        metric_max=metric_max,
    )

    privatized_updates = [processed_deltas[cid] for cid in client_ids]
    trusts = [logs[cid]["smoothed_trust"] for cid in client_ids]
    epsilons = [logs[cid]["epsilon"] for cid in client_ids]
    agg_weights = trust_weighted_fedavg(
        client_updates=privatized_updates,
        client_sizes=client_sizes_list,
        client_trusts=trusts,
        low_trust_threshold=dapi.trust_cfg.low_trust_threshold,
        trust_weight_power=dapi.trust_cfg.trust_weight_power,
        max_agg_weight=dapi.trust_cfg.max_agg_weight,
        min_agg_weight=dapi.trust_cfg.min_agg_weight,
        trust_temperature=dapi.trust_cfg.trust_temperature,
        weight_normalization=dapi.trust_cfg.weight_normalization,
        min_weight_eligibility_threshold=dapi.trust_cfg.min_weight_eligibility_threshold,
    )[1]

    global_update = np.sum(
        np.stack(privatized_updates, axis=0).astype(np.float32) * np.asarray(agg_weights, dtype=np.float32)[:, None],
        axis=0,
    )

    return {
        "round": round_idx,
        "global_update": global_update,
        "client_trusts": dict(zip(client_ids, trusts)),
        "client_epsilons": dict(zip(client_ids, epsilons)),
        "aggregation_weights": dict(zip(client_ids, np.asarray(agg_weights, dtype=np.float32).tolist())),
        "logs": logs,
    }


if __name__ == "__main__":
    class TinyNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(10, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    # Example usage
    dapi = DAPIController()
    model = TinyNet()
    dummy_update = np.random.randn(11).astype(np.float32)
    updated_model = apply_flat_update_to_model(model, dummy_update)
    print("DAPI module test passed.")
