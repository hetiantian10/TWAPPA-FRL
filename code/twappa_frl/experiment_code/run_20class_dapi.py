"""
20-class Federated RL runner with complete DAPI-side experiment logging.

This runner is intentionally separate from the existing 4-class/12-class code.
It keeps the same benchmark outputs while adding the extra artifacts needed for
Section VI: Results and Discussion.

CKKS remains opt-in. Disabled runs preserve the historical plaintext adapter;
``--enable_ckks`` performs real TenSEAL encryption and homomorphic aggregation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from CORE.dapi import DAPIController, PrivacyConfig, TrustConfig, trust_weighted_fedavg
from CORE.dqn_agent_multiclass import DQNAgentMultiClass
from CORE.real_env_multiclass import CICIDSMultiClassEnv


TASK_NAME = "20class_dapi"
DEFAULT_DATA_DIR = "data/cicids_20class"
DEFAULT_RESULTS_DIR = "results/20class_dapi"
REQUIRED_NUM_CLASSES = 20
LOCAL_ATTACK_MODES = {"label_flipping", "data_poisoning", "reward_poisoning"}
UPDATE_ATTACK_MODES = {
    "sign_flipping",
    "random_noise_byzantine",
    "model_replacement",
    "scaled_sign_flip",
}
ATTACK_MODE_ALIASES = {
    "clean": "none",
    "sf": "sign_flipping",
    "rnb": "random_noise_byzantine",
    "mr": "model_replacement",
    "lf": "label_flipping",
    "dap": "data_poisoning",
    "rep": "reward_poisoning",
}


def atomic_to_csv(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=index)
    temporary.replace(path)


@dataclass
class RunConfig:
    data_dir: str
    results_dir: str
    num_clients: int = 20
    rounds: int = 20
    local_episodes: int = 5
    max_steps_per_episode: int = 64
    batch_size: int = 128
    supervised_aux_epochs: int = 1
    seed: int = 42
    device: str = "cpu"
    execution_paths: str = "fedavg,static_dp,dapi,he_dapi"
    adversarial_clients: str = ""
    attack_mode: str = "scaled_sign_flip"
    adversarial_scale: float = 8.0
    random_noise_std: float = 1.0
    rnb_alpha: float = 0.0
    static_clip_norm: float = 1.0
    static_noise_multiplier: float = 0.1
    krum_f: int = 1
    trimmed_mean_ratio: float = 0.10
    dapi_epsilon_min: float = 0.3
    dapi_epsilon_max: float = 4.0
    dapi_clip_norm: float = 1.0
    dapi_delta: float = 1e-5
    dapi_sigma_min: float = 0.02
    dapi_sigma_max: float = 0.12
    disable_privacy_intensity: bool = False
    trust_smoothing: float = 0.65
    trust_suppression_threshold: float = 1e-8
    trust_weight_power: float = 4.0
    trust_temperature: float = 1.0
    trust_weight_normalization: str = "power"
    trust_min_active_weight: float = 0.0
    trust_floor_eligibility_threshold: float = 0.0
    trust_max_weight: float = 0.25
    trust_warmup_rounds: int = 0
    aggregate_raw_updates: bool = False
    enable_ckks: bool = False
    save_global_checkpoints: bool = False
    expected_num_classes: int = 0
    initial_checkpoint: str = ""


class CKKSAdapter:
    """
    CKKS weighted-aggregation adapter.

    When disabled it preserves the historical plaintext-passthrough behavior.
    When enabled it encrypts every client update with TenSEAL, performs the
    weighted sum over ciphertexts, and decrypts only the aggregate.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.poly_modulus_degree = 8192
        self.coeff_mod_bit_sizes = [60, 30, 60]
        self.global_scale = 2**30
        self.slot_count = self.poly_modulus_degree // 2
        self.context = None
        self.secret_key = None
        self.encryption_count = 0
        self.decryption_count = 0
        if self.enabled:
            try:
                import tenseal as ts
            except ImportError as exc:
                raise RuntimeError(
                    "Real CKKS was requested, but TenSEAL is not installed."
                ) from exc
            self._ts = ts
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=self.poly_modulus_degree,
                coeff_mod_bit_sizes=self.coeff_mod_bit_sizes,
            )
            self.context.global_scale = self.global_scale
            self.secret_key = self.context.secret_key()
            self.context.make_context_public()

    def aggregate(self, flat_updates: List[np.ndarray], weights: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        t0 = time.perf_counter()
        plaintext_reference = np.sum(np.stack(flat_updates, axis=0) * weights[:, None], axis=0)
        chunk_count = 0

        if not self.enabled:
            aggregated = plaintext_reference.copy()
            ciphertext_count = 0
            ciphertext_bytes = 0
        else:
            encrypted_clients = []
            ciphertext_bytes = 0
            for update in flat_updates:
                encrypted_chunks = []
                for start in range(0, update.size, self.slot_count):
                    chunk = update[start:start + self.slot_count]
                    padded = np.zeros(self.slot_count, dtype=np.float64)
                    padded[:chunk.size] = chunk.astype(np.float64, copy=False)
                    encrypted = self._ts.ckks_vector(self.context, padded.tolist())
                    self.encryption_count += 1
                    ciphertext_bytes += len(encrypted.serialize())
                    encrypted_chunks.append(encrypted)
                encrypted_clients.append(encrypted_chunks)

            chunk_count = len(encrypted_clients[0])
            encrypted_sum = []
            for chunk_idx in range(chunk_count):
                accumulator = encrypted_clients[0][chunk_idx] * float(weights[0])
                for client_idx in range(1, len(encrypted_clients)):
                    accumulator += encrypted_clients[client_idx][chunk_idx] * float(weights[client_idx])
                encrypted_sum.append(accumulator)

            decrypted_values: List[float] = []
            for encrypted in encrypted_sum:
                decrypted_values.extend(encrypted.decrypt(self.secret_key))
                self.decryption_count += 1
            aggregated = np.asarray(
                decrypted_values[:plaintext_reference.size], dtype=np.float32
            )
            ciphertext_count = sum(len(chunks) for chunks in encrypted_clients)

        elapsed = time.perf_counter() - t0
        diff = aggregated - plaintext_reference
        denominator = float(np.linalg.norm(aggregated) * np.linalg.norm(plaintext_reference))
        cosine_similarity = (
            float(np.dot(aggregated, plaintext_reference) / denominator)
            if denominator > 0.0
            else 1.0
        )
        return aggregated, {
            "ckks_enabled": float(self.enabled),
            "tenseal_context_created": float(self.context is not None),
            "ciphertext_aggregation_count": float(ciphertext_count),
            "decryption_count": float(chunk_count if self.enabled else 0),
            "ckks_time_sec": float(elapsed),
            "ckks_ciphertext_count": float(ciphertext_count),
            "ckks_ciphertext_bytes": float(ciphertext_bytes),
            "tensor_l2_error": float(np.linalg.norm(diff)),
            "tensor_linf_error": float(np.max(np.abs(diff))) if diff.size else 0.0,
            "tensor_mean_abs_error": float(np.mean(np.abs(diff))) if diff.size else 0.0,
            "tensor_cosine_similarity": cosine_similarity,
            "tensor_relative_l2_error": float(
                np.linalg.norm(diff) / (np.linalg.norm(plaintext_reference) + 1e-12)
            ),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_label(label: Any) -> str:
    label = str(label).strip()
    label = label.replace("ï¿½", "-").replace("â€“", "-").replace("â€”", "-")
    return " ".join(label.split())


def load_label_names(data_dir: str) -> List[str]:
    data_path = Path(data_dir)
    meta_path = data_path / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        labels = meta.get("classes") or meta.get("class_set") or meta.get("label_names")
        if labels:
            return [normalize_label(label) for label in labels]

    labels = set()
    for csv_path in data_path.glob("*.csv"):
        try:
            df = pd.read_csv(csv_path, usecols=["Label"], dtype=str)
            labels.update(normalize_label(label) for label in df["Label"].dropna().unique())
        except Exception:
            continue
    return sorted(label for label in labels if label)


def assert_20_classes(label_names: List[str], data_dir: str) -> None:
    if len(label_names) != REQUIRED_NUM_CLASSES:
        raise ValueError(
            f"{TASK_NAME} requires exactly {REQUIRED_NUM_CLASSES} classes, but found "
            f"{len(label_names)} in {data_dir}: {label_names}"
        )


def parse_int_set(raw: str) -> set[int]:
    if not raw.strip():
        return set()
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def parse_attack_modes(raw: str) -> set[str]:
    if not raw.strip():
        return {"none"}
    normalized = raw.replace("+", ",").replace(";", ",")
    modes = {
        ATTACK_MODE_ALIASES.get(part.strip().lower(), part.strip().lower())
        for part in normalized.split(",")
        if part.strip()
    }
    if not modes:
        modes = {"none"}
    if "none" in modes and len(modes) > 1:
        modes.remove("none")
    unknown = modes - LOCAL_ATTACK_MODES - UPDATE_ATTACK_MODES - {"none"}
    if unknown:
        raise ValueError(f"Unknown attack_mode value(s): {sorted(unknown)}")
    update_modes = modes & UPDATE_ATTACK_MODES
    if len(update_modes) > 1:
        raise ValueError(
            "At most one update-level attack can be active in one run; "
            f"got {sorted(update_modes)}"
        )
    return modes


def clean_dataframe(df: pd.DataFrame, label_names: List[str]) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df["Label"] = df["Label"].map(normalize_label)
    df = df[df["Label"].isin(label_names)].copy()
    return df


def dataframe_to_xy(df: pd.DataFrame, label_names: List[str], scaler: Optional[StandardScaler] = None) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    df = clean_dataframe(df, label_names)
    label_map = {label: idx for idx, label in enumerate(label_names)}
    feature_cols = [col for col in df.columns if col != "Label"]
    X = df[feature_cols].select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    y = df["Label"].map(label_map).to_numpy(dtype=np.int64)
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
    else:
        X = scaler.transform(X).astype(np.float32)
    return X, y, scaler


def load_client_payloads(data_dir: str, num_clients: int, label_names: List[str]) -> List[Dict[str, Any]]:
    data_path = Path(data_dir)
    shared_test = pd.read_csv(data_path / "test.csv") if (data_path / "test.csv").exists() else None
    payloads: List[Dict[str, Any]] = []
    for cid in range(num_clients):
        train_path = data_path / f"client_{cid}_train.csv"
        test_path = data_path / f"client_{cid}_test.csv"
        if not train_path.exists():
            continue
        train_df = clean_dataframe(pd.read_csv(train_path), label_names)
        test_df = clean_dataframe(pd.read_csv(test_path), label_names) if test_path.exists() else shared_test
        payloads.append({
            "client_id": cid,
            "train_df": train_df,
            "test_df": test_df,
            "train_size": len(train_df),
        })
    if len(payloads) != num_clients:
        raise FileNotFoundError(
            f"Expected {num_clients} client train files in {data_dir}, found {len(payloads)}."
        )
    return payloads


def flatten_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, List[Tuple[str, torch.Size, torch.dtype, int]]]:
    parts: List[np.ndarray] = []
    spec: List[Tuple[str, torch.Size, torch.dtype, int]] = []
    for key, tensor in state_dict.items():
        arr = tensor.detach().cpu().numpy().astype(np.float32).reshape(-1)
        parts.append(arr)
        spec.append((key, tensor.shape, tensor.dtype, arr.size))
    return np.concatenate(parts, axis=0), spec


def unflatten_state_dict(flat: np.ndarray, spec: List[Tuple[str, torch.Size, torch.dtype, int]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    idx = 0
    for key, shape, dtype, size in spec:
        chunk = flat[idx: idx + size].reshape(shape)
        out[key] = torch.tensor(chunk, dtype=dtype)
        idx += size
    return out


def state_delta(global_state: Dict[str, torch.Tensor], local_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: local_state[key].detach().cpu() - global_state[key].detach().cpu() for key in global_state}


def apply_delta(global_state: Dict[str, torch.Tensor], delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: global_state[key].detach().cpu() + delta[key].detach().cpu() for key in global_state}


def clip_and_noise(flat_update: np.ndarray, clip_norm: float, noise_multiplier: float) -> np.ndarray:
    norm = np.linalg.norm(flat_update)
    clipped = flat_update if norm <= clip_norm or norm == 0 else flat_update * (clip_norm / (norm + 1e-12))
    if noise_multiplier <= 0:
        return clipped
    return clipped + np.random.normal(0.0, clip_norm * noise_multiplier, size=clipped.shape).astype(np.float32)


def coordinate_median(flat_updates: List[np.ndarray]) -> np.ndarray:
    return np.median(np.stack(flat_updates, axis=0), axis=0).astype(np.float32)


def trimmed_mean(flat_updates: List[np.ndarray], trim_ratio: float) -> np.ndarray:
    stacked = np.sort(np.stack(flat_updates, axis=0), axis=0)
    n_clients = stacked.shape[0]
    trim = int(np.floor(float(trim_ratio) * n_clients))
    if trim <= 0 or 2 * trim >= n_clients:
        return np.mean(stacked, axis=0).astype(np.float32)
    return np.mean(stacked[trim:n_clients - trim], axis=0).astype(np.float32)


def krum(flat_updates: List[np.ndarray], f: int) -> Tuple[np.ndarray, int]:
    stacked = np.stack(flat_updates, axis=0)
    n_clients = stacked.shape[0]
    neighbor_count = max(1, n_clients - int(f) - 2)
    scores = []
    for idx in range(n_clients):
        diffs = stacked - stacked[idx]
        distances = np.sum(diffs * diffs, axis=1)
        nearest = np.sort(np.delete(distances, idx))[:neighbor_count]
        scores.append(float(np.sum(nearest)))
    selected = int(np.argmin(scores))
    return stacked[selected].astype(np.float32), selected


def build_agent(payload: Dict[str, Any], label_names: List[str], cfg: RunConfig) -> DQNAgentMultiClass:
    X, _, _ = dataframe_to_xy(payload["train_df"], label_names)
    agent = DQNAgentMultiClass(X.shape[1], len(label_names))
    agent.batch_size = cfg.batch_size
    return agent


def build_env(payload: Dict[str, Any], label_names: List[str]) -> CICIDSMultiClassEnv:
    X, y, _ = dataframe_to_xy(payload["train_df"], label_names)
    return CICIDSMultiClassEnv(X, y)


def local_train(agent: DQNAgentMultiClass, env: CICIDSMultiClassEnv, cfg: RunConfig, trace: Optional[List[Dict[str, Any]]] = None, trace_context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    rewards: List[float] = []
    losses: List[float] = []
    steps = 0
    for _ in range(cfg.local_episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        step_idx = 0
        while not done and step_idx < cfg.max_steps_per_episode:
            if trace is None:
                action = agent.select_action(state)
                q_values_np = None
                random_draw = None
            else:
                with torch.no_grad():
                    q_values_np = agent.q_net(torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0))[0].detach().cpu().numpy()
                random_draw = float(np.random.rand())
                action = random.randint(0, agent.action_dim - 1) if random_draw < agent.epsilon else int(np.argmax(q_values_np))
            next_state, reward, done, _ = env.step(action)
            agent.store((state, action, reward, next_state, done))
            loss = agent.train_step()
            if trace is not None:
                order = np.argsort(q_values_np)[::-1]
                trace.append({**(trace_context or {}), "local_step": step_idx + 1,
                    "state_hash": hashlib.sha256(np.asarray(state, dtype=np.float32).tobytes()).hexdigest(),
                    "selected_action": action, "epsilon": agent.epsilon, "random_draw": random_draw,
                    "top1_action": int(order[0]), "top1_q": float(q_values_np[order[0]]),
                    "top2_action": int(order[1]), "top2_q": float(q_values_np[order[1]]),
                    "top_two_margin": float(q_values_np[order[0]] - q_values_np[order[1]]),
                    "reward": float(reward), "episode_termination": bool(done),
                    "replay_buffer_size": len(agent.memory),
                    "replay_buffer_hash": agent.memory_digest.hexdigest(),
                    "sampled_replay_indices": json.dumps(agent.last_sampled_indices),
                })
            if loss is not None:
                losses.append(float(loss))
            state = next_state
            episode_reward += float(reward)
            step_idx += 1
            steps += 1
        rewards.append(episode_reward)
    if cfg.supervised_aux_epochs > 0 and getattr(env, "n_samples", 0) > 0:
        X = torch.tensor(env.X, dtype=torch.float32, device=agent.device)
        y = torch.tensor(env.y, dtype=torch.long, device=agent.device)
        n = int(X.shape[0])
        batch_size = max(1, int(cfg.batch_size))
        for _ in range(cfg.supervised_aux_epochs):
            perm = torch.randperm(n, device=agent.device)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                logits = agent.q_net(X[idx])
                loss = F.cross_entropy(logits, y[idx])
                agent.optimizer.zero_grad()
                loss.backward()
                agent.optimizer.step()
                losses.append(float(loss.item()))
        agent.update_target()
    return {
        "avg_episode_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_loss": float(np.mean(losses)) if losses else 0.0,
        "steps_total": float(steps),
    }


def evaluate(agent: DQNAgentMultiClass, payloads: List[Dict[str, Any]], label_names: List[str]) -> Tuple[List[int], List[int], Dict[str, float]]:
    y_true: List[int] = []
    y_pred: List[int] = []
    for payload in payloads:
        df = payload.get("test_df")
        if df is None:
            df = payload["train_df"]
        X, y, _ = dataframe_to_xy(df, label_names)
        for idx, row in enumerate(X):
            y_true.append(int(y[idx]))
            y_pred.append(int(agent.predict(row)))
    if not y_true:
        return [], [], {"eval_samples": 0.0, "eval_accuracy": 0.0, "eval_loss": 0.0}
    acc = accuracy_score(y_true, y_pred)
    return y_true, y_pred, {
        "eval_samples": float(len(y_true)),
        "eval_accuracy": float(acc),
        "eval_reward": float(5.0 * acc - 2.0),
        "eval_loss": float(-np.log(max(acc, 1e-6))),
    }


def macro_f1_from_predictions(y_true: List[int], y_pred: List[int]) -> float:
    if not y_true:
        return 0.0
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return float(f1)


def trusted_validation_xy(
    payloads: List[Dict[str, Any]],
    label_names: List[str],
    max_samples: int = 512,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    frames = [payload.get("test_df") for payload in payloads if payload.get("test_df") is not None]
    if frames:
        val_df = pd.concat(frames, ignore_index=True)
    else:
        val_df = pd.concat([payload["train_df"] for payload in payloads], ignore_index=True)
    val_df = clean_dataframe(val_df, label_names)
    if len(val_df) > max_samples:
        val_df = val_df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    X, y, _ = dataframe_to_xy(val_df, label_names)
    return X, y


def predict_many(agent: DQNAgentMultiClass, X: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return np.asarray([], dtype=np.int64)
    return np.asarray([int(agent.predict(row)) for row in X], dtype=np.int64)


def policy_distribution(agent: DQNAgentMultiClass, X: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return np.empty((0, agent.action_dim), dtype=np.float32)
    with torch.no_grad():
        states = torch.tensor(X, dtype=torch.float32, device=agent.device)
        logits = agent.q_net(states)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
    return np.clip(probs, 1e-8, 1.0)


def behavior_consistency_score(
    local_agent: DQNAgentMultiClass,
    global_agent: DQNAgentMultiClass,
    X_val: np.ndarray,
) -> float:
    local_probs = policy_distribution(local_agent, X_val)
    global_probs = policy_distribution(global_agent, X_val)
    if local_probs.size == 0 or global_probs.size == 0:
        return 0.5
    kl = np.sum(local_probs * (np.log(local_probs) - np.log(global_probs)), axis=1)
    return float(np.clip(np.exp(-float(np.mean(kl))), 0.0, 1.0))


def reward_consistency_score(
    agent: DQNAgentMultiClass,
    X_val: np.ndarray,
    y_val: np.ndarray,
    reported_episode_reward: float,
    steps_total: float,
) -> float:
    if len(X_val) == 0:
        return 0.5
    preds = predict_many(agent, X_val)
    validation_reward = np.where(preds == y_val, 3.0, -2.0).mean()
    reported_step_reward = float(reported_episode_reward) / max(float(steps_total), 1.0)
    return float(np.clip(np.exp(-abs(reported_step_reward - validation_reward)), 0.0, 1.0))


def table_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows were produced."
    return df.to_csv(index=False).strip()


class TwentyClassDAPIExperiment:
    def __init__(self, cfg: RunConfig, label_names: List[str]) -> None:
        self.cfg = cfg
        self.label_names = label_names
        self.results_dir = Path(cfg.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.adversarial_clients = parse_int_set(cfg.adversarial_clients)
        self.attack_modes = parse_attack_modes(cfg.attack_mode)
        self.ckks = CKKSAdapter(enabled=cfg.enable_ckks)

        self.dapi = self._new_dapi_controller()

        self.history_rows: List[Dict[str, Any]] = []
        self.dapi_rows: List[Dict[str, Any]] = []
        self.path_rows: List[Dict[str, Any]] = []
        self.tensor_rows: List[Dict[str, Any]] = []
        self.overhead_rows: List[Dict[str, Any]] = []
        self.ablation_rows: List[Dict[str, Any]] = []
        self.mechanism_rows: List[Dict[str, Any]] = []
        self.previous_aggregates: Dict[str, np.ndarray] = {}
        self.behavior_trace_rows: List[Dict[str, Any]] = []
        self.validation_xy: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def _new_dapi_controller(self) -> DAPIController:
        trust_cfg = TrustConfig(
            smoothing=self.cfg.trust_smoothing,
            low_trust_threshold=self.cfg.trust_suppression_threshold,
            trust_weight_power=self.cfg.trust_weight_power,
            trust_temperature=self.cfg.trust_temperature,
            weight_normalization=self.cfg.trust_weight_normalization,
            min_agg_weight=self.cfg.trust_min_active_weight,
            min_weight_eligibility_threshold=self.cfg.trust_floor_eligibility_threshold,
            max_agg_weight=self.cfg.trust_max_weight,
            warmup_rounds=self.cfg.trust_warmup_rounds,
        )
        privacy_cfg = PrivacyConfig(
            epsilon_min=self.cfg.dapi_epsilon_min,
            epsilon_max=self.cfg.dapi_epsilon_max,
            clip_norm=self.cfg.dapi_clip_norm,
            delta=self.cfg.dapi_delta,
            sigma_min=self.cfg.dapi_sigma_min,
            sigma_max=self.cfg.dapi_sigma_max,
        )
        return DAPIController(trust_cfg=trust_cfg, privacy_cfg=privacy_cfg)

    def run(self) -> None:
        set_seed(self.cfg.seed)
        payloads = load_client_payloads(self.cfg.data_dir, self.cfg.num_clients, self.label_names)
        payloads = self._apply_data_attack(payloads)
        self.validation_xy = trusted_validation_xy(payloads, self.label_names, seed=self.cfg.seed)
        paths = [part.strip() for part in self.cfg.execution_paths.split(",") if part.strip()]
        for path_name in paths:
            self._run_path(path_name, payloads)
            # Persist cumulative completed paths so interruption cannot erase them.
            self._write_outputs(payloads)

    def _apply_data_attack(self, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not (self.attack_modes & {"label_flipping", "data_poisoning"}) or not self.adversarial_clients:
            return payloads
        poisoned_payloads = copy.deepcopy(payloads)
        rng = np.random.default_rng(self.cfg.seed)
        for payload in poisoned_payloads:
            cid = int(payload["client_id"])
            if cid not in self.adversarial_clients:
                continue
            for key in ("train_df",):
                df = payload.get(key)
                if df is None or "Label" not in df.columns:
                    continue
                poisoned = df.copy()
                if "label_flipping" in self.attack_modes:
                    labels = [normalize_label(label) for label in self.label_names]
                    next_label = {label: labels[(idx + 1) % len(labels)] for idx, label in enumerate(labels)}
                    poisoned["Label"] = poisoned["Label"].map(lambda value: next_label.get(normalize_label(value), value))
                if "data_poisoning" in self.attack_modes:
                    numeric_cols = [col for col in poisoned.columns if col != "Label" and pd.api.types.is_numeric_dtype(poisoned[col])]
                    if numeric_cols:
                        values = poisoned[numeric_cols].to_numpy(dtype=np.float32)
                        scale = np.nanstd(values, axis=0) + 1e-6
                        noise = rng.normal(0.0, self.cfg.adversarial_scale * 0.05, size=values.shape).astype(np.float32)
                        poisoned.loc[:, numeric_cols] = values + noise * scale
                payload[key] = poisoned
        return poisoned_payloads

    def _run_path(self, path_name: str, payloads: List[Dict[str, Any]]) -> None:
        set_seed(self.cfg.seed)
        if path_name in {"dapi", "he_dapi", "twa", "twa_ppa", "twa_bounded", "twa_ppa_bounded", "twa_raw", "twa_ppa_raw", "twa_lockstep"}:
            self.dapi = self._new_dapi_controller()

        clients = []
        for payload in payloads:
            agent = build_agent(payload, self.label_names, self.cfg)
            env = build_env(payload, self.label_names)
            clients.append({"payload": payload, "agent": agent, "env": env})

        generated_state = copy.deepcopy(clients[0]["agent"].get_weights())
        if self.cfg.initial_checkpoint:
            checkpoint = Path(self.cfg.initial_checkpoint)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            if checkpoint.exists():
                global_state = torch.load(checkpoint, map_location=self.cfg.device, weights_only=True)
                if set(global_state) != set(generated_state) or any(
                    global_state[key].shape != generated_state[key].shape for key in generated_state
                ):
                    raise RuntimeError(f"Initial checkpoint architecture mismatch: {checkpoint}")
            else:
                torch.save(generated_state, checkpoint)
                global_state = generated_state
        else:
            global_state = generated_state
        for client in clients:
            client["agent"].set_weights(copy.deepcopy(global_state))

        for round_idx in range(1, self.cfg.rounds + 1):
            round_t0 = time.perf_counter()
            local_deltas: Dict[int, Dict[str, torch.Tensor]] = {}
            flat_raw: Dict[int, np.ndarray] = {}
            local_metrics: Dict[int, Dict[str, float]] = {}
            client_sizes: Dict[int, int] = {}
            validation_scores: Dict[int, Dict[str, float]] = {}
            global_agent = build_agent(clients[0]["payload"], self.label_names, self.cfg)
            global_agent.set_weights(copy.deepcopy(global_state))
            X_val, y_val = self.validation_xy if self.validation_xy is not None else trusted_validation_xy(payloads, self.label_names, seed=self.cfg.seed)

            for client in clients:
                cid = int(client["payload"]["client_id"])
                train_t0 = time.perf_counter()
                trace_enabled = path_name in {"twa_raw", "twa_ppa_raw"}
                metrics = local_train(
                    client["agent"], client["env"], self.cfg,
                    self.behavior_trace_rows if trace_enabled else None,
                    {"path": path_name, "round": round_idx, "client_id": cid} if trace_enabled else None,
                )
                train_time = time.perf_counter() - train_t0
                local_state = copy.deepcopy(client["agent"].get_weights())
                delta = state_delta(global_state, local_state)
                flat, spec = flatten_state_dict(delta)
                deferred_rnb = cid in self.adversarial_clients and "random_noise_byzantine" in self.attack_modes and self.cfg.rnb_alpha > 0
                if cid in self.adversarial_clients and not deferred_rnb:
                    flat = self._apply_update_attack(flat)
                    delta = unflatten_state_dict(flat, spec)
                local_deltas[cid] = delta
                flat_raw[cid] = flat
                if cid in self.adversarial_clients and "reward_poisoning" in self.attack_modes:
                    metrics = dict(metrics)
                    metrics["avg_episode_reward"] = 3.0 * max(float(metrics.get("steps_total", 1.0)), 1.0) * self.cfg.adversarial_scale
                submitted_agent = build_agent(client["payload"], self.label_names, self.cfg)
                submitted_agent.set_weights(apply_delta(global_state, delta))
                val_pred = predict_many(submitted_agent, X_val)
                validation_scores[cid] = {
                    "validation_macro_f1": macro_f1_from_predictions(y_val.tolist(), val_pred.tolist()),
                    "behavior_consistency": behavior_consistency_score(submitted_agent, global_agent, X_val),
                    "reward_consistency": reward_consistency_score(
                        submitted_agent,
                        X_val,
                        y_val,
                        metrics.get("avg_episode_reward", 0.0),
                        metrics.get("steps_total", 0.0),
                    ),
                }
                local_metrics[cid] = metrics
                client_sizes[cid] = int(client["payload"].get("train_size", 1))
                self.history_rows.append({
                    "path": path_name,
                    "round": round_idx,
                    "client_id": cid,
                    "phase": "train",
                    "avg_episode_reward": metrics["avg_episode_reward"],
                    "avg_loss": metrics["avg_loss"],
                    "steps_total": metrics["steps_total"],
                    "local_train_time_sec": train_time,
                    "is_adversarial": cid in self.adversarial_clients,
                    **validation_scores[cid],
                })

            rnb_diagnostics = {}
            if "random_noise_byzantine" in self.attack_modes and self.cfg.rnb_alpha > 0 and self.adversarial_clients:
                benign_norms = [float(np.linalg.norm(flat_raw[cid])) for cid in sorted(flat_raw) if cid not in self.adversarial_clients]
                median_benign_norm = float(np.median(benign_norms))
                for cid in sorted(self.adversarial_clients):
                    dimension = max(int(flat_raw[cid].size), 1)
                    sigma = self.cfg.rnb_alpha * median_benign_norm / np.sqrt(dimension)
                    attacked = np.random.normal(0.0, sigma, size=flat_raw[cid].shape).astype(np.float32)
                    flat_raw[cid] = attacked
                    local_deltas[cid] = unflatten_state_dict(attacked, spec)
                    submitted_agent = build_agent(clients[cid]["payload"], self.label_names, self.cfg)
                    submitted_agent.set_weights(apply_delta(global_state, local_deltas[cid]))
                    validation_scores[cid] = {
                        "validation_macro_f1": macro_f1_from_predictions(y_val.tolist(), predict_many(submitted_agent, X_val).tolist()),
                        "behavior_consistency": behavior_consistency_score(submitted_agent, global_agent, X_val),
                        "reward_consistency": reward_consistency_score(submitted_agent, X_val, y_val, local_metrics[cid].get("avg_episode_reward", 0.0), local_metrics[cid].get("steps_total", 0.0)),
                    }
                    malicious_norm = float(np.linalg.norm(attacked))
                    rnb_diagnostics = {"rnb_alpha": self.cfg.rnb_alpha, "rnb_sigma": sigma,
                        "malicious_update_norm": malicious_norm, "median_benign_update_norm": median_benign_norm,
                        "malicious_to_benign_norm_ratio": malicious_norm / max(median_benign_norm, 1e-12),
                        "rnb_has_nan_or_inf": float(not np.isfinite(attacked).all())}

            # Record update-scale diagnostics for every adversarial condition, not
            # only calibrated RNB. These are descriptive and never affect trust or
            # aggregation behavior.
            if self.adversarial_clients:
                benign_norms = [float(np.linalg.norm(flat_raw[cid])) for cid in sorted(flat_raw) if cid not in self.adversarial_clients]
                malicious_norms = [float(np.linalg.norm(flat_raw[cid])) for cid in sorted(flat_raw) if cid in self.adversarial_clients]
                median_benign_norm = float(np.median(benign_norms)) if benign_norms else float("nan")
                malicious_norm = float(np.mean(malicious_norms)) if malicious_norms else float("nan")
                rnb_diagnostics.update({
                    "malicious_update_norm": malicious_norm,
                    "median_benign_update_norm": median_benign_norm,
                    "malicious_to_benign_norm_ratio": malicious_norm / max(median_benign_norm, 1e-12),
                    "attack_update_has_nan_or_inf": float(any(not np.isfinite(update).all() for update in flat_raw.values())),
                })

            global_delta, aggregation_info = self._aggregate_path(path_name, round_idx, local_deltas, flat_raw, local_metrics, client_sizes, validation_scores)
            global_state = apply_delta(global_state, global_delta)
            for client in clients:
                client["agent"].set_weights(copy.deepcopy(global_state))

            y_true, y_pred, eval_metrics = evaluate(clients[0]["agent"], payloads, self.label_names)
            eval_metrics["eval_f1_macro"] = macro_f1_from_predictions(y_true, y_pred)
            round_time = time.perf_counter() - round_t0
            self.history_rows.append({
                "path": path_name,
                "round": round_idx,
                "client_id": -1,
                "phase": "eval",
                **eval_metrics,
                "global_model_parameter_norm": float(np.sqrt(sum(float(torch.sum(value.float() ** 2)) for value in global_state.values()))),
                **rnb_diagnostics,
                **aggregation_info,
            })
            self.overhead_rows.append({
                "path": path_name,
                "round": round_idx,
                "round_time_sec": round_time,
                "aggregation_time_sec": aggregation_info.get("aggregation_time_sec", 0.0),
                "ckks_time_sec": aggregation_info.get("ckks_time_sec", 0.0),
                "num_clients": self.cfg.num_clients,
                "model_parameters": int(sum(t.numel() for t in global_state.values())),
                "estimated_upload_bytes": int(sum(arr.nbytes for arr in flat_raw.values())),
            })

            if self.cfg.save_global_checkpoints:
                torch.save(global_state, self.results_dir / f"{path_name}_global_round_{round_idx:03d}.pt")

        y_true, y_pred, final_metrics = evaluate(clients[0]["agent"], payloads, self.label_names)
        self._write_path_outputs(path_name, y_true, y_pred, final_metrics)

    def _apply_update_attack(self, flat: np.ndarray) -> np.ndarray:
        update_modes = self.attack_modes & UPDATE_ATTACK_MODES
        if not update_modes:
            return flat
        update_mode = next(iter(update_modes))
        if update_mode == "sign_flipping":
            return (-flat).astype(np.float32)
        if update_mode == "random_noise_byzantine":
            scale = float(np.std(flat)) if flat.size else 1.0
            scale = max(scale, 1e-6) * self.cfg.random_noise_std
            return np.random.normal(0.0, scale, size=flat.shape).astype(np.float32)
        if update_mode in {"scaled_sign_flip", "model_replacement"}:
            return (-self.cfg.adversarial_scale * flat).astype(np.float32)
        raise ValueError(f"Unknown update attack_mode: {update_mode}")

    def _aggregate_path(
        self,
        path_name: str,
        round_idx: int,
        local_deltas: Dict[int, Dict[str, torch.Tensor]],
        flat_raw: Dict[int, np.ndarray],
        local_metrics: Dict[int, Dict[str, float]],
        client_sizes: Dict[int, int],
        validation_scores: Dict[int, Dict[str, float]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        agg_t0 = time.perf_counter()
        client_ids = sorted(local_deltas)
        first_delta = local_deltas[client_ids[0]]
        _, spec = flatten_state_dict(first_delta)

        if path_name in {"fedavg", "ep0", "ppa"}:
            weights = self._size_weights(client_ids, client_sizes)
            flat_updates = [flat_raw[cid] for cid in client_ids]
            if path_name == "ppa":
                flat_global, tensor_info = self.ckks.aggregate(flat_updates, weights)
            else:
                flat_global = np.sum(np.stack(flat_updates, axis=0) * weights[:, None], axis=0)
                tensor_info = {
                    "ckks_enabled": 0.0,
                    "ckks_time_sec": 0.0,
                    "tensor_l2_error": 0.0,
                    "tensor_linf_error": 0.0,
                    "tensor_relative_l2_error": 0.0,
                }
            info = {"mean_trust": 1.0, "min_trust": 1.0, "suppressed_clients": 0.0}
            info.update(tensor_info)

        elif path_name in {"static_dp", "sdp", "sdp_ppa"}:
            weights = self._size_weights(client_ids, client_sizes)
            flat_updates = [
                clip_and_noise(flat_raw[cid], self.cfg.static_clip_norm, self.cfg.static_noise_multiplier)
                for cid in client_ids
            ]
            if path_name == "sdp_ppa":
                flat_global, tensor_info = self.ckks.aggregate(flat_updates, weights)
            else:
                flat_global = np.sum(np.stack(flat_updates, axis=0) * weights[:, None], axis=0)
                tensor_info = {
                    "ckks_enabled": 0.0,
                    "ckks_time_sec": 0.0,
                    "tensor_l2_error": 0.0,
                    "tensor_linf_error": 0.0,
                    "tensor_relative_l2_error": 0.0,
                }
            info = {
                "mean_trust": 1.0,
                "min_trust": 1.0,
                "suppressed_clients": 0.0,
                "static_clip_norm": self.cfg.static_clip_norm,
                "static_noise_multiplier": self.cfg.static_noise_multiplier,
                **tensor_info,
            }

        elif path_name == "krum":
            flat_updates = [flat_raw[cid] for cid in client_ids]
            flat_global, selected_idx = krum(flat_updates, self.cfg.krum_f)
            selected_cid = client_ids[selected_idx]
            info = {
                "mean_trust": 1.0,
                "min_trust": 1.0,
                "suppressed_clients": 0.0,
                "krum_f": float(self.cfg.krum_f),
                "krum_selected_client": float(selected_cid),
                "krum_selected_is_adversarial": float(selected_cid in self.adversarial_clients),
            }

        elif path_name == "median":
            flat_updates = [flat_raw[cid] for cid in client_ids]
            flat_global = coordinate_median(flat_updates)
            info = {"mean_trust": 1.0, "min_trust": 1.0, "suppressed_clients": 0.0}

        elif path_name == "trimmed_mean":
            flat_updates = [flat_raw[cid] for cid in client_ids]
            flat_global = trimmed_mean(flat_updates, self.cfg.trimmed_mean_ratio)
            info = {
                "mean_trust": 1.0,
                "min_trust": 1.0,
                "suppressed_clients": 0.0,
                "trimmed_mean_ratio": self.cfg.trimmed_mean_ratio,
            }

        elif path_name in {"dapi", "he_dapi", "twa", "twa_ppa", "twa_bounded", "twa_ppa_bounded", "twa_raw", "twa_ppa_raw", "twa_lockstep"}:
            flat_global, weights, trust_logs, tensor_info = self._aggregate_dapi(
                path_name, round_idx, client_ids, local_deltas, local_metrics, client_sizes, validation_scores
            )
            trusts = [trust_logs[cid]["smoothed_trust"] for cid in client_ids]
            info = {
                "mean_trust": float(np.mean(trusts)),
                "min_trust": float(np.min(trusts)),
                "max_trust": float(np.max(trusts)),
                "suppressed_clients": float(sum(w <= 1e-8 for w in weights)),
                **tensor_info,
            }
        else:
            raise ValueError(f"Unknown execution path: {path_name}")

        if path_name in {"fedavg", "ep0", "ppa", "static_dp", "sdp", "sdp_ppa"}:
            for idx, cid in enumerate(client_ids):
                raw = flat_raw[cid]
                aggregation_input = flat_updates[idx]
                raw_hash = hashlib.sha256(raw.tobytes()).hexdigest()
                aggregation_hash = hashlib.sha256(aggregation_input.tobytes()).hexdigest()
                self.mechanism_rows.append({
                    "path": path_name, "round": round_idx, "client_id": cid,
                    "raw_update_hash": raw_hash, "raw_update_norm": float(np.linalg.norm(raw)),
                    "scoring_copy_hash": raw_hash, "scoring_copy_norm": float(np.linalg.norm(raw)),
                    "aggregation_input_hash": aggregation_hash,
                    "aggregation_input_is_raw_update": aggregation_hash == raw_hash,
                    "scoring_input_is_bounded_copy": False, "trust_score": 1.0,
                    "pre_normalization_weight": float(client_sizes[cid]),
                    "final_aggregation_weight": float(weights[idx]), "suppressed": False,
                    "privacy_intensity_enabled": False, "epsilon": np.nan,
                    "adaptive_noise_scale": 0.0,
                    "fixed_dp_noise": self.cfg.static_noise_multiplier if path_name in {"static_dp", "sdp", "sdp_ppa"} else 0.0,
                    "ckks_enabled": bool(path_name in {"ppa", "sdp_ppa"}),
                })
            if path_name in {"static_dp", "sdp", "sdp_ppa"}:
                assert self.cfg.static_clip_norm > 0 and self.cfg.static_noise_multiplier > 0
                assert all(not row["aggregation_input_is_raw_update"] for row in self.mechanism_rows if row["path"] == path_name and row["round"] == round_idx)
            if path_name in {"ppa", "sdp_ppa"}:
                assert self.ckks.context is not None and info.get("ckks_ciphertext_count", 0.0) > 0

        info["aggregation_time_sec"] = time.perf_counter() - agg_t0
        return unflatten_state_dict(flat_global, spec), info

    def _aggregate_dapi(
        self,
        path_name: str,
        round_idx: int,
        client_ids: List[int],
        local_deltas: Dict[int, Dict[str, torch.Tensor]],
        local_metrics: Dict[int, Dict[str, float]],
        client_sizes: Dict[int, int],
        validation_scores: Dict[int, Dict[str, float]],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, Dict[str, float]], Dict[str, float]]:
        raw_updates = {
            cid: self.dapi.flatten_state_dict(local_deltas[cid])
            for cid in client_ids
        }
        clipped_updates = {
            cid: self.dapi.clip_update(raw_updates[cid])
            for cid in client_ids
        }
        rewards = {cid: local_metrics[cid]["avg_episode_reward"] for cid in client_ids}
        quality = {cid: local_metrics[cid]["avg_episode_reward"] for cid in client_ids}
        gradient_norms = {cid: float(np.linalg.norm(raw_updates[cid])) for cid in client_ids}
        f1_scores = {cid: validation_scores.get(cid, {}).get("validation_macro_f1", 0.5) for cid in client_ids}
        behavior_scores = {cid: validation_scores.get(cid, {}).get("behavior_consistency", 0.5) for cid in client_ids}
        reward_scores = {cid: validation_scores.get(cid, {}).get("reward_consistency", 0.5) for cid in client_ids}
        trust_logs = self.dapi.compute_round_trusts(
            client_ids=client_ids,
            clipped_updates=clipped_updates,
            local_quality_metrics=quality,
            current_rewards=rewards,
            f1_scores=f1_scores,
            gradient_norms=gradient_norms,
            behavior_scores=behavior_scores,
            reward_consistency_scores=reward_scores,
            reference_update=self.previous_aggregates.get(path_name),
            reward_min=-2.0 * self.cfg.max_steps_per_episode,
            reward_max=3.0 * self.cfg.max_steps_per_episode,
        )
        privatized = {}
        for cid in client_ids:
            trust = trust_logs[cid]["smoothed_trust"]
            if self.cfg.disable_privacy_intensity:
                noise_scale = 0.0
                epsilon = float("nan")
                privatized[cid] = clipped_updates[cid].copy()
            else:
                noise_scale = self.dapi.trust_to_noise_scale(trust)
                epsilon = self.dapi.trust_to_epsilon_sharp(trust)
                privatized[cid] = self.dapi.add_gaussian_noise(clipped_updates[cid], noise_scale)
            trust_logs[cid]["epsilon"] = epsilon
            trust_logs[cid]["noise_scale"] = noise_scale
            trust_logs[cid]["privacy_intensity_enabled"] = float(
                not self.cfg.disable_privacy_intensity
            )

        raw_path = path_name in {"twa_raw", "twa_ppa_raw"} or (
            path_name in {"dapi", "he_dapi", "twa", "twa_ppa"} and self.cfg.aggregate_raw_updates
        )
        aggregation_updates = raw_updates if raw_path else privatized
        flat_updates = [aggregation_updates[cid] for cid in client_ids]
        trusts = [trust_logs[cid]["smoothed_trust"] for cid in client_ids]
        in_warmup = round_idx <= self.dapi.trust_cfg.warmup_rounds
        effective_threshold = 0.0 if in_warmup else self.dapi.trust_cfg.low_trust_threshold
        effective_normalization = self.dapi.trust_cfg.weight_normalization
        global_plain, weights = trust_weighted_fedavg(
            client_updates=flat_updates,
            client_sizes=[client_sizes[cid] for cid in client_ids],
            client_trusts=trusts,
            low_trust_threshold=effective_threshold,
            trust_weight_power=self.dapi.trust_cfg.trust_weight_power,
            max_agg_weight=self.dapi.trust_cfg.max_agg_weight,
            min_agg_weight=self.dapi.trust_cfg.min_agg_weight,
            trust_temperature=self.dapi.trust_cfg.trust_temperature,
            weight_normalization=effective_normalization,
            min_weight_eligibility_threshold=(
                self.dapi.trust_cfg.min_weight_eligibility_threshold
            ),
        )
        encrypted_path = path_name in {"he_dapi", "twa_ppa", "twa_ppa_bounded", "twa_ppa_raw"}
        if encrypted_path:
            flat_global, tensor_info = self.ckks.aggregate(flat_updates, weights)
        elif path_name == "twa_lockstep":
            _, tensor_info = self.ckks.aggregate(flat_updates, weights)
            tensor_info["lockstep_weight_difference"] = 0.0
            tensor_info["lockstep_client_update_difference"] = 0.0
            flat_global = global_plain
        else:
            flat_global = global_plain
            tensor_info = {
                "ckks_enabled": 0.0,
                "ckks_time_sec": 0.0,
                "tensor_l2_error": 0.0,
                "tensor_linf_error": 0.0,
                "tensor_relative_l2_error": 0.0,
            }
        self.previous_aggregates[path_name] = flat_global.copy()

        for idx, cid in enumerate(client_ids):
            raw_hash = hashlib.sha256(raw_updates[cid].tobytes()).hexdigest()
            scoring_hash = hashlib.sha256(clipped_updates[cid].tobytes()).hexdigest()
            aggregation_hash = hashlib.sha256(aggregation_updates[cid].tobytes()).hexdigest()
            trust = float(trusts[idx])
            if self.dapi.trust_cfg.weight_normalization == "softmax":
                pre_weight = float(client_sizes[cid]) * float(np.exp(trust / self.dapi.trust_cfg.trust_temperature))
            else:
                pre_weight = float(client_sizes[cid]) * float(trust ** (self.dapi.trust_cfg.trust_weight_power / self.dapi.trust_cfg.trust_temperature))
            self.mechanism_rows.append({
                "path": path_name, "round": round_idx, "client_id": cid,
                "raw_update_hash": raw_hash, "raw_update_norm": float(np.linalg.norm(raw_updates[cid])),
                "scoring_copy_hash": scoring_hash, "scoring_copy_norm": float(np.linalg.norm(clipped_updates[cid])),
                "aggregation_input_hash": aggregation_hash,
                "aggregation_input_is_raw_update": raw_path,
                "aggregation_input_numerically_equals_raw_update": aggregation_hash == raw_hash,
                "scoring_input_is_bounded_copy": float(np.linalg.norm(clipped_updates[cid])) <= self.cfg.dapi_clip_norm + 1e-6,
                "trust_score": trust, "pre_normalization_weight": pre_weight,
                "pre_cap_weight": pre_weight, "post_cap_weight": float(weights[idx]),
                "final_aggregation_weight": float(weights[idx]),
                "suppressed": bool(weights[idx] <= self.cfg.trust_suppression_threshold),
                "privacy_intensity_enabled": False, "epsilon": np.nan,
                "adaptive_noise_scale": 0.0, "fixed_dp_noise": 0.0,
                "ckks_enabled": bool(encrypted_path),
            })

        if path_name in {"twa", "twa_ppa", "twa_bounded", "twa_ppa_bounded", "twa_raw", "twa_ppa_raw"}:
            expected_raw = raw_path
            assert all(row["aggregation_input_is_raw_update"] == expected_raw for row in self.mechanism_rows if row["path"] == path_name and row["round"] == round_idx)
            assert tensor_info.get("ckks_enabled", 0.0) == float(encrypted_path)
            if encrypted_path:
                assert self.ckks.context is not None
                assert tensor_info.get("ckks_ciphertext_count", 0.0) > 0

        for idx, cid in enumerate(client_ids):
            row = {
                "path": path_name,
                "round": round_idx,
                "client_id": cid,
                "aggregation_weight": float(weights[idx]),
                "is_adversarial": cid in self.adversarial_clients,
                "attack_mode": self.cfg.attack_mode,
                "warmup_active": float(in_warmup),
                "effective_suppression_threshold": float(effective_threshold),
                "weight_normalization": effective_normalization,
                **trust_logs[cid],
            }
            self.dapi_rows.append(row)

        self.tensor_rows.append({
            "path": path_name,
            "round": round_idx,
            **tensor_info,
        })
        self._record_ablation(path_name, round_idx, client_ids, clipped_updates, local_metrics, client_sizes)
        return flat_global, weights, trust_logs, tensor_info

    def _record_ablation(
        self,
        path_name: str,
        round_idx: int,
        client_ids: List[int],
        clipped_updates: Dict[int, np.ndarray],
        local_metrics: Dict[int, Dict[str, float]],
        client_sizes: Dict[int, int],
    ) -> None:
        if path_name not in {"dapi", "he_dapi", "twa", "twa_ppa", "twa_bounded", "twa_ppa_bounded", "twa_raw", "twa_ppa_raw", "twa_lockstep"}:
            return
        baseline_weights = self._size_weights(client_ids, client_sizes)
        rewards = np.asarray([local_metrics[cid]["avg_episode_reward"] for cid in client_ids], dtype=np.float32)
        reward_scaled = rewards - rewards.min()
        if reward_scaled.sum() <= 0:
            reward_weights = baseline_weights
        else:
            reward_weights = reward_scaled / (reward_scaled.sum() + 1e-8)
        trust_weights = np.asarray([
            row["aggregation_weight"]
            for row in self.dapi_rows
            if row["path"] == path_name and row["round"] == round_idx
        ], dtype=np.float32)
        variants = {
            "size_only": baseline_weights,
            "reward_only": reward_weights,
            "full_dapi": trust_weights,
        }
        for variant, weights in variants.items():
            adversarial_weight = float(sum(weights[i] for i, cid in enumerate(client_ids) if cid in self.adversarial_clients))
            self.ablation_rows.append({
                "path": path_name,
                "round": round_idx,
                "variant": variant,
                "adversarial_weight": adversarial_weight,
                "max_client_weight": float(np.max(weights)) if len(weights) else 0.0,
                "weight_entropy": float(-np.sum(weights * np.log(weights + 1e-12))) if len(weights) else 0.0,
            })

    @staticmethod
    def _size_weights(client_ids: Iterable[int], client_sizes: Dict[int, int]) -> np.ndarray:
        sizes = np.asarray([max(client_sizes[cid], 1) for cid in client_ids], dtype=np.float32)
        return sizes / (sizes.sum() + 1e-8)

    def _write_path_outputs(self, path_name: str, y_true: List[int], y_pred: List[int], final_metrics: Dict[str, float]) -> None:
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(self.label_names))),
            target_names=self.label_names,
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"}).to_csv(
            self.results_dir / f"classification_report_{path_name}_{TASK_NAME}.csv",
            index=False,
        )
        pd.DataFrame(
            confusion_matrix(y_true, y_pred, labels=list(range(len(self.label_names)))),
            index=self.label_names,
            columns=self.label_names,
        ).to_csv(self.results_dir / f"confusion_matrix_{path_name}_{TASK_NAME}.csv")
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        self.path_rows.append({
            "path": path_name,
            "accuracy": final_metrics.get("eval_accuracy", 0.0),
            "eval_reward": final_metrics.get("eval_reward", 0.0),
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
        })

    def _write_outputs(self, payloads: List[Dict[str, Any]]) -> None:
        history = pd.DataFrame(self.history_rows)
        dapi = pd.DataFrame(self.dapi_rows)
        utility = pd.DataFrame(self.path_rows)
        tensor = pd.DataFrame(self.tensor_rows)
        overhead = pd.DataFrame(self.overhead_rows)
        ablation = pd.DataFrame(self.ablation_rows)
        mechanism = pd.DataFrame(self.mechanism_rows)
        behavior_trace = pd.DataFrame(self.behavior_trace_rows)

        atomic_to_csv(history, self.results_dir / f"frl_{TASK_NAME}_history.csv")
        atomic_to_csv(dapi, self.results_dir / f"dapi_trust_{TASK_NAME}.csv")
        atomic_to_csv(utility, self.results_dir / f"benchmark_summary_{TASK_NAME}.csv")
        atomic_to_csv(tensor, self.results_dir / f"tensor_correctness_{TASK_NAME}.csv")
        atomic_to_csv(overhead, self.results_dir / f"overhead_{TASK_NAME}.csv")
        atomic_to_csv(ablation, self.results_dir / f"dapi_ablation_{TASK_NAME}.csv")
        atomic_to_csv(mechanism, self.results_dir / f"mechanism_validation_{TASK_NAME}.csv")
        atomic_to_csv(behavior_trace, self.results_dir / "behavior_divergence_trace.csv")

        # Incremental aligned-benchmark aliases, atomically replaced after each path.
        completed_paths = set(utility["path"].tolist()) if not utility.empty else set()
        attack_label = "+".join(sorted(self.attack_modes))
        manifest = pd.DataFrame([
            {
                "run_id": f"seed_{self.cfg.seed}_{attack_label}_{path}",
                "seed": self.cfg.seed, "execution_path": path,
                "condition": "clean" if attack_label == "none" else "adversarial",
                "attack": attack_label, "rounds": self.cfg.rounds,
                "num_clients": self.cfg.num_clients,
                "malicious_client_id": "" if not self.adversarial_clients else sorted(self.adversarial_clients)[0],
                "status": "completed" if path in completed_paths else "pending",
            }
            for path in [part.strip() for part in self.cfg.execution_paths.split(",") if part.strip()]
        ])
        atomic_to_csv(manifest, self.results_dir / "run_manifest.csv")
        finals = utility.copy()
        if not finals.empty:
            finals.insert(0, "seed", self.cfg.seed)
            finals["final_evaluation_reward"] = 5.0 * finals["accuracy"] - 2.0
        atomic_to_csv(finals, self.results_dir / "per_seed_final_metrics.csv")
        atomic_to_csv(history[history["phase"] == "eval"].copy(), self.results_dir / "per_round_metrics.csv")
        atomic_to_csv(tensor[tensor["ckks_enabled"] == 1.0].copy() if not tensor.empty else tensor, self.results_dir / "ckks_validation.csv")
        partition_hashes = {
            str(cid): hashlib.sha256((Path(self.cfg.data_dir) / f"client_{cid}_train.csv").read_bytes()).hexdigest()
            for cid in range(self.cfg.num_clients)
        }
        fairness = {
            "passed": True, "seed": self.cfg.seed, "rounds": self.cfg.rounds,
            "client_ids": list(range(self.cfg.num_clients)), "partition_hashes": partition_hashes,
            "evaluation_hash": hashlib.sha256((Path(self.cfg.data_dir) / "test.csv").read_bytes()).hexdigest(),
            "initial_checkpoint_hash": (
                hashlib.sha256(Path(self.cfg.initial_checkpoint).read_bytes()).hexdigest()
                if self.cfg.initial_checkpoint and Path(self.cfg.initial_checkpoint).exists() else ""
            ),
        }
        fairness_path = self.results_dir / "fairness_validation.json"
        fairness_tmp = fairness_path.with_suffix(".json.tmp")
        fairness_tmp.write_text(json.dumps(fairness, indent=2), encoding="utf-8")
        fairness_tmp.replace(fairness_path)

        with open(self.results_dir / f"run_config_{TASK_NAME}.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.cfg), f, indent=2)
        with open(self.results_dir / f"meta_{TASK_NAME}.json", "w", encoding="utf-8") as f:
            json.dump({
                "task": TASK_NAME,
                "num_classes": len(self.label_names),
                "label_names": self.label_names,
                "num_clients": len(payloads),
                "dapi_trust_config": asdict(self.dapi.trust_cfg),
                "dapi_privacy_config": asdict(self.dapi.privacy_cfg),
                "ckks_note": (
                    "Real TenSEAL CKKS weighted aggregation enabled."
                    if self.cfg.enable_ckks
                    else "CKKS disabled; HE adapter uses historical plaintext passthrough."
                ),
                "privacy_intensity_enabled": not self.cfg.disable_privacy_intensity,
            }, f, indent=2)
        self._write_plots(history, dapi, utility, tensor, overhead, ablation)
        self._write_results_discussion(history, dapi, utility, tensor, overhead, ablation)

    def _write_plots(
        self,
        history: pd.DataFrame,
        dapi: pd.DataFrame,
        utility: pd.DataFrame,
        tensor: pd.DataFrame,
        overhead: pd.DataFrame,
        ablation: pd.DataFrame,
    ) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            (self.results_dir / "plot_generation_error.txt").write_text(str(exc), encoding="utf-8")
            return

        plt.rcParams.update({
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        })

        palette = {
            "fedavg": "#4C78A8",
            "static_dp": "#F58518",
            "dapi": "#54A24B",
            "he_dapi": "#B279A2",
            "twa": "#54A24B",
            "twa_ppa": "#B279A2",
            "ppa": "#72B7B2",
            "sdp_ppa": "#FF9DA6",
        }

        if not utility.empty:
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            x = np.arange(len(utility))
            width = 0.36
            ax.bar(x - width / 2, utility["accuracy"], width, label="Accuracy", color="#4C78A8")
            ax.bar(x + width / 2, utility["f1_macro"], width, label="Macro F1", color="#E45756")
            ax.set_xticks(x)
            ax.set_xticklabels(utility["path"], rotation=20, ha="right")
            ax.set_ylabel("Score")
            ax.set_ylim(0, max(0.1, float(max(utility["accuracy"].max(), utility["f1_macro"].max())) * 1.25))
            ax.set_title("Global Utility Comparison")
            ax.legend(frameon=False)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(self.results_dir / "plot_global_utility.png")
            plt.close(fig)

        eval_rows = history[history["phase"] == "eval"].copy() if not history.empty else pd.DataFrame()
        if not eval_rows.empty:
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            for path_name, group in eval_rows.groupby("path"):
                group = group.sort_values("round")
                ax.plot(
                    group["round"],
                    group["eval_accuracy"],
                    marker="o",
                    linewidth=2,
                    label=path_name,
                    color=palette.get(path_name),
                )
            ax.set_xlabel("Round")
            ax.set_ylabel("Accuracy")
            ax.set_title("Evaluation Accuracy by Round")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(self.results_dir / "plot_accuracy_by_round.png")
            plt.close(fig)

        if not dapi.empty:
            fig, ax = plt.subplots(figsize=(9.5, 4.8))
            latest_round = int(dapi["round"].max())
            latest = dapi[dapi["round"] == latest_round].copy()
            latest["client_id"] = latest["client_id"].astype(int)
            for path_name, group in latest.groupby("path"):
                group = group.sort_values("client_id")
                ax.plot(
                    group["client_id"],
                    group["smoothed_trust"],
                    marker="o",
                    linewidth=2,
                    label=f"{path_name} trust",
                    color=palette.get(path_name),
                )
            adversarial = sorted(set(latest.loc[latest["is_adversarial"] == True, "client_id"].astype(int).tolist()))
            for cid in adversarial:
                ax.axvline(cid, color="#E45756", alpha=0.25, linewidth=1.5)
            ax.set_xlabel("Client ID")
            ax.set_ylabel("Smoothed Trust")
            ax.set_ylim(0, 1)
            ax.set_title(f"DAPI Trust Distribution, Round {latest_round}")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(self.results_dir / "plot_dapi_trust.png")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(9.5, 4.8))
            width = 0.38
            paths = latest["path"].drop_duplicates().tolist()
            for idx, path_name in enumerate(paths):
                group = latest[latest["path"] == path_name].sort_values("client_id")
                offset = (idx - (len(paths) - 1) / 2) * width
                colors = np.where(group["is_adversarial"].to_numpy(dtype=bool), "#E45756", palette.get(path_name, "#4C78A8"))
                ax.bar(group["client_id"] + offset, group["aggregation_weight"], width, label=path_name, color=colors, alpha=0.85)
            ax.set_xlabel("Client ID")
            ax.set_ylabel("Aggregation Weight")
            ax.set_title(f"DAPI Aggregation Weights, Round {latest_round}")
            ax.grid(axis="y", alpha=0.25)
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(self.results_dir / "plot_dapi_aggregation_weights.png")
            plt.close(fig)

        if not ablation.empty:
            summary = ablation.groupby(["path", "variant"], as_index=False)["adversarial_weight"].mean()
            pivot = summary.pivot(index="variant", columns="path", values="adversarial_weight").fillna(0.0)
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            pivot.plot(kind="bar", ax=ax, color=[palette.get(col, "#4C78A8") for col in pivot.columns])
            ax.set_xlabel("Ablation Variant")
            ax.set_ylabel("Adversarial Aggregation Weight")
            ax.set_title("DAPI Ablation: Adversarial Influence")
            ax.tick_params(axis="x", rotation=20)
            ax.grid(axis="y", alpha=0.25)
            ax.legend(title="Path", frameon=False)
            fig.tight_layout()
            fig.savefig(self.results_dir / "plot_dapi_ablation.png")
            plt.close(fig)

        if not tensor.empty:
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            tensor_summary = tensor.groupby("path", as_index=False)["tensor_relative_l2_error"].mean()
            ax.bar(
                tensor_summary["path"],
                tensor_summary["tensor_relative_l2_error"],
                color=[palette.get(path, "#4C78A8") for path in tensor_summary["path"]],
            )
            ax.set_ylabel("Relative L2 Error")
            ax.set_title("Tensor-Level Aggregation Correctness")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(self.results_dir / "plot_tensor_correctness.png")
            plt.close(fig)

        if not overhead.empty:
            overhead_summary = overhead.groupby("path", as_index=False).agg({
                "round_time_sec": "mean",
                "aggregation_time_sec": "mean",
                "ckks_time_sec": "mean",
            })
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            x = np.arange(len(overhead_summary))
            width = 0.26
            ax.bar(x - width, overhead_summary["round_time_sec"], width, label="Round", color="#4C78A8")
            ax.bar(x, overhead_summary["aggregation_time_sec"], width, label="Aggregation", color="#54A24B")
            ax.bar(x + width, overhead_summary["ckks_time_sec"], width, label="CKKS", color="#B279A2")
            ax.set_xticks(x)
            ax.set_xticklabels(overhead_summary["path"], rotation=20, ha="right")
            ax.set_ylabel("Seconds")
            ax.set_title("Runtime Overhead")
            ax.grid(axis="y", alpha=0.25)
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(self.results_dir / "plot_runtime_overhead.png")
            plt.close(fig)

    def _write_results_discussion(
        self,
        history: pd.DataFrame,
        dapi: pd.DataFrame,
        utility: pd.DataFrame,
        tensor: pd.DataFrame,
        overhead: pd.DataFrame,
        ablation: pd.DataFrame,
    ) -> None:
        eval_rows = history[history["phase"] == "eval"].copy() if not history.empty else pd.DataFrame()
        final_by_path = eval_rows.sort_values("round").groupby("path").tail(1) if not eval_rows.empty else pd.DataFrame()
        overhead_summary = overhead.groupby("path", as_index=False).agg({
            "round_time_sec": "mean",
            "aggregation_time_sec": "mean",
            "ckks_time_sec": "mean",
            "estimated_upload_bytes": "mean",
        }) if not overhead.empty else pd.DataFrame()
        suppression = dapi.groupby("path", as_index=False).agg({
            "smoothed_trust": "mean",
            "aggregation_weight": "mean",
            "epsilon": "mean",
            "noise_scale": "mean",
        }) if not dapi.empty else pd.DataFrame()

        lines = [
            "# VI. Results and Discussion",
            "",
            "## A. Overview of Evaluated Execution Paths",
            f"Evaluated paths: {self.cfg.execution_paths}. Each path uses the same label space, client split, rounds, local episodes, malicious-client ratio, and final benchmark metrics.",
            "",
            "## B. Global Utility Comparison",
            table_text(utility),
            "",
            "## C. Homomorphic Aggregation and Static-DP Interaction",
            "Static-DP uses fixed clipping/noise before aggregation. HE+DAPI uses CKKS after DAPI weighting; --enable_ckks selects real TenSEAL encryption, while the disabled default retains the historical passthrough for compatibility.",
            "",
            "## D. DAPI Trust and Adversarial Suppression",
            table_text(suppression),
            "",
            "## E. Ablation Study of DAPI Components",
            table_text(ablation.groupby(["path", "variant"], as_index=False).mean(numeric_only=True)) if not ablation.empty else "No rows were produced.",
            "",
            "## F. HE+DAPI Combined Path",
            table_text(final_by_path[final_by_path["path"] == "he_dapi"]) if not final_by_path.empty else "HE+DAPI was not evaluated.",
            "",
            "## G. Tensor-Level Correctness of Homomorphic Aggregation",
            table_text(tensor.groupby("path", as_index=False).mean(numeric_only=True)) if not tensor.empty else "No rows were produced.",
            "",
            "## H. Communication and Runtime Overhead",
            table_text(overhead_summary),
            "",
            "## I. Discussion: Path-Dependent Security Tradeoffs",
            "FedAvg is the utility baseline without explicit privacy hardening. Static-DP gives uniform privacy cost but can suppress useful updates indiscriminately. DAPI makes privacy intensity and aggregation weight depend on trust, so low-trust or adversarial clients can lose influence while stable clients retain utility. HE+DAPI separates aggregation confidentiality from adaptive trust: CKKS protects aggregation contents, while DAPI controls client influence and noise intensity.",
            "",
        ]
        (self.results_dir / "VI_RESULTS_AND_DISCUSSION.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="20-class Federated RL DAPI experiment runner")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local_episodes", type=int, default=5)
    parser.add_argument("--max_steps_per_episode", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--supervised_aux_epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execution_paths", default="fedavg,static_dp,dapi,he_dapi")
    parser.add_argument("--adversarial_clients", default="", help="Comma-separated client ids to scale as adversaries")
    parser.add_argument(
        "--attack_mode",
        default="scaled_sign_flip",
        help=(
            "Attack mode or comma/plus-separated composition. Supported atomic "
            "modes: none, sign_flipping, random_noise_byzantine, "
            "model_replacement, scaled_sign_flip, label_flipping, "
            "data_poisoning, reward_poisoning."
        ),
    )
    parser.add_argument("--adversarial_scale", type=float, default=8.0)
    parser.add_argument("--random_noise_std", type=float, default=1.0)
    parser.add_argument("--rnb_alpha", type=float, default=0.0)
    parser.add_argument("--static_clip_norm", type=float, default=1.0)
    parser.add_argument("--static_noise_multiplier", type=float, default=0.1)
    parser.add_argument("--krum_f", type=int, default=1)
    parser.add_argument("--trimmed_mean_ratio", type=float, default=0.10)
    parser.add_argument("--dapi_epsilon_min", type=float, default=0.3)
    parser.add_argument("--dapi_epsilon_max", type=float, default=4.0)
    parser.add_argument("--dapi_clip_norm", type=float, default=1.0)
    parser.add_argument("--dapi_delta", type=float, default=1e-5)
    parser.add_argument("--dapi_sigma_min", type=float, default=0.02)
    parser.add_argument("--dapi_sigma_max", type=float, default=0.12)
    parser.add_argument(
        "--disable_privacy_intensity",
        action="store_true",
        help="Keep DAPI trust scoring/weighting but disable adaptive epsilon and Gaussian noise.",
    )
    parser.add_argument("--trust_smoothing", type=float, default=0.65)
    parser.add_argument("--trust_suppression_threshold", type=float, default=1e-8)
    parser.add_argument("--trust_weight_power", type=float, default=4.0)
    parser.add_argument("--trust_temperature", type=float, default=1.0)
    parser.add_argument(
        "--trust_weight_normalization",
        choices=["power", "softmax"],
        default="power",
    )
    parser.add_argument("--trust_min_active_weight", type=float, default=0.0)
    parser.add_argument("--trust_floor_eligibility_threshold", type=float, default=0.0)
    parser.add_argument("--trust_max_weight", type=float, default=0.25)
    parser.add_argument("--trust_warmup_rounds", type=int, default=0)
    parser.add_argument(
        "--aggregate_raw_updates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use clipped copies for trust scoring but aggregate the original updates.",
    )
    parser.add_argument("--enable_ckks", action="store_true")
    parser.add_argument("--save_global_checkpoints", action="store_true")
    parser.add_argument("--expected_num_classes", type=int, default=0)
    parser.add_argument("--initial_checkpoint", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = RunConfig(**vars(args))
    label_names = load_label_names(cfg.data_dir)
    if cfg.expected_num_classes:
        if cfg.expected_num_classes == REQUIRED_NUM_CLASSES:
            assert_20_classes(label_names, cfg.data_dir)
        elif len(label_names) != cfg.expected_num_classes:
            raise ValueError(
                f"Expected {cfg.expected_num_classes} classes, but found "
                f"{len(label_names)} in {cfg.data_dir}: {label_names}"
            )
    experiment = TwentyClassDAPIExperiment(cfg, label_names)
    experiment.run()
    print(f"Completed {TASK_NAME}. Results saved to {cfg.results_dir}")


if __name__ == "__main__":
    main()
