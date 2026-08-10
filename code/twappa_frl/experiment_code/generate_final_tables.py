"""Build the final paper tables exclusively from the official aligned benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = (42, 123, 2025)
ATTACKS = ("none", "sign_flipping", "random_noise_byzantine", "model_replacement", "label_flipping", "data_poisoning", "reward_poisoning")
PATHS = ("fedavg", "static_dp", "twa", "ppa", "sdp_ppa", "twa_ppa")
PATH_LABEL = dict(zip(PATHS, ("ep_0", "ep_sdp", "ep_twa", "ep_ppa", "ep_sdp-ppa", "ep_twa-ppa")))
ATTACK_LABEL = {"none": "∅", "sign_flipping": "Sign flipping", "random_noise_byzantine": "RNB-Primary (alpha=0.5)", "model_replacement": "Model replacement", "label_flipping": "Label flipping", "data_poisoning": "Data poisoning", "reward_poisoning": "Reward poisoning"}
SUPPRESSION_THRESHOLD = 1e-8


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pm(mean: float, std: float, digits: int = 6) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def aggregate(frame: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    stats = frame.groupby(keys, sort=False)[metrics].agg(["mean", "std"]).reset_index()
    stats.columns = ["_".join(x).rstrip("_") if isinstance(x, tuple) else x for x in stats.columns]
    for metric in metrics:
        stats[f"{metric}_mean_pm_std"] = [pm(m, s) for m, s in zip(stats[f"{metric}_mean"], stats[f"{metric}_std"])]
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/aligned_benchmark_official_clean_20260717")
    args = parser.parse_args()
    root = Path(args.root)
    output = root / "final_tables"
    output.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(root / "run_manifest.csv")
    fairness = json.loads((root / "fairness_validation.json").read_text(encoding="utf-8"))
    histories: list[dict] = []
    table3_source: list[dict] = []
    history_checks: list[dict] = []

    for seed in SEEDS:
        for attack in ATTACKS:
            folder = root / "runs" / f"seed_{seed}" / attack
            history_path = folder / "frl_20class_dapi_history.csv"
            history = pd.read_csv(history_path)
            evaluations = history[history.phase.eq("eval")].copy()
            counts = evaluations.groupby("path").agg(round_count=("round", "nunique"), min_round=("round", "min"), max_round=("round", "max"))
            exact = set(counts.index) == set(PATHS) and bool((counts.round_count == 30).all()) and bool((counts.min_round == 1).all()) and bool((counts.max_round == 30).all())
            history_checks.append({"seed": seed, "attack": attack, "history_sha256": sha256(history_path), "evaluation_rows": int(len(evaluations)), "path_count": int(evaluations.path.nunique()), "exact_6_paths_x_30_rounds": exact})
            if not exact:
                raise RuntimeError(f"Invalid evaluation-round structure: {history_path}")

            overhead = pd.read_csv(folder / "overhead_20class_dapi.csv")
            trust = pd.read_csv(folder / "dapi_trust_20class_dapi.csv")
            correctness = pd.read_csv(folder / "tensor_correctness_20class_dapi.csv")
            for path in PATHS:
                final = evaluations[(evaluations.path == path) & (evaluations["round"] == 30)].iloc[0]
                mean_round_time = float(overhead[overhead.path == path].round_time_sec.mean())
                histories.append({"seed": seed, "attack": attack, "condition": ATTACK_LABEL[attack], "path": path, "execution_path": PATH_LABEL[path], "round30_accuracy": float(final.eval_accuracy), "round30_macro_f1": float(final.eval_f1_macro), "final_evaluation_reward": float(5 * final.eval_accuracy - 2), "mean_round_time_sec": mean_round_time})

            if attack != "none":
                for path in ("twa", "twa_ppa"):
                    path_eval = evaluations[evaluations.path == path].sort_values("round")
                    adversary = trust[(trust.path == path) & (trust.client_id == 4)].sort_values("round")
                    weights = adversary.aggregation_weight.astype(float)
                    final = path_eval[path_eval["round"] == 30].iloc[0]
                    ckks = correctness[correctness.path == path]
                    table3_source.append({
                        "seed": seed, "attack": attack, "condition": ATTACK_LABEL[attack], "path": path,
                        "final_accuracy": float(final.eval_accuracy), "final_macro_f1": float(final.eval_f1_macro),
                        "mean_malicious_weight": float(weights.mean()), "final_malicious_weight": float(weights.iloc[-1]),
                        "suppressed_rounds_out_of_30": int((weights <= SUPPRESSION_THRESHOLD).sum()),
                        "suppressed_rounds_21_30": int((weights[adversary["round"].between(21, 30).to_numpy()] <= SUPPRESSION_THRESHOLD).sum()),
                        "mean_malicious_ema_trust": float(adversary.smoothed_trust.mean()),
                        "mean_B_adv": float(adversary.behavior_consistency.mean()), "mean_R_adv": float(adversary.reward_consistency.mean()),
                        "malicious_to_benign_update_norm_ratio": float(path_eval.malicious_to_benign_norm_ratio.mean()),
                        "mean_round_time_sec": float(overhead[overhead.path == path].round_time_sec.mean()),
                        "ckks_mean_absolute_error": float(ckks.tensor_mean_abs_error.mean()) if path == "twa_ppa" else np.nan,
                        "ckks_max_relative_l2_error": float(ckks.tensor_relative_l2_error.max()) if path == "twa_ppa" else np.nan,
                    })

    source2 = pd.DataFrame(histories)
    source3 = pd.DataFrame(table3_source)
    condition_order = [ATTACK_LABEL[attack] for attack in ATTACKS]
    source2["condition"] = pd.Categorical(source2["condition"], categories=condition_order, ordered=True)
    source2["execution_path"] = pd.Categorical(source2["execution_path"], categories=[PATH_LABEL[path] for path in PATHS], ordered=True)
    source3["condition"] = pd.Categorical(source3["condition"], categories=condition_order[1:], ordered=True)
    source2.to_csv(output / "table_II_per_seed_source.csv", index=False)
    source3.to_csv(output / "table_III_per_seed_source.csv", index=False)

    metrics2 = ["round30_accuracy", "round30_macro_f1", "final_evaluation_reward", "mean_round_time_sec"]
    table2 = aggregate(source2, ["condition", "execution_path", "attack", "path"], metrics2)
    plaintext_pairs = {"ppa": "fedavg", "sdp_ppa": "static_dp", "twa_ppa": "twa"}
    for metric in metrics2:
        table2[f"paired_abs_{metric}_difference_mean"] = np.nan
        table2[f"paired_abs_{metric}_difference_std"] = np.nan
        table2[f"paired_abs_{metric}_difference_mean_pm_std"] = ""
    table2["paired_plaintext_path"] = ""
    for index, row in table2.iterrows():
        if row.path not in plaintext_pairs:
            continue
        plain = plaintext_pairs[row.path]
        merged = source2[(source2.attack == row.attack) & (source2.path == row.path)].merge(source2[(source2.attack == row.attack) & (source2.path == plain)], on="seed", suffixes=("_encrypted", "_plaintext"))
        table2.loc[index, "paired_plaintext_path"] = PATH_LABEL[plain]
        for metric in metrics2:
            difference = (merged[f"{metric}_encrypted"] - merged[f"{metric}_plaintext"]).abs()
            mean, std = float(difference.mean()), float(difference.std())
            table2.loc[index, f"paired_abs_{metric}_difference_mean"] = mean
            table2.loc[index, f"paired_abs_{metric}_difference_std"] = std
            table2.loc[index, f"paired_abs_{metric}_difference_mean_pm_std"] = pm(mean, std)
    table2.to_csv(output / "table_II_attack_path_performance.csv", index=False)

    pivot = source3.pivot(index=["seed", "attack", "condition"], columns="path")
    flat = pivot.reset_index(); flat.columns = ["_".join(x).rstrip("_") if isinstance(x, tuple) else x for x in flat.columns]
    flat["condition"] = pd.Categorical(flat["condition"], categories=condition_order[1:], ordered=True)
    flat = flat.sort_values(["condition", "seed"])
    table3_metrics = []
    rename = {}
    for original, label in (("final_accuracy_twa", "twa_final_accuracy"), ("final_macro_f1_twa", "twa_final_macro_f1"), ("final_accuracy_twa_ppa", "twa_ppa_final_accuracy"), ("final_macro_f1_twa_ppa", "twa_ppa_final_macro_f1"), ("mean_malicious_weight_twa", "mean_malicious_twa_weight"), ("final_malicious_weight_twa", "final_malicious_twa_weight"), ("suppressed_rounds_out_of_30_twa", "suppressed_rounds_out_of_30"), ("suppressed_rounds_21_30_twa", "suppressed_rounds_21_30"), ("mean_malicious_ema_trust_twa", "mean_malicious_ema_trust"), ("mean_B_adv_twa", "mean_B_adv"), ("mean_R_adv_twa", "mean_R_adv"), ("malicious_to_benign_update_norm_ratio_twa", "malicious_to_benign_update_norm_ratio"), ("mean_round_time_sec_twa", "twa_mean_round_time_sec"), ("mean_round_time_sec_twa_ppa", "twa_ppa_mean_round_time_sec"), ("ckks_mean_absolute_error_twa_ppa", "ckks_mean_absolute_error"), ("ckks_max_relative_l2_error_twa_ppa", "ckks_max_relative_l2_error")):
        rename[original] = label; table3_metrics.append(label)
    flat = flat.rename(columns=rename)
    table3 = aggregate(flat, ["condition", "attack"], table3_metrics)
    table3.to_csv(output / "table_III_multiple_poisoning_results.csv", index=False)

    display2 = table2[["condition", "execution_path"] + [f"{m}_mean_pm_std" for m in metrics2]]
    display3 = table3[["condition"] + [f"{m}_mean_pm_std" for m in table3_metrics]]
    (output / "table_II_attack_path_performance.tex").write_text(display2.to_latex(index=False, escape=True), encoding="utf-8")
    (output / "table_III_multiple_poisoning_results.tex").write_text(display3.to_latex(index=False, escape=True), encoding="utf-8")

    checkpoint_alignment = manifest.groupby("seed").initial_checkpoint_hash.nunique().eq(1).all()
    actual_checkpoint_hashes = {str(seed): sha256(root / "checkpoints" / f"initial_seed_{seed}.pt") for seed in SEEDS}
    manifest_checkpoint_hashes = {str(seed): sorted(manifest[manifest.seed == seed].initial_checkpoint_hash.unique().tolist()) for seed in SEEDS}
    validation = {
        "source_policy": "Programmatically generated exclusively from official benchmark CSV files under this root; no manually entered or historical result values.",
        "official_root": str(root.resolve()), "expected_history_count": 21, "actual_history_count": len(history_checks),
        "all_histories_exactly_6_paths_x_30_evaluation_rounds": all(x["exact_6_paths_x_30_rounds"] for x in history_checks),
        "history_checks": history_checks, "expected_manifest_entries": 126, "actual_manifest_entries": int(len(manifest)),
        "all_manifest_entries_completed": bool(manifest.status.eq("completed").all()),
        "unique_manifest_config_hashes": sorted(manifest.config_hash.unique().tolist()),
        "single_partition_hash": bool(manifest.partition_hash.nunique() == 1), "partition_hash": fairness["partition_hash"],
        "partition_hashes": fairness["partition_hashes"], "evaluation_set_hash": fairness["evaluation_set_hash"],
        "shared_initial_checkpoint_within_each_seed": bool(checkpoint_alignment),
        "actual_checkpoint_hashes_match_manifest": all(manifest_checkpoint_hashes[str(seed)] == [actual_checkpoint_hashes[str(seed)]] for seed in SEEDS),
        "initial_checkpoint_hashes_by_seed": manifest_checkpoint_hashes,
        "actual_initial_checkpoint_hashes_by_seed": actual_checkpoint_hashes,
        "malicious_client_id": 4, "clean_malicious_client_id_is_empty": bool(manifest[manifest.attack == "none"].malicious_client_id.isna().all()),
        "adversarial_malicious_client_id_is_4": bool(manifest[manifest.attack != "none"].malicious_client_id.eq(4).all()),
        "rounds": 30, "seeds": list(SEEDS), "conditions": list(ATTACKS), "execution_paths": list(PATHS),
        "aligned_configuration": {"profile_name": "twa_manuscript_5client", "num_clients": 5, "client_ids": [0, 1, 2, 3, 4], "full_participation": True, "local_episodes": 1, "max_steps_per_episode": 16, "batch_size": 128, "aggregate_update_type": "norm_bounded", "twa_norm_bound": 1.0, "trust_ema_smoothing": 0.65, "trust_normalization": "power", "trust_temperature": 1.0, "trust_power": 4.0, "maximum_client_weight": 0.25, "privacy_intensity_enabled": False, "real_ckks": True, "rnb_primary_alpha": 0.5},
        "suppression_count_criterion": "A malicious-client round is counted as suppressed if and only if its recorded TWA aggregation weight w_adv <= 1e-8. Rounds 21-30 are filtered explicitly by round number. This is not the trust gate lambda=0.50 and is not a comparison with benign weights.",
        "rnb_primary": {"alpha": 0.5, "main_row_label": "RNB-Primary", "statement": "RNB-Primary was selected as the lowest tested magnitude satisfying the predefined potency criterion across all three seeds. Lower-magnitude RNB conditions were retained as calibration results but excluded from the main robustness table."},
        "excluded_sources": ["invalid diagnostic clipped/raw experiments", "TWA aggregation ablations", "one-round smoke tests", "RNB-Weak calibration", "historical manually entered results"],
        "generated_files": ["table_II_attack_path_performance.csv", "table_II_attack_path_performance.tex", "table_III_multiple_poisoning_results.csv", "table_III_multiple_poisoning_results.tex", "table_II_per_seed_source.csv", "table_III_per_seed_source.csv", "final_results_validation.json"],
    }
    (output / "final_results_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
