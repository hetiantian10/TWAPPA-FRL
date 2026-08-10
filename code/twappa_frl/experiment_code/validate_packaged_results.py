"""Validate the compact CSV evidence distributed with the paper release."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import pandas as pd


SEEDS = (42, 123, 2025)
ATTACKS = (
    "none",
    "sign_flipping",
    "random_noise_byzantine",
    "model_replacement",
    "label_flipping",
    "data_poisoning",
    "reward_poisoning",
)
PATHS = ("fedavg", "static_dp", "twa", "ppa", "sdp_ppa", "twa_ppa")
PATH_LABELS = ("ep_0", "ep_sdp", "ep_twa", "ep_ppa", "ep_sdp-ppa", "ep_twa-ppa")
CKKS_PATHS = ("ppa", "sdp_ppa", "twa_ppa")
ROUNDS = tuple(range(1, 31))


def _combinations(*columns: tuple[object, ...]) -> set[tuple[object, ...]]:
    return set(product(*columns))


def _observed(frame: pd.DataFrame, columns: list[str]) -> set[tuple[object, ...]]:
    return set(frame[columns].itertuples(index=False, name=None))


def validate_packaged_results(results_root: Path | str) -> dict[str, int | bool]:
    root = Path(results_root)
    manifest = pd.read_csv(root / "per_seed/run_manifest.csv")
    per_seed = pd.read_csv(root / "per_seed/per_seed_final_metrics.csv")
    per_round = pd.read_csv(root / "per_round/per_round_metrics.csv")
    ckks = pd.read_csv(root / "validation/ckks_validation.csv")

    expected_runs = _combinations(SEEDS, ATTACKS, PATHS)
    expected_manifest_runs = _combinations(SEEDS, ATTACKS, PATH_LABELS)
    run_columns = ["seed", "attack", "path"]
    manifest_columns = ["seed", "attack", "execution_path"]
    if len(manifest) != len(expected_manifest_runs) or _observed(manifest, manifest_columns) != expected_manifest_runs:
        raise ValueError("run manifest does not contain the complete seed/attack/path matrix")
    if not manifest["status"].eq("completed").all() or not manifest["rounds"].eq(30).all():
        raise ValueError("run manifest contains incomplete entries")

    if len(per_seed) != len(expected_runs) or _observed(per_seed, run_columns) != expected_runs:
        raise ValueError("per-seed metrics do not contain the complete run matrix")

    expected_rounds = _combinations(SEEDS, ATTACKS, PATHS, ROUNDS)
    round_columns = ["seed", "attack", "path", "round"]
    if len(per_round) != len(expected_rounds) or _observed(per_round, round_columns) != expected_rounds:
        raise ValueError("results do not contain the complete per-round matrix")

    expected_ckks = _combinations(SEEDS, ATTACKS, CKKS_PATHS)
    if len(ckks) != len(expected_ckks) or _observed(ckks, run_columns) != expected_ckks:
        raise ValueError("CKKS validation does not contain the complete encrypted-path matrix")
    if not ckks["real_ckks"].eq(True).all():
        raise ValueError("CKKS validation contains a non-CKKS run")

    return {
        "passed": True,
        "manifest_rows": len(manifest),
        "per_seed_rows": len(per_seed),
        "per_round_rows": len(per_round),
        "ckks_rows": len(ckks),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
    )
    arguments = parser.parse_args()
    print(json.dumps(validate_packaged_results(arguments.results_root), indent=2))


if __name__ == "__main__":
    main()
