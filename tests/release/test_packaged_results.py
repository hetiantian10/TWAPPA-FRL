from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pandas as pd
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_ROOT = REPOSITORY_ROOT / "code" / "twappa_frl"
VALIDATOR_PATH = BUNDLE_ROOT / "experiment_code" / "validate_packaged_results.py"


def load_validator():
    assert VALIDATOR_PATH.exists(), "the packaged-results validator is missing"
    specification = importlib.util.spec_from_file_location(
        "validate_packaged_results", VALIDATOR_PATH
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_official_packaged_results_pass_release_validation() -> None:
    validator = load_validator()

    report = validator.validate_packaged_results(BUNDLE_ROOT / "results")

    assert report == {
        "passed": True,
        "manifest_rows": 126,
        "per_seed_rows": 126,
        "per_round_rows": 3780,
        "ckks_rows": 63,
    }


def test_release_validation_rejects_an_incomplete_round_matrix(
    tmp_path: Path,
) -> None:
    validator = load_validator()
    source = BUNDLE_ROOT / "results"
    for relative_path in (
        Path("per_seed/run_manifest.csv"),
        Path("per_seed/per_seed_final_metrics.csv"),
        Path("per_round/per_round_metrics.csv"),
        Path("validation/ckks_validation.csv"),
    ):
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative_path, destination)

    rounds_path = tmp_path / "per_round/per_round_metrics.csv"
    rounds = pd.read_csv(rounds_path)
    rounds.iloc[:-1].to_csv(rounds_path, index=False)

    with pytest.raises(ValueError, match="complete per-round matrix"):
        validator.validate_packaged_results(tmp_path)
