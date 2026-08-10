# TWAPPA-FRL

TWAPPA-FRL is a trust-weighted, privacy-preserving aggregation framework for
federated reinforcement learning in collaborative intrusion detection. This
repository releases the manuscript sources, the aligned five-client benchmark
implementation, frozen experiment configuration, compact result evidence, and
validation tooling.

## Release contents

- `article/` — main manuscript, supplementary material, bibliography, figures,
  and LaTeX build configuration.
- `code/twappa_frl/experiment_code/` — self-contained benchmark runner, result
  collector, RNB calibration script, and packaged-result validator.
- `code/twappa_frl/config/` — frozen five-client benchmark configuration.
- `code/twappa_frl/results/` — compact CSV and JSON evidence for the official
  aligned benchmark.
- `tests/` — release-integrity and figure-generation tests.
- `Environment.yml` — pinned CPU-oriented Conda environment.
- `requirements.txt` — minimum dependency ranges for an existing Python
  environment.

Historical implementations, development notes, downloaded papers, raw
datasets, training checkpoints, and generated run directories are intentionally
excluded from the release tree. They are not required to inspect or validate
the packaged result matrices.

## Installation

The pinned Conda environment is the recommended reproduction environment. It
uses Python 3.10.14, CPU-only PyTorch 2.2.2, and TenSEAL 0.3.16.

```bash
git clone https://github.com/hetiantian10/TWAPPA-FRL.git
cd TWAPPA-FRL
conda env create -f Environment.yml
conda activate frl-ckks-repro
python -c "import pandas, torch, tenseal; print('environment imports passed')"
```

As a convenience for an existing Python 3.10 installation, a local virtual
environment can instead be created with:

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` specifies compatible minimum versions rather than the exact
frozen environment. Use `Environment.yml` when dependency-level reproducibility
is required. A TeX distribution with `latexmk` and the packages used by the
IEEEtran sources is additionally required to rebuild the manuscripts.

## Dataset

The experiments use a four-class subset of
[CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html): benign, DDoS,
port-scan, and FTP brute-force traffic represented by 78 numeric features. The
dataset is distributed by the Canadian Institute for Cybersecurity and is not
redistributed in this repository.

The aligned runner does not repartition records at run time. It expects five
pre-generated non-i.i.d. client partitions and one shared test set under a
user-supplied directory:

```text
client_0_train.csv
client_1_train.csv
client_2_train.csv
client_3_train.csv
client_4_train.csv
test.csv
meta.json                 # optional class-name metadata
```

Each CSV must contain a `Label` column and numeric feature columns. Exact
retraining of the reported experiment additionally requires files matching the
recorded hashes. The aggregate partition hash is
`33723ea59876dbcc6067be0b5a1b3736a3f16494f4814d52c2c06cbdd8cd8693`, and
the shared test-set hash is
`83231db890ee1b87f8d33e9c70d6a2f57690b3b2eb682b35aac6dbf45daaf065`.
The five individual partition hashes are recorded in
`code/twappa_frl/results/validation/final_results_validation.json`.

The exact fixed partitions are not redistributed. Consequently, the packaged
results can be audited without the dataset, while retraining requires the user
to obtain CIC-IDS2017 and prepare compatible files. A run with different file
hashes is a new reproduction run, not a bitwise recreation of the released
training trajectories.

## Validate the packaged evidence

Validation does not require the raw dataset or training checkpoints. From the
repository root, run:

```bash
python code/twappa_frl/experiment_code/validate_packaged_results.py
python -m pytest -q
```

The validator requires the complete official matrix and prints:

```json
{
  "passed": true,
  "manifest_rows": 126,
  "per_seed_rows": 126,
  "per_round_rows": 3780,
  "ckks_rows": 63
}
```

It checks all three seeds, seven conditions, six execution paths, and 30 rounds
per run, together with the 63 encrypted-path records marked as real CKKS. The
tests also confirm that incomplete round matrices are rejected and that the
published federation-sensitivity figure can be regenerated from its summarized
source values.

## Reproduce the benchmark

### One-seed smoke run

After preparing the dataset directory, the following command exercises all six
paths for one clean condition and one round:

```bash
python code/twappa_frl/experiment_code/run_aligned_benchmark.py \
  --data_dir /path/to/cicids_20client_4class \
  --results_dir results/smoke_seed42_clean \
  --seeds 42 \
  --attacks none \
  --rounds 1
```

This is an environment and integration check only; it does not reproduce a
reported result.

### Full aligned benchmark

The frozen default command evaluates seeds 42, 123, and 2025 under clean
training and six standalone attacks for all six execution paths:

```bash
python code/twappa_frl/experiment_code/run_aligned_benchmark.py \
  --data_dir data/cicids_20client_4class \
  --results_dir results/aligned_benchmark \
  --skip_existing
```

Using the frozen relative data path above preserves the configuration identity
recorded by the released benchmark; supplying another path intentionally
changes the serialized configuration hash.

The default schedule uses 30 federated rounds, one local episode capped at 16
steps, one auxiliary supervised epoch per round, and a batch size of 128. The
three PPA-enabled paths use real TenSEAL CKKS. `--skip_existing` preserves a
condition-level result only when all six paths contain exactly the requested
number of evaluation rounds; incomplete conditions are rerun.

The frozen RNB-Primary strength is `alpha = 0.5`. Its calibration evidence is
packaged under `code/twappa_frl/results/rnb_calibration/`, and the calibration
runner can be applied to the completed aligned-run root with:

```bash
python code/twappa_frl/experiment_code/run_rnb_calibration.py \
  --clean_root results/aligned_benchmark \
  --results_dir results/rnb_calibration \
  --skip_existing
```

This calibration command reuses the saved seed-specific initial checkpoints
from the aligned-run root and expects the dataset at the frozen relative path.

### Regenerate the sensitivity figure

The separate five-to-twenty-client sensitivity study is reported in
Supplementary Table S3 and Figure 3. Regenerate Figure 3 from the summarized
table values with:

```bash
python article/figures/generate_sensitivity_scalability_summary.py
```

This produces `article/figures/sensitivity_scalability_summary.png` and
`article/figures/sensitivity_scalability_summary.pdf`. The sensitivity study is
not part of the 126-run aligned five-client package and is not covered by
`validate_packaged_results.py`.

### Build the manuscripts

```bash
cd article
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

cd supplementary
latexmk -pdf -interaction=nonstopmode -halt-on-error supp.tex
```

The expected PDFs are `article/main.pdf` and
`article/supplementary/supp.pdf`.

## Expected benchmark outputs

A new aligned run writes the following structure beneath `--results_dir`:

```text
benchmark_config.yaml
fairness_validation.json
failed_runs.log
checkpoints/initial_seed_<seed>.pt
runs/seed_<seed>/<attack>/
  frl_20class_dapi_history.csv
  benchmark_summary_20class_dapi.csv
  overhead_20class_dapi.csv
  dapi_trust_20class_dapi.csv
  tensor_correctness_20class_dapi.csv
  run.log
  run.err.log
  wall_time_sec.txt
run_manifest.csv
per_seed_final_metrics.csv
per_round_metrics.csv
ckks_validation.csv
clean_global_utility_summary.csv
adversarial_robustness_summary.csv
suppression_summary.csv
runtime_summary.csv
table_attack_path_performance.tex
table_multiple_poisoning_results.tex
plots/
```

On successful completion of the full default matrix, `failed_runs.log` is
empty, `run_manifest.csv` contains 126 `completed` entries,
`per_seed_final_metrics.csv` contains 126 rows,
`per_round_metrics.csv` contains 3,780 rows, and `ckks_validation.csv` contains
63 rows. Generated checkpoints and run directories are ignored by Git; the
release retains only the compact evidence under `code/twappa_frl/results/`.

## Benchmark provenance

The official aligned benchmark was completed on 2026-07-17 and is identified
by the following frozen properties:

| Property | Released value |
| --- | --- |
| Configuration | `code/twappa_frl/config/benchmark_config.yaml` |
| Result package | `code/twappa_frl/results/` |
| Seeds | 42, 123, 2025 |
| Conditions | clean, sign flipping, RNB-Primary, model replacement, label flipping, data poisoning, reward poisoning |
| Execution paths | `ep_0`, `ep_sdp`, `ep_twa`, `ep_ppa`, `ep_sdp-ppa`, `ep_twa-ppa` |
| Matrix size | 3 seeds × 7 conditions × 6 paths = 126 runs |
| Training length | 30 rounds per run |
| Federation | five clients with full participation; clean uses five benign clients, attacks use four benign clients plus malicious client 4 |
| Encryption | real CKKS for the three PPA-enabled paths, 63 records total |
| Configuration hash | `22dc37846e3e0377e62421871dd7e22f1a7cf66b3f9f9ad5c3cf0380c7a18450` |
| Aggregate partition hash | `33723ea59876dbcc6067be0b5a1b3736a3f16494f4814d52c2c06cbdd8cd8693` |
| Test-set hash | `83231db890ee1b87f8d33e9c70d6a2f57690b3b2eb682b35aac6dbf45daaf065` |

Within each seed, every path and condition reused the same saved initial model
state; the corresponding checkpoint hashes are preserved in the run manifest
and final validation metadata. RNB-Primary uses `alpha = 0.5`, selected as the
lowest tested strength satisfying the recorded potency rule across all three
seeds. Lower-potency RNB trials are calibration evidence and are excluded from
the main matrix. Combined attacks are also excluded from this aligned
benchmark.

For metric definitions, table-source mappings, and the complete result-file
inventory, see `code/twappa_frl/README.md` and
`code/twappa_frl/results/validation/final_results_validation.json`.

## License and citation

The software is distributed under the BSD 3-Clause License in `LICENSE.txt`.
Citation metadata is provided in `CITATION.cff`. A Zenodo DOI is not required
to use or cite a tagged GitHub release.
