# Aligned Clean and Adversarial FRL Experiments

## Scope

This folder contains the code and CSV outputs for the official three-seed aligned benchmark used for the clean and six standalone attack evaluations completed on 2026-07-17. All six execution paths (`ep_0`, `ep_sdp`, `ep_twa`, `ep_ppa`, `ep_sdp-ppa`, and `ep_twa-ppa`) were run. Table II focuses on TWA and TWA-PPA, while Table III compares all six paths. Combined attacks were not included.

## Experimental configuration

- Seeds: 42, 123, and 2025.
- Clients: 5 with full participation.
- Clean condition: 5 benign clients.
- Adversarial conditions: 4 benign clients and malicious client 4.
- Federated rounds: 30.
- Local episodes: 1.
- Maximum local steps: 16.
- Batch size: 128.
- PPA paths use real TenSEAL CKKS.
- TWA uses norm-bounded aggregation with a norm bound of 1.0.
- Adaptive privacy intensity is disabled.

## Attacks

- Sign flipping.
- RNB-Primary (`alpha = 0.5`).
- Model replacement.
- Label flipping.
- Data poisoning.
- Reward poisoning.

Low-potency RNB conditions were calibration-only. RNB-Primary uses `alpha = 0.5`, the lowest tested strength satisfying the potency criterion across all three seeds. The same RNB condition was applied to all six execution paths.

## Result files

- `results/summary/table_II_attack_path_performance.csv`: all six execution paths under clean and the six standalone attacks; this is the all-path utility source used by the official manuscript tables.
- `results/summary/table_III_multiple_poisoning_results.csv`: TWA and TWA-PPA attack utility and malicious-client influence results.
- `results/per_seed/table_II_per_seed_source.csv`: seed-level source records underlying the all-path attack-performance summary.
- `results/per_seed/table_III_per_seed_source.csv`: seed-level TWA and TWA-PPA utility and influence records.
- `results/per_seed/per_seed_final_metrics.csv`: final-round metrics for every seed, condition, and execution path.
- `results/per_seed/run_manifest.csv`: the 126-run completion manifest for three seeds, seven conditions, and six paths.
- `results/per_round/per_round_metrics.csv`: round-level metrics from the official aligned benchmark.
- `results/rnb_calibration/rnb_calibration_curve.csv`: aggregate RNB potency results across tested strengths.
- `results/rnb_calibration/rnb_calibration_per_seed.csv`: seed-level RNB calibration measurements.
- `results/validation/ckks_validation.csv`: CKKS correctness checks from the official benchmark.
- `results/validation/lockstep_ckks_summary.csv`: mechanism-validation evidence for CKKS numerical fidelity. It was not used to calculate Table II or Table III utility results.

The JSON files alongside these CSVs record the RNB-Primary selection, fairness validation, and final-results validation metadata. `experiment_code/generate_final_tables.py` is the exact official benchmark collector used to generate the final table CSVs.

## Main result summary

- Clean TWA accuracy: 0.967444 ± 0.003657.
- Clean TWA-PPA accuracy: 0.967778 ± 0.003289.
- Attack accuracy for TWA/TWA-PPA: approximately 0.9681–0.9737.
- Hard-zero criterion: `w_adv <= 1e-8`.
- Hard-zero count: 0/30 for all attacks.

These results support utility preservation and malicious-client influence control, not complete malicious-client exclusion.

## Validate the packaged results

From the repository root:

```bash
python code/twappa_frl/experiment_code/validate_packaged_results.py
```

This checks the complete 126-run manifest, the corresponding final and
round-wise matrices, and the 63 real-CKKS encrypted-path records. It does not
require raw data or checkpoints.

## Run the benchmark

The runner and its `CORE` dependencies are contained in `experiment_code/`.
After preparing the five client partitions described in the root README, run:

```bash
python code/twappa_frl/experiment_code/run_aligned_benchmark.py \
  --data_dir /path/to/cicids_20client_4class \
  --results_dir results/aligned_benchmark
```

Add `--skip_existing` to retain completed condition-level runs. Generated
checkpoints and run directories are excluded from Git.

## Reproduction boundary

The official compact CSV evidence, frozen configuration, runner, and validation
tools are included. CIC-IDS2017 data, the fixed pre-generated partitions, and
training checkpoints are not redistributed. Consequently, the packaged
evidence can be audited directly, while retraining requires compatible
user-prepared data. The fixed partition and test-set hashes are preserved in
`results/validation/final_results_validation.json`.
