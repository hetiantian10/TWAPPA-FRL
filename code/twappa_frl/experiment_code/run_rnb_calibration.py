"""Calibrate RNB strength on undefended ep_0 only."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

ALPHAS=(0.5,1.0,2.0,4.0,8.0)
SEEDS=(42,123,2025)
RUNNER=Path(__file__).with_name("run_20class_dapi.py")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--clean_root",default="results/aligned_benchmark_official_clean_20260717"); p.add_argument("--results_dir",default="results/rnb_calibration_official_5client"); p.add_argument("--skip_existing",action="store_true"); a=p.parse_args()
    root=Path(a.results_dir); root.mkdir(parents=True,exist_ok=True)
    clean=pd.read_csv(Path(a.clean_root)/"per_seed_final_metrics.csv"); clean=clean[clean.path=="fedavg"].set_index("seed")
    rows=[]
    candidates=[("RNB-Weak",0.0)]+[(f"alpha_{alpha:g}",alpha) for alpha in ALPHAS]
    for label,alpha in candidates:
      for seed in SEEDS:
        out=root/label/f"seed_{seed}"; history=out/"frl_20class_dapi_history.csv"
        if not (a.skip_existing and history.exists()):
          out.mkdir(parents=True,exist_ok=True)
          cmd=[sys.executable,str(RUNNER),"--data_dir","data/cicids_20client_4class","--results_dir",str(out),"--num_clients","5","--rounds","30","--local_episodes","1","--max_steps_per_episode","16","--batch_size","128","--supervised_aux_epochs","1","--execution_paths","fedavg","--adversarial_clients","4","--attack_mode","random_noise_byzantine","--expected_num_classes","4","--seed",str(seed),"--disable_privacy_intensity","--trust_smoothing","0.65","--trust_suppression_threshold","1e-8","--trust_weight_power","4","--trust_temperature","1","--trust_weight_normalization","power","--trust_max_weight","0.25","--no-aggregate_raw_updates","--enable_ckks","--initial_checkpoint",str(Path(a.clean_root)/"checkpoints"/f"initial_seed_{seed}.pt")]
          if alpha>0: cmd += ["--rnb_alpha",str(alpha)]
          subprocess.run(cmd,check=True,env={**os.environ,"PYTHONHASHSEED":str(seed),"MPLCONFIGDIR":str(root/".mplconfig")})
        h=pd.read_csv(history); ev=h[(h.phase=="eval")&(h.path=="fedavg")].sort_values("round"); final=ev.iloc[-1]
        rows.append({"condition":label,"alpha":alpha,"seed":seed,"clean_accuracy":clean.loc[seed].final_accuracy,"attacked_accuracy":final.eval_accuracy,"accuracy_degradation":clean.loc[seed].final_accuracy-final.eval_accuracy,"clean_macro_f1":clean.loc[seed].final_macro_f1,"attacked_macro_f1":final.eval_f1_macro,"macro_f1_degradation":clean.loc[seed].final_macro_f1-final.eval_f1_macro,"malicious_update_norm":ev.malicious_update_norm.mean() if "malicious_update_norm" in ev else np.nan,"median_benign_update_norm":ev.median_benign_update_norm.mean() if "median_benign_update_norm" in ev else np.nan,"malicious_to_benign_norm_ratio":ev.malicious_to_benign_norm_ratio.mean() if "malicious_to_benign_norm_ratio" in ev else np.nan,"has_nan_or_inf":bool(ev.rnb_has_nan_or_inf.fillna(0).astype(bool).any()) if "rnb_has_nan_or_inf" in ev else False,"training_stable":bool(np.isfinite(ev.eval_accuracy).all())})
        pd.DataFrame(rows).to_csv(root/"rnb_calibration_per_seed.csv.tmp",index=False); (root/"rnb_calibration_per_seed.csv.tmp").replace(root/"rnb_calibration_per_seed.csv")
    frame=pd.DataFrame(rows); summary=frame.groupby(["condition","alpha"],as_index=False).agg(mean_accuracy_degradation=("accuracy_degradation","mean"),std_accuracy_degradation=("accuracy_degradation","std"),min_accuracy_degradation=("accuracy_degradation","min"),mean_f1_degradation=("macro_f1_degradation","mean"),mean_attacked_accuracy=("attacked_accuracy","mean"),mean_norm_ratio=("malicious_to_benign_norm_ratio","mean"),any_nan_or_inf=("has_nan_or_inf","max"),all_stable=("training_stable","min"))
    summary.to_csv(root/"rnb_calibration_curve.csv",index=False)
    eligible=summary[(summary.alpha>0)&(summary.mean_accuracy_degradation>=0.05)&(summary.min_accuracy_degradation>0)&(~summary.any_nan_or_inf)&(summary.all_stable)&(summary.mean_attacked_accuracy>0.30)].sort_values("alpha")
    selected=None if eligible.empty else eligible.iloc[0].to_dict()
    (root/"rnb_primary_selection.json").write_text(json.dumps({"selection_rule":"lowest alpha with >=5pp mean degradation, positive degradation for every seed, finite stable training, and attacked accuracy >0.30","selected":selected},indent=2),encoding="utf-8")

if __name__=="__main__": main()
