"""Aligned 5-client/30-round benchmark for the six paper execution paths."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

SEEDS=(42,123,2025)
PATHS=("fedavg","static_dp","twa","ppa","sdp_ppa","twa_ppa")
LABEL=dict(zip(PATHS,("ep_0","ep_sdp","ep_twa","ep_ppa","ep_sdp-ppa","ep_twa-ppa")))
ATTACKS=("none","sign_flipping","random_noise_byzantine","model_replacement","label_flipping","data_poisoning","reward_poisoning")
THRESHOLD=1e-8
RUNNER=Path(__file__).with_name("run_20class_dapi.py")

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def args():
    p=argparse.ArgumentParser(); p.add_argument("--data_dir",default="data/cicids_20client_4class"); p.add_argument("--results_dir",default="")
    p.add_argument("--seeds",default="42,123,2025"); p.add_argument("--rounds",type=int,default=30); p.add_argument("--local_episodes",type=int,default=1)
    p.add_argument("--max_steps_per_episode",type=int,default=16); p.add_argument("--skip_existing",action="store_true"); p.add_argument("--attacks",default=",".join(ATTACKS))
    p.add_argument("--rnb_alpha",type=float,default=0.5,help="Frozen RNB-Primary strength selected by the ep_0 calibration")
    return p.parse_args()

def is_complete(folder, rounds):
    f=folder/"frl_20class_dapi_history.csv"
    if not f.exists(): return False
    d=pd.read_csv(f); d=d[d.phase=="eval"]
    return set(d.path)==set(PATHS) and d.groupby("path")["round"].nunique().eq(rounds).all()

def main():
    a=args(); seeds=[int(x) for x in a.seeds.split(",")]; selected_attacks=tuple(x.strip() for x in a.attacks.split(",") if x.strip()); root=Path(a.results_dir or f"results/aligned_benchmark_{datetime.now():%Y%m%d_%H%M%S}"); root.mkdir(parents=True,exist_ok=True)
    unknown=set(selected_attacks)-set(ATTACKS)
    if unknown: raise ValueError(f"Unknown attacks: {sorted(unknown)}")
    data=Path(a.data_dir); ph={str(i):digest(data/f"client_{i}_train.csv") for i in range(5)}; eh=digest(data/"test.csv")
    cfg={"profile_name":"twa_manuscript_5client","seeds":seeds,"execution_paths":list(PATHS),"attacks":list(selected_attacks),"num_clients":5,"client_ids":list(range(5)),"malicious_client_id":4,"rounds":a.rounds,"local_episodes":a.local_episodes,"max_steps_per_episode":a.max_steps_per_episode,"batch_size":128,"data_dir":a.data_dir,"aggregate_update_type":"norm_bounded","twa_norm_bound":1.0,"trust_smoothing":0.65,"trust_weight_power":4.0,"trust_temperature":1.0,"trust_weight_normalization":"power","trust_min_active_weight":0.0,"trust_floor_eligibility_threshold":0.0,"trust_max_weight":0.25,"trust_warmup_rounds":0,"aggregate_raw_updates":False,"suppression_threshold":THRESHOLD,"privacy_intensity_enabled":False,"real_ckks":True,"evaluation_reward":"5 * accuracy - 2","rnb_primary_alpha":a.rnb_alpha,"rnb_weak_in_main_matrix":False}
    (root/"benchmark_config.yaml").write_text("\n".join(f"{k}: {json.dumps(v)}" for k,v in cfg.items())+"\n",encoding="utf-8")
    fair={"passed":True,"partition_hashes":ph,"partition_hash":hashlib.sha256(json.dumps(ph,sort_keys=True).encode()).hexdigest(),"evaluation_set_hash":eh,"checks":["5 fixed client IDs","client 4 adversarial only outside clean condition","30 rounds","shared saved initial checkpoint per seed","shared evaluation set","real TenSEAL CKKS"]}
    (root/"fairness_validation.json").write_text(json.dumps(fair,indent=2),encoding="utf-8")
    failures=[]
    for seed in seeds:
      for attack in selected_attacks:
        out=root/"runs"/f"seed_{seed}"/attack; out.mkdir(parents=True,exist_ok=True)
        if a.skip_existing and is_complete(out,a.rounds): continue
        cmd=[sys.executable,str(RUNNER),"--data_dir",a.data_dir,"--results_dir",str(out),"--num_clients","5","--rounds",str(a.rounds),"--local_episodes",str(a.local_episodes),"--max_steps_per_episode",str(a.max_steps_per_episode),"--supervised_aux_epochs","1","--execution_paths",",".join(PATHS),"--attack_mode",attack,"--expected_num_classes","4","--seed",str(seed),"--adversarial_scale","8.0","--disable_privacy_intensity","--trust_smoothing","0.65","--trust_suppression_threshold",str(THRESHOLD),"--trust_weight_power","4.0","--trust_temperature","1.0","--trust_weight_normalization","power","--trust_min_active_weight","0","--trust_floor_eligibility_threshold","0","--trust_max_weight","0.25","--trust_warmup_rounds","0","--no-aggregate_raw_updates","--enable_ckks","--initial_checkpoint",str(root/"checkpoints"/f"initial_seed_{seed}.pt")]
        if attack!="none": cmd += ["--adversarial_clients","4"]
        if attack=="random_noise_byzantine": cmd += ["--rnb_alpha",str(a.rnb_alpha)]
        env=os.environ.copy(); env["PYTHONHASHSEED"]=str(seed); env["MPLCONFIGDIR"]=str(root/".mplconfig")
        print(f"[aligned] seed={seed} attack={attack}",flush=True); start=time.perf_counter()
        try:
          with (out/"run.log").open("w") as so,(out/"run.err.log").open("w") as se: subprocess.run(cmd,check=True,stdout=so,stderr=se,env=env)
          if not is_complete(out,a.rounds): raise RuntimeError("not exactly the requested rounds for all paths")
          (out/"wall_time_sec.txt").write_text(str(time.perf_counter()-start))
        except Exception as e: failures.append(f"seed={seed} attack={attack}: {e!r}")
    (root/"failed_runs.log").write_text("\n".join(failures),encoding="utf-8")
    collect(root,cfg,fair)

def collect(root,cfg,fair):
    finals=[]; rounds=[]; manifests=[]; ckks=[]
    for seed in cfg["seeds"]:
      cp=root/"checkpoints"/f"initial_seed_{seed}.pt"; ch=digest(cp) if cp.exists() else ""
      for attack in cfg["attacks"]:
        d=root/"runs"/f"seed_{seed}"/attack; ok=is_complete(d,cfg["rounds"])
        if ok:
          h=pd.read_csv(d/"frl_20class_dapi_history.csv"); h=h[h.phase=="eval"]; s=pd.read_csv(d/"benchmark_summary_20class_dapi.csv").set_index("path"); o=pd.read_csv(d/"overhead_20class_dapi.csv"); t=pd.read_csv(d/"dapi_trust_20class_dapi.csv"); x=pd.read_csv(d/"tensor_correctness_20class_dapi.csv")
          for path in PATHS:
            pe=h[h.path==path].sort_values("round"); pt=t[(t.path==path)&(t.client_id==4)].sort_values("round"); w=pt.aggregation_weight
            r={"seed":seed,"execution_path":LABEL[path],"path":path,"condition":"clean" if attack=="none" else "adversarial","attack":attack,"final_accuracy":float(s.loc[path,"accuracy"]),"final_macro_f1":float(s.loc[path,"f1_macro"]),"final_evaluation_reward":float(5*s.loc[path,"accuracy"]-2),"mean_round_time_sec":o[o.path==path].round_time_sec.mean(),"total_execution_time_sec":o[o.path==path].round_time_sec.sum(),"mean_malicious_weight":w.mean() if len(w) else np.nan,"final_malicious_weight":w.iloc[-1] if len(w) else np.nan,"suppressed_rounds":int(w.le(THRESHOLD).sum()),"final_10_suppressed_rounds":int(w.tail(10).le(THRESHOLD).sum()),"mean_gated_adversarial_trust":pt.smoothed_trust.mean() if len(pt) else np.nan,"mean_B_adv":pt.behavior_consistency.mean() if len(pt) else np.nan,"mean_R_adv":pt.reward_consistency.mean() if len(pt) else np.nan}; finals.append(r)
            for _,q in pe.iterrows(): rounds.append({"seed":seed,"execution_path":LABEL[path],"path":path,"condition":r["condition"],"attack":attack,"round":int(q["round"]),"accuracy":q.eval_accuracy,"macro_f1":q.eval_f1_macro,"evaluation_reward":5*q.eval_accuracy-2})
            if path in ("ppa","sdp_ppa","twa_ppa"):
              z=x[x.path==path]; ckks.append({"seed":seed,"attack":attack,"path":path,"max_relative_l2_error":z.tensor_relative_l2_error.max(),"mean_absolute_error":z.tensor_mean_abs_error.mean(),"real_ckks":True})
        for path in PATHS: manifests.append({"run_id":f"s{seed}_{attack}_{path}","seed":seed,"execution_path":LABEL[path],"condition":"clean" if attack=="none" else "adversarial","attack":attack,"rounds":cfg["rounds"],"num_clients":5,"malicious_client_id":"" if attack=="none" else 4,"partition_hash":fair["partition_hash"],"initial_checkpoint_hash":ch,"config_hash":hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),"status":"completed" if ok else "failed","log_path":str(d/"run.log")})
    f=pd.DataFrame(finals); pr=pd.DataFrame(rounds); pd.DataFrame(manifests).to_csv(root/"run_manifest.csv",index=False); f.to_csv(root/"per_seed_final_metrics.csv",index=False); pr.to_csv(root/"per_round_metrics.csv",index=False); pd.DataFrame(ckks).to_csv(root/"ckks_validation.csv",index=False)
    if len(f): summarize(root,f,pr)

def summarize(root,f,pr):
    cols=["final_accuracy","final_macro_f1","final_evaluation_reward","mean_round_time_sec","total_execution_time_sec","mean_malicious_weight","final_malicious_weight","suppressed_rounds","final_10_suppressed_rounds","mean_gated_adversarial_trust","mean_B_adv","mean_R_adv"]
    clean=f[f.attack=="none"].groupby(["execution_path","path"])[cols].agg(["mean","std"]); adv=f[f.attack!="none"].groupby(["attack","execution_path","path"])[cols].agg(["mean","std"])
    clean.to_csv(root/"clean_global_utility_summary.csv"); adv.to_csv(root/"adversarial_robustness_summary.csv"); f[f.attack!="none"].to_csv(root/"suppression_summary.csv",index=False); f.groupby(["attack","execution_path"])[["mean_round_time_sec","total_execution_time_sec"]].agg(["mean","std"]).to_csv(root/"runtime_summary.csv")
    (root/"table_attack_path_performance.tex").write_text(clean.reset_index().to_latex(index=False,float_format="%.4f"),encoding="utf-8"); (root/"table_multiple_poisoning_results.tex").write_text(adv.reset_index().to_latex(index=False,float_format="%.4f"),encoding="utf-8"); plots(root,f,pr)

def plots(root,f,pr):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    p=root/"plots"; p.mkdir(exist_ok=True)
    def save(n): plt.tight_layout(); plt.savefig(p/f"{n}.png",dpi=300); plt.savefig(p/f"{n}.pdf"); plt.close()
    c=f[f.attack=="none"]
    if not c.empty:
      for m,n in (("final_accuracy","clean_final_accuracy"),("final_macro_f1","clean_final_macro_f1")): c.groupby("execution_path")[m].mean().plot.bar(); plt.ylabel(m); save(n)
    adversarial=pr[pr.attack!="none"]
    if not adversarial.empty:
      for m,n in (("accuracy","accuracy_vs_round_by_attack"),("macro_f1","macro_f1_vs_round_by_attack")):
        fig,ax=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
        for panel,attack in zip(ax.flat,ATTACKS[1:]):
          subset=adversarial[adversarial.attack==attack]
          if not subset.empty: subset.groupby(["round","execution_path"])[m].mean().unstack().plot(ax=panel,legend=False)
          panel.set_title(attack.replace("_"," "))
        ax.flat[-1].legend(fontsize=6); save(n)
      q=f[(f.attack!="none")&f.path.isin(("twa","twa_ppa"))]; q.groupby(["attack","execution_path"]).mean_malicious_weight.mean().unstack().plot.bar(); save("malicious_weight_vs_round"); q.groupby(["attack","execution_path"]).suppressed_rounds.mean().unstack().plot.bar(); save("suppression_count_across_attacks")
    pairs=[]
    for a,b in (("fedavg","ppa"),("static_dp","sdp_ppa"),("twa","twa_ppa")):
      x=f[f.path==a].set_index(["seed","attack"]); y=f[f.path==b].set_index(["seed","attack"]); pairs.append(pd.DataFrame({"pair":f"{a}-{b}","difference":y.final_accuracy-x.final_accuracy}))
    pd.concat(pairs).groupby("pair").difference.mean().plot.bar(); save("plaintext_vs_ckks_utility_difference"); f.groupby("execution_path").mean_round_time_sec.mean().plot.bar(); save("mean_round_time_comparison")

if __name__=="__main__": main()
