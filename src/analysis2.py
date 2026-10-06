"""
analysis2.py: analyses added in response to review.

M  Aggregation of the seven dimensions: arithmetic, geometric (TaskFit), harmonic, and minimum; Spearman correlation with
   retention in the random settings (bootstrap 95 percent intervals) and with the change in target AUC across the eight real
   shifts (one-sided permutation p-values).
N  Label-quality estimators in the label-noise sweeps: anchor-point estimates at the 90th, 95th, 97.5th, 99th, and 100th
   percentiles and confident learning. Each implies a noise estimate e_hat = 1 - q; reported are the Spearman correlation with
   the true injected rate, and the bias and mean absolute error of the baseline-corrected estimate
   e_hat - e_hat(clean), where e_hat(clean) is the mean estimate of the same task without injected noise.
O  The METRIC-inspired model-independent baseline against TaskFit, the pilot model, and their combination: Spearman correlation
   with retention, unfit-data AUC, and the correlation of its change with the change in target AUC under real shift.
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from analysis import boot_rho, loto, save, DIMS, RNG, OUT

AGG = {"arithmetic": lambda V: V.mean(1), "geometric (TaskFit)": lambda V: np.exp(np.log(np.clip(V, 1e-3, 1)).mean(1)),
       "harmonic": lambda V: 1 / (1 / np.clip(V, 1e-3, 1)).mean(1), "minimum": lambda V: V.min(1)}


def perm_p(x, y, B=20000):
    r = spearmanr(x, y)[0]
    return r, np.mean([spearmanr(RNG.permutation(x), y)[0] >= r for _ in range(B)])


def shift_changes(D, cols):
    G = D.groupby(["task", "setting"])[["lr_auc"] + cols].mean().reset_index()
    b = G[G.setting == "random"].set_index("task"); S = G[G.setting != "random"].copy()
    for c in ["lr_auc"] + cols:
        S["d_" + c] = S[c].values - b.loc[S.task, c].values
    return S


def main():
    D = pd.read_parquet(OUT / "instances.parquet")
    for name, f in AGG.items():
        D["agg_" + name] = f(D[DIMS].values)
    R = D[D.setting == "random"].reset_index(drop=True)
    cols = ["agg_" + n for n in AGG]
    S = shift_changes(D, cols)
    rows = []
    for n in AGG:
        c = "agg_" + n; lo, hi = boot_rho(R[c].values, R.lr_retention.values, R.task.values)
        r_s, p_s = perm_p(S["d_" + c].values, S.d_lr_auc.values)
        rows.append({"aggregation": n, "rho_retention": spearmanr(R[c], R.lr_retention)[0], "ci_lo": lo, "ci_hi": hi,
                     "unfit_auc": roc_auc_score((R.lr_retention < 0.9).astype(int), -R[c]), "rho_shift_change": r_s, "p_shift": p_s})
    save(pd.DataFrame(rows), "M_aggregation")
    Q = pd.read_parquet(OUT / "instances_single.parquet"); L = Q[Q.factor == "label"]
    clean = D[(D.setting == "random") & (D.lv_label == 0)]
    est = {"anchor p90": "_q_label_p900", "anchor p95": "_q_label_p950", "anchor p97.5 (TaskFit)": "q_label", "anchor p99": "_q_label_p990",
           "anchor max": "_q_label_p1000", "confident learning": "_q_label_cl"}
    rows = []
    for n, c in est.items():
        e = 1 - L[c].values; base = L.task.map(1 - clean.groupby("task")[c].mean()).values
        corr = e - base
        rows.append({"estimator": n, "rho_true_rate": spearmanr(L.true_label_noise, e)[0],
                     "bias_corrected": np.mean(corr - L.true_label_noise), "mae_corrected": np.mean(np.abs(corr - L.true_label_noise)),
                     "mean_clean_estimate": float(np.mean(1 - clean[c]))})
    save(pd.DataFrame(rows), "N_label_estimators")
    R["TaskFit_plus_pilot"] = loto(R, DIMS + ["B_pilot"])
    S2 = shift_changes(D, ["TaskFit_fixed", "B_metric", "B_pilot", "B_generic"])
    rows = []
    for c in ["B_generic", "B_metric", "TaskFit_fixed", "B_pilot", "TaskFit_plus_pilot"]:
        lo, hi = boot_rho(R[c].values, R.lr_retention.values, R.task.values)
        row = {"score": c, "rho_retention": spearmanr(R[c], R.lr_retention)[0], "ci_lo": lo, "ci_hi": hi,
               "unfit_auc": roc_auc_score((R.lr_retention < 0.9).astype(int), -R[c])}
        if "d_" + c in S2:
            row["rho_shift_change"], row["p_shift"] = perm_p(S2["d_" + c].values, S2.d_lr_auc.values)
        rows.append(row)
    save(pd.DataFrame(rows), "O_metric_baseline")
    # paired: TaskFit minus METRIC-inspired baseline
    a, b, y, st = R.TaskFit_fixed.values, R.B_metric.values, R.lr_retention.values, R.task.values
    idx = [np.flatnonzero(st == s) for s in np.unique(st)]; d = []
    for _ in range(2000):
        ii = np.concatenate([RNG.choice(i, len(i)) for i in idx]); d.append(spearmanr(a[ii], y[ii])[0] - spearmanr(b[ii], y[ii])[0])
    save(pd.DataFrame([{"comparison": "TaskFit minus METRIC-inspired (rho with retention)", "difference": spearmanr(a, y)[0] - spearmanr(b, y)[0],
                        "ci_lo": np.percentile(d, 2.5), "ci_hi": np.percentile(d, 97.5)}]), "O_paired")


if __name__ == "__main__":
    main()
