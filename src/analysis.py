"""
analysis.py

A  Predictive validity: Spearman correlation between each score and logistic-regression retention in the random
   (controlled-degradation) settings, pooled over tasks and within each task, with 95 percent bootstrap intervals
   (2,000 resamples of instances, stratified by task).
B  Learned TaskFit, leave-one-task-out: monotone gradient boosting from the seven dimension scores (and, as an
   extension, the dimension scores plus the pilot AUC) to retention, fitted on four tasks and evaluated on the fifth;
   compared with the pilot AUC alone by Spearman correlation and mean absolute error after a monotone (isotonic)
   mapping fitted on the same four tasks.
C  Detection of unfit training data: retention below 0.90; area under the ROC curve of each score.
D  Secondary outcomes: Spearman correlation of each score with expected calibration error and with the subgroup
   AUC of the protected group; correlation of q_rep with the subgroup AUC.
E  Diagnosis: among instances with exactly one active degradation, the share in which the lowest dimension score
   corresponds to the injected degradation (frac, pos_keep -> q_size; mcar -> q_comp; mnar -> q_mnar or q_comp;
   label -> q_label; sub_keep -> q_rep; selbias -> q_shift).
F  Blind spot: retention and TaskFit under measurement noise alone versus no degradation.
G  Real shift: in the shifted settings, Spearman correlation of each score with absolute target AUC, and the mean
   score in the shifted versus the random settings.
Outputs: outputs/tables/*.csv
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; TAB = OUT / "tables"; TAB.mkdir(exist_ok=True)
RNG = np.random.default_rng(2026)
DIMS = ["q_comp", "q_mnar", "q_size", "q_label", "q_rep", "q_shift", "q_time"]
SCORES = ["TaskFit_fixed", "TaskFit_learned", "B_profile", "B_generic", "B_size", "B_pilot"]
LVL = ["lv_frac", "lv_mcar", "lv_mnar", "lv_label", "lv_pos_keep", "lv_sub_keep", "lv_selbias", "lv_meas"]


def save(df, n):
    df.to_csv(TAB / f"{n}.csv", index=False); print(f"\n== {n}\n" + df.round(3).to_string(index=False), flush=True)


def boot_rho(x, y, strata, B=2000):
    idx_by = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    v = []
    for _ in range(B):
        ii = np.concatenate([RNG.choice(i, len(i)) for i in idx_by])
        v.append(spearmanr(x[ii], y[ii])[0])
    return np.percentile(v, [2.5, 97.5])


def loto(D, feats, target="lr_retention"):
    pred = pd.Series(np.nan, index=D.index)
    for t in D.task.unique():
        tr, te = D.task != t, D.task == t
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3, random_state=2026,
                                          monotonic_cst=[1] * len(feats))
        m.fit(D.loc[tr, feats], D.loc[tr, target]); pred[te] = m.predict(D.loc[te, feats])
    return pred


def main():
    D = pd.read_parquet(OUT / "instances.parquet")
    R = D[D.setting == "random"].reset_index(drop=True)
    R["TaskFit_learned"] = loto(R, DIMS)
    R["TaskFit_plus_pilot"] = loto(R, DIMS + ["B_pilot"])
    sc = SCORES + ["TaskFit_plus_pilot"]
    # A
    rows = []
    for s in sc:
        r = spearmanr(R[s], R.lr_retention)[0]; lo, hi = boot_rho(R[s].values, R.lr_retention.values, R.task.values)
        row = {"score": s, "rho_pooled": r, "ci_lo": lo, "ci_hi": hi}
        for t in R.task.unique():
            g = R[R.task == t]; row[f"rho_{t}"] = spearmanr(g[s], g.lr_retention)[0]
        rows.append(row)
    save(pd.DataFrame(rows), "A_validity")
    # B: monotone mapping and MAE, leave one task out
    rows = []
    for s in ["TaskFit_fixed", "B_pilot", "B_profile"]:
        pred = pd.Series(np.nan, index=R.index)
        for t in R.task.unique():
            tr, te = R.task != t, R.task == t
            iso = IsotonicRegression(out_of_bounds="clip").fit(R.loc[tr, s], R.loc[tr, "lr_retention"]); pred[te] = iso.predict(R.loc[te, s])
        rows.append({"predictor": s + " (isotonic)", "MAE": (pred - R.lr_retention).abs().mean()})
    for s in ["TaskFit_learned", "TaskFit_plus_pilot"]:
        rows.append({"predictor": s, "MAE": (R[s] - R.lr_retention).abs().mean()})
    rows.append({"predictor": "constant (mean retention)", "MAE": (R.lr_retention - R.lr_retention.mean()).abs().mean()})
    save(pd.DataFrame(rows), "B_mae")
    # C
    unfit = (R.lr_retention < 0.90).astype(int)
    rows = []
    for s in sc:
        a = roc_auc_score(unfit, -R[s]); bs = []
        for _ in range(2000):
            ii = RNG.integers(0, len(R), len(R))
            if unfit.iloc[ii].nunique() == 2:
                bs.append(roc_auc_score(unfit.iloc[ii], -R[s].iloc[ii]))
        rows.append({"score": s, "AUROC_unfit": a, "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5), "unfit_rate": unfit.mean()})
    save(pd.DataFrame(rows), "C_unfit_detection")
    # D
    rows = []
    for s in sc + ["q_rep"]:
        rows.append({"score": s, "rho_ece": spearmanr(R[s], R.lr_ece)[0], "rho_auc_subgroup": spearmanr(R[s], R.lr_auc_sub, nan_policy="omit")[0],
                     "rho_gb_retention": spearmanr(R[s], R.gb_retention)[0]})
    save(pd.DataFrame(rows), "D_secondary")
    # E diagnosis
    active = pd.DataFrame({"frac": R.lv_frac < 1, "mcar": R.lv_mcar > 0, "mnar": R.lv_mnar > 0, "label": R.lv_label > 0,
                           "pos_keep": R.lv_pos_keep < 1, "sub_keep": R.lv_sub_keep < 1, "selbias": R.lv_selbias > 0, "meas": R.lv_meas > 0})
    target = {"frac": ["q_size"], "pos_keep": ["q_size"], "mcar": ["q_comp"], "mnar": ["q_mnar", "q_comp"], "label": ["q_label"],
              "sub_keep": ["q_rep"], "selbias": ["q_shift"]}
    single = active.sum(1) == 1
    rows = []
    for k, tg in target.items():
        sel = single & active[k]
        if sel.sum() == 0:
            continue
        lowest = R.loc[sel, DIMS].idxmin(axis=1)
        rows.append({"degradation": k, "instances": int(sel.sum()), "lowest_dimension_correct": lowest.isin(tg).mean(),
                     "most_common_lowest": lowest.mode().iloc[0]})
    save(pd.DataFrame(rows), "E_diagnosis")
    # F blind spot
    none = active.sum(1) == 0; meas_only = single & active["meas"]
    save(pd.DataFrame([{"condition": "no degradation", "n": int(none.sum()), "retention": R.loc[none, "lr_retention"].mean(), "TaskFit": R.loc[none, "TaskFit_fixed"].mean(), "pilot": R.loc[none, "B_pilot"].mean()},
                       {"condition": "measurement noise only", "n": int(meas_only.sum()), "retention": R.loc[meas_only, "lr_retention"].mean(), "TaskFit": R.loc[meas_only, "TaskFit_fixed"].mean(), "pilot": R.loc[meas_only, "B_pilot"].mean()}]), "F_blindspot")
    # G real shift
    S = D[D.setting != "random"].copy()
    rows = []
    for s in ["TaskFit_fixed", "q_shift", "B_profile", "B_generic", "B_size", "B_pilot"]:
        rows.append({"score": s, "rho_target_auc_shifted": spearmanr(S[s], S.lr_auc)[0],
                     "mean_shifted": S[s].mean(), "mean_random": D.loc[D.setting == "random", s].mean()})
    save(pd.DataFrame(rows), "G_real_shift")
    gs = D.groupby(["task", "setting"]).agg(target_auc=("lr_auc", "mean"), ref_auc=("lr_ref_auc", "first"), TaskFit=("TaskFit_fixed", "mean"),
                                           q_shift=("q_shift", "mean"), pilot=("B_pilot", "mean")).reset_index()
    save(gs, "G_settings")
    R.to_parquet(OUT / "random_with_learned.parquet")


if __name__ == "__main__":
    main()


def extended():
    """H: learned TaskFit on real shifts; I: single-factor dose-response and diagnosis; J: label-estimator sensitivity."""
    D = pd.read_parquet(OUT / "instances.parquet")
    R = D[D.setting == "random"].reset_index(drop=True); S = D[D.setting != "random"].reset_index(drop=True)
    # H: train the learned map on the random settings of the other tasks, apply to the shifted settings of the held-out task
    rows = []
    S["TaskFit_learned"] = np.nan; S["TaskFit_plus_pilot"] = np.nan
    for t in S.task.unique():
        for col, feats in [("TaskFit_learned", DIMS), ("TaskFit_plus_pilot", DIMS + ["B_pilot"])]:
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3, random_state=2026, monotonic_cst=[1] * len(feats))
            m.fit(R.loc[R.task != t, feats], R.loc[R.task != t, "lr_retention"]); S.loc[S.task == t, col] = m.predict(S.loc[S.task == t, feats])
    for s in ["TaskFit_learned", "TaskFit_plus_pilot", "TaskFit_fixed", "q_shift", "B_profile", "B_generic", "B_size", "B_pilot"]:
        lo, hi = boot_rho(S[s].values, S.lr_auc.values, S.task.values)
        rows.append({"score": s, "rho_target_auc": spearmanr(S[s], S.lr_auc)[0], "ci_lo": lo, "ci_hi": hi})
    save(pd.DataFrame(rows), "H_real_shift_learned")
    # setting-level: does each score rank the 13 settings by mean target AUC?
    allD = pd.concat([R.assign(TaskFit_learned=loto(R, DIMS)), S])
    G = allD.groupby(["task", "setting"]).agg(auc=("lr_auc", "mean"), **{s: (s, "mean") for s in ["TaskFit_learned", "TaskFit_fixed", "q_shift", "B_pilot", "B_size"]}).reset_index()
    save(pd.DataFrame([{"score": s, "rho_setting_level": spearmanr(G[s], G.auc)[0], "n_settings": len(G)} for s in ["TaskFit_learned", "TaskFit_fixed", "q_shift", "B_pilot", "B_size"]]), "H_setting_level")
    # I single-factor
    Q = pd.read_parquet(OUT / "instances_single.parquet")
    match = {"frac": "q_size", "pos_keep": "q_size", "mcar": "q_comp", "mnar": "q_mnar", "label": "q_label", "sub_keep": "q_rep", "selbias": "q_shift", "meas": None}
    dr = Q.groupby(["factor", "level"]).agg(retention=("lr_retention", "mean"), TaskFit=("TaskFit_fixed", "mean"), pilot=("B_pilot", "mean"),
                                            **{d: (d, "mean") for d in DIMS}).reset_index()
    save(dr, "I_dose_response")
    rows = []
    for k, d in match.items():
        g = Q[Q.factor == k]
        row = {"factor": k, "matched_dimension": d, "n": len(g),
               "rho_level_retention": spearmanr(g.level, g.lr_retention)[0],
               "rho_level_matched_dim": spearmanr(g.level, g[d])[0] if d else np.nan,
               "lowest_dim_correct": (g[DIMS].idxmin(axis=1) == d).mean() if d else np.nan,
               "matched_in_lowest_two": g[DIMS].apply(lambda r: d in r.nsmallest(2).index, axis=1).mean() if d else np.nan}
        rows.append(row)
    save(pd.DataFrame(rows), "I_diagnosis")
    # J label estimator sensitivity under label noise only
    L = Q[Q.factor == "label"]
    save(pd.DataFrame([{"estimator": "anchor point (q_label)", "rho_with_noise_level": spearmanr(L.level, L.q_label)[0], "mean_clean_baseline": R.loc[(R.lv_label == 0), "q_label"].mean()},
                       {"estimator": "confident learning", "rho_with_noise_level": spearmanr(L.level, L._q_label_cl)[0], "mean_clean_baseline": R.loc[(R.lv_label == 0), "_q_label_cl"].mean()}]), "J_label_estimators")


if __name__ == "__main__":
    extended()


def final_tests():
    """K: paired bootstrap of the Spearman difference (TaskFit + pilot minus pilot) in the random settings.
    L: within-task real-shift analysis: change from the random setting to each shifted setting in mean target AUC and in
       mean scores; Spearman correlation across the eight shifted settings, with a permutation p-value."""
    R = pd.read_parquet(OUT / "random_with_learned.parquet")
    a, b, y, st = R.TaskFit_plus_pilot.values, R.B_pilot.values, R.lr_retention.values, R.task.values
    idx_by = [np.flatnonzero(st == s) for s in np.unique(st)]; d = []
    for _ in range(2000):
        ii = np.concatenate([RNG.choice(i, len(i)) for i in idx_by])
        d.append(spearmanr(a[ii], y[ii])[0] - spearmanr(b[ii], y[ii])[0])
    d0 = spearmanr(a, y)[0] - spearmanr(b, y)[0]
    save(pd.DataFrame([{"comparison": "TaskFit+pilot minus pilot (rho with retention)", "difference": d0,
                        "ci_lo": np.percentile(d, 2.5), "ci_hi": np.percentile(d, 97.5), "p_one_sided": np.mean(np.array(d) <= 0)}]), "K_paired")
    D = pd.read_parquet(OUT / "instances.parquet")
    G = D.groupby(["task", "setting"]).agg(auc=("lr_auc", "mean"), TaskFit=("TaskFit_fixed", "mean"), q_shift=("q_shift", "mean"), pilot=("B_pilot", "mean")).reset_index()
    base = G[G.setting == "random"].set_index("task")
    S = G[G.setting != "random"].copy()
    for c in ["auc", "TaskFit", "q_shift", "pilot"]:
        S["d_" + c] = S[c].values - base.loc[S.task, c].values
    rows = []
    for c in ["TaskFit", "q_shift", "pilot"]:
        r = spearmanr(S["d_" + c], S.d_auc)[0]
        perm = [spearmanr(RNG.permutation(S["d_" + c].values), S.d_auc)[0] for _ in range(20000)]
        rows.append({"score": c, "rho_change_vs_auc_change": r, "p_permutation_one_sided": np.mean(np.array(perm) >= r), "n_shifted_settings": len(S)})
    save(S[["task", "setting", "d_auc", "d_TaskFit", "d_q_shift", "d_pilot"]], "L_shift_changes")
    save(pd.DataFrame(rows), "L_shift_change_tests")


if __name__ == "__main__":
    final_tests()
