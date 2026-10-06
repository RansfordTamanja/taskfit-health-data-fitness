"""
taskfit.py

TaskFit: task-aware fitness of a training dataset D = (X, y, s) for a prediction task deployed on a
target population with unlabelled covariates X_T and expected subgroup share pi_T.

Dimension scores (each in [0, 1], higher = fitter):
  q_comp   task-weighted completeness   sum_j w_j (1 - m_j), w_j proportional to the task relevance
           |rho_S(x_j, y)| of feature j on observed values (Spearman), normalised to sum to one
  q_mnar   missingness informativeness  1 - 2|AUC(y ~ missingness indicators) - 1/2|
  q_size   effective minority sample    min(1, log10(1 + n_min / p) / log10(1 + 50)), n_min = minority-class
           count, p = number of features (50 events per feature saturates the score)
  q_label  label quality                1 - e_hat, with the anchor-point estimate e_hat = 1 - Q_0.975(max(p, 1 - p)) from
           5-fold out-of-sample probabilities (Liu and Tao, 2016); the confident-learning estimate (Northcutt et al.,
           2021) is recorded as _q_label_cl for sensitivity analysis
  q_rep    subgroup representation      min(1, pi_D / pi_T) * min(1, log10(1 + n_s) / log10(1 + 100)), n_s = subgroup count
  q_shift  covariate shift              1 - 2 max(0, AUC_dom - 1/2), AUC_dom = cross-validated AUC of a classifier
           separating training from target covariates
  q_time   temporal stability           1 - max(0, AUC_rand - AUC_temp) / (AUC_rand - 1/2): performance of a model trained on
           the earlier half of the training data and tested on the later half, relative to random cross-validation
           (1 if no time index)
TaskFit-Fixed = geometric mean of the dimension scores.
TaskFit-Learned = a monotone-constrained gradient-boosting regressor from the dimension scores to performance
retention, always evaluated on tasks it was not fitted to (leave-one-task-out).
Baselines:
  B_size    log10 of the number of training records
  B_generic task-agnostic quality: mean of unweighted completeness, uniqueness (1 - duplicate share), and
            1 - outlier share (|z| > 4)
  B_profile all seven dimensions with uniform feature weights (no task weighting) and geometric mean
  B_metric  METRIC-inspired model-independent profile: mean of unweighted completeness, uniqueness, share without extreme
            values, class balance min(1, 2 x minority share), subgroup balance min(1, 2 min(pi_D, 1 - pi_D)), and size
            min(1, log10(n) / 4)
  B_pilot   5-fold cross-validated AUC of a logistic pilot model on the training data
Degradations of the training data (target data untouched): sample fraction, MCAR missingness, MNAR missingness
(values above the 60th percentile of the most task-relevant feature removed with probability r), symmetric
label noise, minority-class downsampling, subgroup downsampling, covariate selection bias (sampling weighted
towards high values of a relevant feature), and feature measurement noise (Gaussian, in units of the
feature's standard deviation; not measured by any TaskFit dimension: a blind-spot test).
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

EPS = 1e-9


def lr_model():
    return make_pipeline(SimpleImputer(add_indicator=True), StandardScaler(), LogisticRegression(max_iter=2000))


def _oos_proba(X, y, seed=2026):
    k = int(min(5, np.bincount(y).min())) if np.bincount(y).min() >= 2 else 0
    if k < 2:
        return None
    return cross_val_predict(lr_model(), X, y, cv=StratifiedKFold(k, shuffle=True, random_state=seed), method="predict_proba")[:, 1]


def relevance(X, y):
    r = []
    for c in X.columns:
        o = X[c].notna().values
        v = X[c].values[o]
        r.append(0.0 if o.sum() < 10 or np.nanstd(v) == 0 else abs(np.nan_to_num(spearmanr(v, y[o])[0])))
    r = np.array(r) + 1e-3
    return r / r.sum()


def _dom_model():
    # imputed values only: missingness indicators would let injected missingness masquerade as population shift
    return make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000))


def _dom_auc(A, B, seed=2026):
    Z = pd.concat([A, B], ignore_index=True); lab = np.r_[np.zeros(len(A)), np.ones(len(B))].astype(int)
    p = cross_val_predict(_dom_model(), Z, lab, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return roc_auc_score(lab, p)


def dimensions(X, y, s, X_T, pi_T, order=None, task_aware=True, seed=2026):
    p = X.shape[1]; m = X.isna().mean().values
    w = relevance(X, y) if task_aware else np.full(p, 1 / p)
    q = {}
    q["q_comp"] = float(np.sum(w * (1 - m)))
    M = X.isna().astype(float)
    M = M.loc[:, M.std() > 0]
    if M.shape[1] and len(np.unique(y)) == 2:
        pm = _oos_proba(M, y, seed); q["q_mnar"] = 1 - 2 * abs(roc_auc_score(y, pm) - 0.5) if pm is not None else 1.0
    else:
        q["q_mnar"] = 1.0
    n_min = min(np.bincount(y, minlength=2))
    q["q_size"] = float(min(1.0, np.log10(1 + n_min / p) / np.log10(51)))
    pr = _oos_proba(X, y, seed)
    if pr is None:
        q["q_label"] = 0.0; cl = 0.0
    else:
        # anchor-point estimate (Liu and Tao, 2016): with symmetric noise rate e, calibrated probabilities cannot exceed
        # 1 - e, so e_hat = 1 - (97.5th percentile of max(p, 1 - p)); confident learning is kept as a sensitivity measure
        conf = np.maximum(pr, 1 - pr)
        q["q_label"] = float(np.clip(np.quantile(conf, 0.975), 0, 1))
        for qq in (0.90, 0.95, 0.99, 1.0):
            q[f"_q_label_p{int(round(qq * 1000))}"] = float(np.quantile(conf, qq))
        t1, t0 = pr[y == 1].mean(), (1 - pr[y == 0]).mean()
        sus = ((y == 1) & (pr < t1) & ((1 - pr) >= t0)) | ((y == 0) & ((1 - pr) < t0) & (pr >= t1))
        cl = float(1 - sus.mean())
    n_s = int(s.sum()); pi_D = s.mean()
    q["q_rep"] = float(min(1, pi_D / max(pi_T, EPS)) * min(1, np.log10(1 + n_s) / np.log10(101)))
    sub = X_T.sample(min(len(X_T), len(X)), random_state=seed) if len(X_T) > len(X) else X_T
    q["q_shift"] = float(1 - 2 * max(0, _dom_auc(X, sub, seed) - 0.5))
    if order is not None and len(np.unique(order)) > 10 and pr is not None:
        # temporal performance stability: a model trained on the earlier half, tested on the later half, relative to
        # random cross-validation
        o = np.argsort(order, kind="stable"); h = len(o) // 2; e, l = o[:h], o[h:]
        if len(np.unique(y[e])) == 2 and len(np.unique(y[l])) == 2:
            a_t = roc_auc_score(y[l], lr_model().fit(X.iloc[e], y[e]).predict_proba(X.iloc[l])[:, 1])
            a_r = roc_auc_score(y, pr)
            q["q_time"] = float(np.clip(1 - max(0, a_r - a_t) / max(a_r - 0.5, 0.02), 0, 1))
        else:
            q["q_time"] = 1.0
    else:
        q["q_time"] = 1.0
    q["_q_label_cl"] = cl
    if pr is None:
        for qq in (0.90, 0.95, 0.99, 1.0):
            q[f"_q_label_p{int(round(qq * 1000))}"] = 0.0
    return q


def taskfit_fixed(q):
    v = np.clip(np.array([v for k, v in q.items() if not k.startswith("_")]), 1e-3, 1)
    return float(np.exp(np.mean(np.log(v))))


def baselines(X, y, s, X_T, pi_T, order=None, seed=2026):
    b = {"B_size": float(np.log10(len(y)))}
    comp = 1 - X.isna().mean().mean()
    uniq = 1 - X.duplicated().mean()
    Z = (X - X.mean()) / (X.std() + EPS)
    out = (Z.abs() > 4).any(axis=1).mean()
    b["B_generic"] = float((comp + uniq + (1 - out)) / 3)
    b["B_profile"] = taskfit_fixed(dimensions(X, y, s, X_T, pi_T, order, task_aware=False, seed=seed))
    # METRIC-inspired model-independent profile (Schwabe et al., 2024; Becker et al., 2026): metrics from the informativeness,
    # representativeness, and consistency dimensions that need neither a model nor target-population data
    minority = min(np.bincount(y, minlength=2)) / len(y)
    b["B_metric"] = float(np.mean([comp, uniq, 1 - out, min(1.0, 2 * minority), min(1.0, 2 * min(s.mean(), 1 - s.mean())),
                                   min(1.0, np.log10(len(y)) / 4)]))
    pr = _oos_proba(X, y, seed)
    b["B_pilot"] = float(roc_auc_score(y, pr)) if pr is not None else 0.5
    return b


LEVELS = {"frac": [1.0, 0.5, 0.25, 0.1], "mcar": [0, 0.1, 0.3, 0.5], "mnar": [0, 0.3, 0.6], "label": [0, 0.05, 0.1, 0.2, 0.3],
          "pos_keep": [1.0, 0.5, 0.25, 0.1], "sub_keep": [1.0, 0.5, 0.2, 0.05], "selbias": [0, 0.5, 0.8], "meas": [0, 0.5, 1.0]}


def sample_levels(rng):
    """Each degradation is active with probability 0.4; active ones draw a non-zero level uniformly."""
    lv = {}
    for k, v in LEVELS.items():
        lv[k] = v[0] if rng.random() > 0.4 else v[rng.integers(1, len(v))]
    return lv


def degrade(X, y, s, order, lv, rel, rng):
    X = X.copy(); y = y.copy(); idx = np.arange(len(y))
    top = X.columns[int(np.argmax(rel))]
    if lv["selbias"] > 0:                                     # selection bias towards high values of the top feature
        v = X[top].rank(pct=True).fillna(0.5).values
        p = (1 - lv["selbias"]) + lv["selbias"] * 2 * v; p = p / p.sum()
        idx = rng.choice(idx, size=len(idx), replace=True, p=p)
    keep = np.ones(len(idx), bool)
    yy, ss = y[idx], s[idx]
    keep &= ~((yy == 1) & (rng.random(len(idx)) > lv["pos_keep"]))
    keep &= ~((ss == 1) & (rng.random(len(idx)) > lv["sub_keep"]))
    idx = idx[keep]
    if lv["frac"] < 1:
        idx = rng.choice(idx, size=max(30, int(len(idx) * lv["frac"])), replace=False)
    X = X.iloc[idx].reset_index(drop=True); y = y[idx]; s = s[idx]; order = None if order is None else order[idx]
    num = [c for c in X.columns if X[c].nunique() > 2]
    if lv["meas"] > 0:
        for c in num:
            X[c] = X[c] + rng.normal(0, lv["meas"] * X[c].std(), len(X))
    if lv["mnar"] > 0:
        thr = X[top].quantile(0.6)
        X.loc[(X[top] > thr) & (rng.random(len(X)) < lv["mnar"]), top] = np.nan
    if lv["mcar"] > 0:
        mask = rng.random(X.shape) < lv["mcar"]
        X = X.mask(mask)
    if lv["label"] > 0:
        flip = rng.random(len(y)) < lv["label"]; y = np.where(flip, 1 - y, y)
    return X, y.astype(int), s.astype(int), order
