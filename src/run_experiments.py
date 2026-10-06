"""
run_experiments.py

Settings per task: one random split (70 percent training pool, 30 percent target) and real-shift splits in which a
natural group is the target population: each Gambian region (3), village groups 0 and 1 for the Malawi tasks (2 each),
and the later 40 percent of the Ebola epidemic by onset date (1). In each setting, 150 degradation instances are
drawn (taskfit.sample_levels); TaskFit and the baselines are computed from the degraded training data and the
unlabelled target covariates only, and logistic regression and gradient boosting are trained on the degraded data and
evaluated on the clean target data. The reference model is trained on the undegraded training pool of the setting.
Outcomes: target AUC, retention = AUC / reference AUC, Brier score, expected calibration error (10 bins), and the
absolute AUC gap between the protected subgroup and the rest.
Output: outputs/instances.parquet. Seed 2026.
"""
import warnings
import zlib
import numpy as np, pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from tasks import TASKS
from taskfit import dimensions, taskfit_fixed, baselines, sample_levels, degrade, relevance, lr_model

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[1] / "outputs"; OUT.mkdir(exist_ok=True)
N_INST = 150


def ece(y, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1); e = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def evaluate(model, X, y, Xt, yt, st):
    model.fit(X, y); p = model.predict_proba(Xt)[:, 1]
    auc = roc_auc_score(yt, p)
    g = [roc_auc_score(yt[st == k], p[st == k]) if len(np.unique(yt[st == k])) == 2 else np.nan for k in (0, 1)]
    return {"auc": auc, "brier": brier_score_loss(yt, p), "ece": ece(yt, p), "sub_gap": abs(g[1] - g[0]) if np.isfinite(g).all() else np.nan,
            "auc_sub": g[1]}


def models():
    return {"lr": lr_model(), "gb": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, random_state=2026)}


def settings(t, rng):
    n = len(t["y"]); out = []
    perm = rng.permutation(n); cut = int(0.7 * n)
    out.append(("random", perm[:cut], perm[cut:]))
    if t["site"] is not None:
        groups = np.unique(t["site"]) if t["name"] == "gambia" else [0, 1]
        for g in groups:
            out.append((f"site{g}", np.flatnonzero(t["site"] != g), np.flatnonzero(t["site"] == g)))
    if t["order"] is not None:
        o = np.argsort(t["order"], kind="stable"); c = int(0.6 * n)
        out.append(("later", o[:c], o[c:]))
    return out


def one(t, sname, tr, te, i, ref):
    rng = np.random.default_rng(zlib.crc32(f"{t['name']}|{sname}".encode()) % 2**31 + 1000 * i)
    X, y, s = t["X"].iloc[tr].reset_index(drop=True), t["y"][tr], t["sub"][tr]
    order = None if t["order"] is None else t["order"][tr]
    Xt, yt, st = t["X"].iloc[te].reset_index(drop=True), t["y"][te], t["sub"][te]
    lv = sample_levels(rng)
    Xd, yd, sd, od = degrade(X, y, s, order, lv, relevance(X, y), rng)
    if len(np.unique(yd)) < 2 or np.bincount(yd).min() < 5:
        return None
    pi_T = st.mean()
    q = dimensions(Xd, yd, sd, Xt, pi_T, od)
    row = {"task": t["name"], "setting": sname, "inst": i, "n_train": len(yd), **{f"lv_{k}": v for k, v in lv.items()}, **q,
           "TaskFit_fixed": taskfit_fixed(q), **baselines(Xd, yd, sd, Xt, pi_T, od)}
    for mn, m in models().items():
        r = evaluate(m, Xd, yd, Xt, yt, st)
        for k, v in r.items():
            row[f"{mn}_{k}"] = v
        row[f"{mn}_retention"] = r["auc"] / ref[mn]
        row[f"{mn}_ref_auc"] = ref[mn]
    return row


SINGLE_REPS = 15


def one_single(t, tr, te, k, level, rep, ref):
    """single-factor sweep: only degradation k is active, at the given level"""
    from taskfit import LEVELS
    rng = np.random.default_rng(zlib.crc32(f"{t['name']}|{k}|{level}".encode()) % 2**31 + rep)
    X, y, s = t["X"].iloc[tr].reset_index(drop=True), t["y"][tr], t["sub"][tr]
    order = None if t["order"] is None else t["order"][tr]
    Xt, yt, st = t["X"].iloc[te].reset_index(drop=True), t["y"][te], t["sub"][te]
    lv = {kk: v[0] for kk, v in LEVELS.items()}; lv[k] = level
    Xd, yd, sd, od = degrade(X, y, s, order, lv, relevance(X, y), rng)
    if len(np.unique(yd)) < 2 or np.bincount(yd).min() < 5:
        return None
    q = dimensions(Xd, yd, sd, Xt, st.mean(), od)
    row = {"task": t["name"], "factor": k, "level": level, "rep": rep, "n_train": len(yd), "true_label_noise": lv["label"], **q, "TaskFit_fixed": taskfit_fixed(q),
           **baselines(Xd, yd, sd, Xt, st.mean(), od)}
    r = evaluate(lr_model(), Xd, yd, Xt, yt, st)
    row.update({"lr_auc": r["auc"], "lr_retention": r["auc"] / ref, "lr_ece": r["ece"], "lr_auc_sub": r["auc_sub"]})
    return row


def main_single():
    from taskfit import LEVELS
    rows = []
    for f in TASKS:
        t = f(); name, tr, te = settings(t, np.random.default_rng(2026))[0]
        ref = evaluate(lr_model(), t["X"].iloc[tr], t["y"][tr], t["X"].iloc[te].reset_index(drop=True), t["y"][te], t["sub"][te])["auc"]
        jobs = [(k, lvl, rp) for k, v in LEVELS.items() for lvl in v[1:] for rp in range(SINGLE_REPS)]
        res = Parallel(n_jobs=-1)(delayed(one_single)(t, tr, te, k, lvl, rp, ref) for k, lvl, rp in jobs)
        rows += [r for r in res if r is not None]; print(t["name"], "single-factor done", flush=True)
    pd.DataFrame(rows).to_parquet(OUT / "instances_single.parquet")


def main():
    rows = []
    for f in TASKS:
        t = f(); rng = np.random.default_rng(2026)
        for sname, tr, te in settings(t, rng):
            ref = {mn: evaluate(m, t["X"].iloc[tr], t["y"][tr], t["X"].iloc[te].reset_index(drop=True), t["y"][te], t["sub"][te])["auc"]
                   for mn, m in models().items()}
            res = Parallel(n_jobs=-1)(delayed(one)(t, sname, tr, te, i, ref) for i in range(N_INST))
            rows += [r for r in res if r is not None]
            print(t["name"], sname, "reference AUC lr %.3f gb %.3f" % (ref["lr"], ref["gb"]), "instances", sum(r is not None for r in res), flush=True)
    D = pd.DataFrame(rows); D.to_parquet(OUT / "instances.parquet")
    print(D.shape)


if __name__ == "__main__":
    main()
    main_single()
