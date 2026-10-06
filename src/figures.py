"""figures.py: TaskFit article figures at exact IEEE print sizes (column 3.5 in, text 7.16 in)."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; TAB = OUT / "tables"; FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"],
                     "mathtext.fontset": "stix",
                     "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7, "lines.linewidth": 1.3, "lines.markersize": 3.5, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 600})


def save(fig, n):
    fig.savefig(FIG / f"{n}.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(FIG / f"{n}.png", bbox_inches="tight", pad_inches=0.02); plt.close(fig)


def box(ax, x, y, w, h, t, c, fs=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.05", fc=c, ec="k", lw=0.7))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, linespacing=1.15)


def arr(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=0.9, color="k", shrinkA=1, shrinkB=1))


def fig_framework():
    W, H = 7.0, 2.75
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    box(ax, 0.02, 1.55, 1.35, 0.85, "Training data\n$D=(X, y, s)$\nwith task label $y$", "#deebf7")
    box(ax, 0.02, 0.35, 1.35, 0.85, "Target population\nunlabelled $X_T$,\nsubgroup share $\\pi_T$", "#deebf7")
    dims = ["completeness (task-weighted)", "informative missingness", "effective minority sample", "label quality (anchor point)",
            "subgroup representation", "covariate shift", "temporal stability"]
    for i, d in enumerate(dims):
        box(ax, 1.75, 2.17 - i * 0.31, 2.0, 0.26, d, "#fff7bc", fs=7)
    arr(ax, (1.37, 1.97), (1.75, 1.97)); arr(ax, (1.37, 0.78), (1.75, 0.78))
    box(ax, 4.1, 0.3, 1.25, 0.75, "TaskFit\ngeometric mean\nof $q_1,\\dots,q_7$", "#e5f5e0")
    box(ax, 4.1, 1.45, 1.25, 0.75, "Pilot model\ncross-validated\nAUC on $D$", "#fde0dd")
    arr(ax, (3.75, 1.3), (4.1, 0.68))
    ax.plot([0.7, 0.7, 4.72], [2.4, 2.6, 2.6], color="k", lw=0.9); arr(ax, (4.72, 2.6), (4.72, 2.2))
    box(ax, 5.7, 0.75, 1.27, 1.0, "Fitness for the task:\nshift and data condition\n(TaskFit) + learnability\n(pilot)", "#f0f0f0", fs=7)
    arr(ax, (5.35, 1.82), (5.7, 1.45)); arr(ax, (5.35, 0.68), (5.7, 1.05))
    save(fig, "fig_framework")


def fig_dose():
    Q = pd.read_parquet(OUT / "instances_single.parquet")
    panels = [("mcar", "q_comp", "MCAR missingness rate", "completeness $q_{comp}$"),
              ("sub_keep", "q_rep", "subgroup records kept", "representation $q_{rep}$"),
              ("label", "q_label", "label noise rate", "label quality $q_{label}$"),
              ("selbias", "q_shift", "selection bias strength", "shift $q_{shift}$"),
              ("frac", "q_size", "share of records kept", "effective sample $q_{size}$"),
              ("meas", "TaskFit_fixed", "measurement noise (SD units)", "TaskFit (blind spot)")]
    base = {"mcar": 0.0, "sub_keep": 1.0, "label": 0.0, "selbias": 0.0, "frac": 1.0, "meas": 0.0}
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.9))
    for ax, (f, d, xl, yl) in zip(axes.ravel(), panels):
        g = Q[Q.factor == f].groupby("level").agg(m=(d, "mean"), s=(d, "sem"), r=("lr_retention", "mean"), rs=("lr_retention", "sem"),
                                                  p=("B_pilot", "mean")).reset_index()
        ax.errorbar(g.level, g.m, yerr=1.96 * g.s, fmt="o-", color="#d62728", capsize=2, label=yl)
        if f == "meas":
            ax.plot(g.level, g.p, "s--", color="#7f7f7f", label="pilot AUC")
        ax.set_xlabel(xl); ax.set_ylabel(yl, color="#d62728")
        a2 = ax.twinx(); a2.errorbar(g.level, g.r, yerr=1.96 * g.rs, fmt="^:", color="#1f77b4", capsize=2)
        a2.set_ylabel("retention", color="#1f77b4"); a2.spines["right"].set_visible(True); a2.spines["top"].set_visible(False)
        if f in ("sub_keep", "frac"):
            ax.invert_xaxis()
        if f == "meas":
            ax.legend(frameon=False, loc="lower left", fontsize=6.5)
    fig.tight_layout(w_pad=1.2); save(fig, "fig_dose")


def fig_shift():
    S = pd.read_csv(TAB / "L_shift_changes.csv")
    T = pd.read_csv(TAB / "L_shift_change_tests.csv").set_index("score")
    r_q, p_q = T.loc["q_shift", "rho_change_vs_auc_change"], T.loc["q_shift", "p_permutation_one_sided"]
    r_p = T.loc["pilot", "rho_change_vs_auc_change"]
    lab = {("ebola", "later"): "Ebola later", ("gambia", "site0"): "Gambia W", ("gambia", "site1"): "Gambia C", ("gambia", "site2"): "Gambia E",
           ("hivgot", "site0"): "Malawi-c 0", ("hivgot", "site1"): "Malawi-c 1", ("hivpos", "site0"): "Malawi-p 0", ("hivpos", "site1"): "Malawi-p 1"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, c, t in [(axes[0], "d_q_shift", f"change in shift score $q_{{shift}}$ ($\\rho={r_q:.2f}$, $p={p_q:.3f}$)"),
                     (axes[1], "d_pilot", f"change in pilot AUC ($\\rho={r_p:.2f}$)")]:
        ax.scatter(S[c], S.d_auc, s=22, color="#d62728" if c == "d_q_shift" else "#7f7f7f", zorder=3)
        for _, r in S.iterrows():
            if r.task in ("ebola", "gambia") and not (c == "d_pilot" and r.setting == "site2"):
                ax.annotate(lab[(r.task, r.setting)], (r[c], r.d_auc), textcoords="offset points", xytext=(4, 2), fontsize=6.3)
        m = S[S.task.isin(["hivgot", "hivpos"])]
        ax.annotate("Malawi (4 settings)", (m[c].mean(), m.d_auc.max()), textcoords="offset points", xytext=(0, 6), fontsize=6.3, ha="center")
        ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6); ax.set_xlabel(t); ax.grid(alpha=0.3)
    axes[0].set_ylabel("change in target AUC")
    fig.tight_layout(); save(fig, "fig_shift")


def fig_validity():
    O = pd.read_csv(TAB / "O_metric_baseline.csv").set_index("score"); A = pd.read_csv(TAB / "A_validity.csv").set_index("score")
    L = pd.read_csv(TAB / "L_shift_change_tests.csv").set_index("score")
    names = {"B_generic": "generic quality", "B_size": "sample size", "B_metric": "METRIC-inspired", "TaskFit_fixed": "TaskFit",
             "B_pilot": "pilot model", "TaskFit_plus_pilot": "TaskFit + pilot"}
    cols = {"B_generic": "#bdbdbd", "B_size": "#bdbdbd", "B_metric": "#6baed6", "TaskFit_fixed": "#d62728", "B_pilot": "#7f7f7f",
            "TaskFit_plus_pilot": "#a50f15", "q_shift": "#fb6a4a"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    k1 = ["B_generic", "B_size", "B_metric", "TaskFit_fixed", "B_pilot", "TaskFit_plus_pilot"]
    v = [O.loc[k, "rho_retention"] if k in O.index else A.loc[k, "rho_pooled"] for k in k1]
    lo = [O.loc[k, "ci_lo"] if k in O.index else A.loc[k, "ci_lo"] for k in k1]; hi = [O.loc[k, "ci_hi"] if k in O.index else A.loc[k, "ci_hi"] for k in k1]
    y = np.arange(len(k1)); ax = axes[0]
    ax.barh(y, v, color=[cols[k] for k in k1], xerr=[np.array(v) - lo, np.array(hi) - v], capsize=2)
    ax.set_yticks(y); ax.set_yticklabels([names[k] for k in k1]); ax.axvline(0, color="k", lw=0.6)
    ax.set_title("controlled degradation: $\\rho$ with retention"); ax.grid(axis="x", alpha=0.3)
    k2 = [("B_generic", O.loc["B_generic", "rho_shift_change"]), ("B_metric", O.loc["B_metric", "rho_shift_change"]),
          ("TaskFit_fixed", L.loc["TaskFit", "rho_change_vs_auc_change"]), ("q_shift", L.loc["q_shift", "rho_change_vs_auc_change"]),
          ("B_pilot", L.loc["pilot", "rho_change_vs_auc_change"])]
    ax = axes[1]; y = np.arange(len(k2))
    ax.barh(y, [x for _, x in k2], color=[cols[k] for k, _ in k2])
    ax.set_yticks(y); ax.set_yticklabels([names.get(k, "TaskFit shift $q_{shift}$") for k, _ in k2]); ax.axvline(0, color="k", lw=0.6)
    ax.set_title("real shift: $\\rho$ of changes with $\\Delta$ target AUC"); ax.grid(axis="x", alpha=0.3); ax.set_xlim(-0.8, 1.0)
    fig.tight_layout(); save(fig, "fig_validity")


SETTING_LABELS = {("saheart", "random"): "S. Africa, CHD", ("gambia", "random"): "Gambia, malaria", ("gambia", "site0"): "Gambia, west (shift)",
                  ("gambia", "site1"): "Gambia, central (shift)", ("gambia", "site2"): "Gambia, east (shift)",
                  ("hivgot", "random"): "Malawi, collection", ("hivgot", "site0"): "Malawi, coll., gr. 0 (shift)",
                  ("hivgot", "site1"): "Malawi, coll., gr. 1 (shift)", ("hivpos", "random"): "Malawi, HIV status",
                  ("hivpos", "site0"): "Malawi, HIV, gr. 0 (shift)", ("hivpos", "site1"): "Malawi, HIV, gr. 1 (shift)",
                  ("ebola", "random"): "Sierra Leone, Ebola", ("ebola", "later"): "Ebola, later phase (shift)"}


def clean_profiles():
    """Dimension profile, TaskFit, pilot AUC, and target AUC of the undegraded training pool of every setting."""
    from tasks import TASKS
    from taskfit import dimensions, taskfit_fixed, baselines, lr_model
    from run_experiments import settings, evaluate
    rows = []
    for f in TASKS:
        t = f()
        for sname, tr, te in settings(t, np.random.default_rng(2026)):
            X, y, s = t["X"].iloc[tr].reset_index(drop=True), t["y"][tr], t["sub"][tr]
            Xt, yt, st = t["X"].iloc[te].reset_index(drop=True), t["y"][te], t["sub"][te]
            order = None if t["order"] is None else t["order"][tr]
            q = dimensions(X, y, s, Xt, st.mean(), order)
            b = baselines(X, y, s, Xt, st.mean(), order)
            rows.append({"task": t["name"], "setting": sname, **{k: v for k, v in q.items() if not k.startswith("_")},
                         "TaskFit": taskfit_fixed(q), "pilot": b["B_pilot"], "target_auc": evaluate(lr_model(), X, y, Xt, yt, st)["auc"]})
    P = pd.DataFrame(rows); P.to_csv(TAB / "P_clean_profiles.csv", index=False)
    return P


def fig_profiles(P):
    dims = ["q_comp", "q_mnar", "q_size", "q_label", "q_rep", "q_shift", "q_time"]
    dl = ["completeness", "informative\nmissingness", "effective\nsample", "label\nquality", "represent-\nation", "covariate\nshift", "temporal\nstability"]
    M = P[dims].values
    fig, ax = plt.subplots(figsize=(3.5, 3.9))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(dims))); ax.set_xticklabels(dl, fontsize=6.3, rotation=90)
    ax.set_yticks(range(len(P))); ax.set_yticklabels([SETTING_LABELS[(a, b)] for a, b in zip(P.task, P.setting)], fontsize=6.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.6, color="k" if 0.25 < M[i, j] < 0.85 else ("w" if M[i, j] <= 0.25 else "k"))
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.ax.tick_params(labelsize=6.5); cb.set_label("dimension score", fontsize=7)
    fig.tight_layout(); save(fig, "fig_profiles")


def fig_learnability(P):
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    sc = ax.scatter(P.TaskFit, P.pilot, c=P.target_auc, cmap="viridis", vmin=0.45, vmax=0.9, s=34, edgecolor="k", lw=0.4, zorder=3)
    short = {("saheart", "random"): "S. Africa CHD", ("gambia", "random"): "Gambia", ("hivgot", "random"): "Malawi coll.",
             ("hivpos", "random"): "Malawi HIV", ("ebola", "random"): "Ebola", ("ebola", "later"): "Ebola later",
             ("gambia", "site0"): "Gambia W", ("gambia", "site2"): "Gambia E", ("gambia", "site1"): "Gambia C"}
    off = {("saheart", "random"): (-6, -4, "right"), ("gambia", "random"): (-4, 7, "right"), ("hivgot", "random"): (-5, 6, "right"),
           ("hivpos", "random"): (-6, -3, "right"), ("ebola", "random"): (6, -3, "left"), ("ebola", "later"): (6, -3, "left"),
           ("gambia", "site0"): (6, 2, "left"), ("gambia", "site2"): (-6, -3, "right"), ("gambia", "site1"): (-6, -11, "right")}
    for _, r in P.iterrows():
        k = (r.task, r.setting)
        if k in short:
            dx, dy, ha = off[k]
            ax.annotate(short[k], (r.TaskFit, r.pilot), textcoords="offset points", xytext=(dx, dy), fontsize=6.2, ha=ha)
    xm, ym = 0.75, 0.65
    ax.axvline(xm, color="k", lw=0.6, ls="--"); ax.axhline(ym, color="k", lw=0.6, ls="--")
    ax.set_xlim(0.22, 1.04); ax.set_ylim(0.5, 0.97)
    kw = dict(fontsize=6.3, style="italic", color="#444444")
    ax.text(0.77, 0.955, "fit and learnable", ha="left", va="top", **kw)
    ax.text(1.03, 0.505, "fit, little signal", ha="right", va="bottom", **kw)
    ax.text(0.24, 0.80, "learnable,\ndefect flagged", ha="left", va="center", **kw)
    ax.text(0.24, 0.52, "defect flagged,\nlittle signal", ha="left", va="bottom", **kw)
    ax.set_xlabel("TaskFit score (data fitness)"); ax.set_ylabel("pilot AUC (learnability)")
    cb = fig.colorbar(sc, ax=ax, pad=0.02); cb.set_label("target AUC", fontsize=7); cb.ax.tick_params(labelsize=6.5)
    ax.grid(alpha=0.3); fig.tight_layout(); save(fig, "fig_learnability")


if __name__ == "__main__":
    fig_framework(); fig_dose(); fig_shift(); fig_validity()
    P = clean_profiles(); fig_profiles(P); fig_learnability(P); print("ok")