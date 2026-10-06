"""
tasks.py

Five prediction tasks from four openly downloadable African health datasets (no registration):
  saheart   South Africa, Western Cape: coronary heart disease (Rousseauw et al., 1983;
            R package ElemStatLearn, github.com/cran/ElemStatLearn)
  gambia    The Gambia: malaria parasitaemia in children (Thomson et al., 1999; R package geoR,
            github.com/cran/geoR)
  hivgot    Malawi: collecting one's HIV test result (Thornton, 2008; Rdatasets causaldata)
  hivpos    Malawi: HIV positivity among those tested (same source)
  ebola     Sierra Leone 2014: laboratory confirmation of reported Ebola cases
            (R package outbreaks, github.com/reconverse/outbreaks)
Each task returns X (DataFrame, may contain missing values), y (0/1), sub (0/1 protected subgroup used
for subgroup analysis and under-representation), site (natural grouping for real-shift splits), and
order (time index where available).
"""
from pathlib import Path
import numpy as np, pandas as pd
import pyreadr

INP = Path(__file__).resolve().parents[1] / "inputs"


def saheart():
    d = list(pyreadr.read_r(INP / "SAheart.RData").values())[0].reset_index(drop=True)
    X = d[["sbp", "tobacco", "ldl", "adiposity", "typea", "obesity", "alcohol", "age"]].astype(float)
    X["famhist"] = (d.famhist == "Present").astype(float)
    return dict(name="saheart", X=X, y=d.chd.astype(int).values, sub=(d.age < 45).astype(int).values,
                site=None, order=None, sub_label="age < 45")


def gambia():
    d = pd.read_csv(INP / "gambia.csv")
    X = pd.DataFrame({"age_yr": d.age / 365.25, "netuse": d.netuse, "treated": d.treated, "green": d.green, "phc": d.phc})
    region = pd.cut(d.x, bins=[-np.inf, 400000, 520000, np.inf], labels=[0, 1, 2]).astype(int)   # west, central, east
    return dict(name="gambia", X=X, y=d.pos.astype(int).values, sub=(X.age_yr < 3).astype(int).values,
                site=region.values, order=None, sub_label="age < 3 years")


def _thornton():
    d = pd.read_csv(INP / "thornton_hiv.csv")
    return d[d.villnum.notna()].reset_index(drop=True)


def hivgot():
    d = _thornton(); d = d[d.got.notna()].reset_index(drop=True)
    X = d[["distvct", "tinc", "any", "age"]].astype(float)
    return dict(name="hivgot", X=X, y=d.got.astype(int).values, sub=(d.age < 30).fillna(False).astype(int).values,
                site=(d.villnum % 4).astype(int).values, order=None, sub_label="age < 30")


def hivpos():
    d = _thornton(); d = d[d.hiv2004.isin([0, 1])].reset_index(drop=True)   # -1 = indeterminate result, excluded
    X = d[["distvct", "age", "tinc"]].astype(float)
    return dict(name="hivpos", X=X, y=d.hiv2004.astype(int).values, sub=(d.age < 30).fillna(False).astype(int).values,
                site=(d.villnum % 4).astype(int).values, order=None, sub_label="age < 30")


def ebola():
    d = list(pyreadr.read_r(INP / "ebola_sierraleone_2014.RData").values())[0].reset_index(drop=True)
    on, sa = pd.to_datetime(d.date_of_onset), pd.to_datetime(d.date_of_sample)
    X = pd.DataFrame({"age": d.age.astype(float), "female": d.sex.astype(object).map({"F": 1.0, "M": 0.0}).astype(float),
                      "delay_days": (sa - on).dt.days.clip(lower=0, upper=60).astype(float),
                      "week": ((on - on.min()).dt.days // 7).astype(float)})
    for k in d.district.astype(str).value_counts().index[:12]:
        X[f"d_{k}"] = (d.district.astype(str) == k).astype(float)
    sub = (d.age < 15).astype(int).values
    return dict(name="ebola", X=X, y=(d.status.astype(str) == "confirmed").astype(int).values, sub=sub,
                site=None, order=on.values.astype("datetime64[D]").astype(int), sub_label="age < 15")


def _as_float(fn):
    def g():
        t = fn(); t["X"] = t["X"].astype(float); return t
    g.__name__ = fn.__name__
    return g


TASKS = [_as_float(f) for f in (saheart, gambia, hivgot, hivpos, ebola)]

if __name__ == "__main__":
    rows = []
    for f in TASKS:
        t = f()
        rows.append({"task": t["name"], "n": len(t["y"]), "features": t["X"].shape[1], "prevalence": t["y"].mean(),
                     "missing_share": t["X"].isna().mean().mean(), "subgroup_share": t["sub"].mean(),
                     "sites": None if t["site"] is None else len(np.unique(t["site"]))})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
