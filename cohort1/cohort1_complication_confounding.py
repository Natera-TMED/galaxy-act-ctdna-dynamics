import os
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import fisher_exact

from _common import OUT_DIR, DAYS_PER_MONTH, load_act_timing_cohort, \
    simplify_stage, stage_dummies, fmt_p

LM = 84
MODEL_COVS = ["stage_num", "age", "ecog_ge1", "log_mtm"]

m = load_act_timing_cohort()
m["stage_num"] = m["p_StagePath"].apply(simplify_stage)
pos = m[m["mrd_pos"]].copy()

ed = pos[pos["arm"].isin(["early", "delayed"])].copy()
ed["age"] = pd.to_numeric(ed["Age"], errors="coerce")
ed["ecog_ge1"] = np.where(pd.to_numeric(ed["ECOG"], errors="coerce") >= 1, 1.0, 0.0)
ed.loc[ed["ECOG"].isna(), "ecog_ge1"] = np.nan
ed["log_mtm"] = np.log1p(pd.to_numeric(ed["mrd_mtm"], errors="coerce"))
ed["cd2"] = pd.to_numeric(ed["cd2"], errors="coerce")

ed = ed[ed["dfs_days"].notna() & (ed["dfs_days"] >= LM)].copy()
ed["_t"] = (pd.to_numeric(ed["dfs_days"], errors="coerce") - LM) / DAYS_PER_MONTH
ed["_e"] = pd.to_numeric(ed["dfs_evt"], errors="coerce").astype(int)
same = (ed["_t"] == 0) & (ed["_e"] == 1)
ed.loc[same, "_t"] = 1.0 / DAYS_PER_MONTH

base_full = ed.dropna(subset=["_t", "_e"] + MODEL_COVS).copy()
cc = base_full.dropna(subset=["cd2"]).copy()


def fit_delayed_hr(dd, extra_cols=()):
    parts = [stage_dummies(dd["stage_num"], "stage"), dd[["age"]].astype(float),
             dd[["ecog_ge1"]].astype(float), dd[["log_mtm"]].astype(float)]
    for c in extra_cols:
        parts.append(dd[[c]].astype(float))
    adj = pd.concat(parts, axis=1)
    arm = pd.get_dummies(dd["arm"], prefix="g").astype(float)
    if "g_early" in arm.columns:
        arm = arm.drop(columns=["g_early"])
    fit_df = pd.concat([dd[["_t", "_e"]].reset_index(drop=True), arm.reset_index(drop=True),
                        adj.reset_index(drop=True)], axis=1)
    for pen in (0.0, 0.1, 0.5):
        try:
            cph = CoxPHFitter(penalizer=pen).fit(fit_df, "_t", "_e")
            col = "g_delayed"
            return (float(np.exp(cph.params_[col])),
                    float(np.exp(cph.confidence_intervals_.loc[col, "95% lower-bound"])),
                    float(np.exp(cph.confidence_intervals_.loc[col, "95% upper-bound"])),
                    float(cph.summary.loc[col, "p"]), pen > 0, len(dd), int(dd["_e"].sum()))
        except Exception:
            continue
    return (np.nan,) * 7


def fit_cov_hr(dd, col):
    parts = [stage_dummies(dd["stage_num"], "stage"), dd[["age"]].astype(float),
             dd[["ecog_ge1"]].astype(float), dd[["log_mtm"]].astype(float), dd[[col]].astype(float)]
    arm = pd.get_dummies(dd["arm"], prefix="g").astype(float)
    if "g_early" in arm.columns:
        arm = arm.drop(columns=["g_early"])
    parts.append(arm)
    fit_df = pd.concat([dd[["_t", "_e"]].reset_index(drop=True),
                        pd.concat(parts, axis=1).reset_index(drop=True)], axis=1)
    for pen in (0.0, 0.1, 0.5):
        try:
            cph = CoxPHFitter(penalizer=pen).fit(fit_df, "_t", "_e")
            return (float(np.exp(cph.params_[col])),
                    float(np.exp(cph.confidence_intervals_.loc[col, "95% lower-bound"])),
                    float(np.exp(cph.confidence_intervals_.loc[col, "95% upper-bound"])),
                    float(cph.summary.loc[col, "p"]), pen > 0)
        except Exception:
            continue
    return (np.nan,) * 5


print("=" * 78)
print("Surgical-complication (Clavien-Dindo >=2) confounding analysis")
print("=" * 78)
print(f"Early+delayed after 12-wk landmark & complete model covariates : n={len(base_full)} "
      f"(early={int((base_full['arm']=='early').sum())}, delayed={int((base_full['arm']=='delayed').sum())})")
print(f"Complication status known (CD>=2 non-missing)                  : n={len(cc)} "
      f"(missing = {len(base_full) - len(cc)})")

tab = pd.crosstab(cc["arm"], cc["cd2"])
print("\n(a) Association of complication with ACT timing (exposure):")
rate = {}
for a in ["early", "delayed"]:
    n1 = int(tab.loc[a].get(1.0, 0)); n0 = int(tab.loc[a].get(0.0, 0))
    rate[a] = 100 * n1 / (n0 + n1)
    print(f"    {a:<8}: CD>=2 {n1}/{n0 + n1} ({rate[a]:.1f}%)")
orr, pf = fisher_exact([[int(tab.loc["early", 1.0]), int(tab.loc["early", 0.0])],
                        [int(tab.loc["delayed", 1.0]), int(tab.loc["delayed", 0.0])]])
print(f"    Fisher exact (delayed vs early CD>=2): OR={orr:.3f}, p={pf:.3f}")

cd_hr = fit_cov_hr(cc, "cd2")
print("\n(b) Association of complication with DFS (adjusted for arm+stage+age+ECOG+log(1+MTM)):")
print(f"    CD>=2 vs none : HR={cd_hr[0]:.3f} ({cd_hr[1]:.3f}-{cd_hr[2]:.3f}) p={cd_hr[3]:.3f}")

hr_primary = fit_delayed_hr(base_full)
hr_base_cc = fit_delayed_hr(cc)
hr_adj_cc  = fit_delayed_hr(cc, extra_cols=["cd2"])
pct = 100 * (hr_adj_cc[0] - hr_base_cc[0]) / hr_base_cc[0]


def inv(tup):
    return 1 / tup[0], 1 / tup[2], 1 / tup[1], tup[3]


print("\n(c) Delayed (8-12w) vs Early (0-8w) DFS HR — change-in-estimate:")
print(f"    Primary model (full n={hr_primary[5]}, ev={hr_primary[6]})          : "
      f"HR={hr_primary[0]:.3f} ({hr_primary[1]:.3f}-{hr_primary[2]:.3f}) p={hr_primary[3]:.4f}")
print(f"    Base model, complication-known subset (n={hr_base_cc[5]}, ev={hr_base_cc[6]}) : "
      f"HR={hr_base_cc[0]:.3f} ({hr_base_cc[1]:.3f}-{hr_base_cc[2]:.3f}) p={hr_base_cc[3]:.4f}")
print(f"    + adjusted for CD>=2 complication (same subset)          : "
      f"HR={hr_adj_cc[0]:.3f} ({hr_adj_cc[1]:.3f}-{hr_adj_cc[2]:.3f}) p={hr_adj_cc[3]:.4f}")
print(f"    Change in timing HR from adding complication             : {pct:+.1f}%")
e = inv(hr_base_cc)
print(f"\n    Manuscript direction — Early (0-8w) vs Delayed (8-12w), adjusted, N={hr_base_cc[5]}: "
      f"HR={e[0]:.2f} ({e[1]:.2f}-{e[2]:.2f}) p={fmt_p(e[3]) or '<0.001'}")

res = pd.DataFrame([
    {"step": "(a) complication ~ timing", "metric": "CD>=2 rate early (%)", "estimate": round(rate["early"], 1)},
    {"step": "(a) complication ~ timing", "metric": "CD>=2 rate delayed (%)", "estimate": round(rate["delayed"], 1)},
    {"step": "(a) complication ~ timing", "metric": "CD>=2 delayed vs early OR", "estimate": round(orr, 3), "p": round(pf, 4)},
    {"step": "(b) complication ~ DFS", "metric": "HR CD>=2 (adj)", "estimate": round(cd_hr[0], 3),
     "ci_low": round(cd_hr[1], 3), "ci_high": round(cd_hr[2], 3), "p": round(cd_hr[3], 4)},
    {"step": "(c) timing HR base (cc subset)", "metric": "HR delayed vs early", "estimate": round(hr_base_cc[0], 3),
     "ci_low": round(hr_base_cc[1], 3), "ci_high": round(hr_base_cc[2], 3), "p": round(hr_base_cc[3], 4)},
    {"step": "(c) timing HR + complication", "metric": "HR delayed vs early (adj CD>=2)", "estimate": round(hr_adj_cc[0], 3),
     "ci_low": round(hr_adj_cc[1], 3), "ci_high": round(hr_adj_cc[2], 3), "p": round(hr_adj_cc[3], 4)},
    {"step": "(c) change-in-estimate", "metric": "percent change in timing HR", "estimate": round(pct, 1)},
    {"step": "(summary) manuscript direction", "metric": "Early vs Delayed adj HR", "estimate": round(e[0], 3),
     "ci_low": round(e[1], 3), "ci_high": round(e[2], 3), "p": round(e[3], 4)},
])
out = os.path.join(OUT_DIR, "complication_confounding_summary.csv")
res.to_csv(out, index=False)
print(f"\nSaved -> {out}")
