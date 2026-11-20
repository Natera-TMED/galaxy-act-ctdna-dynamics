import os
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from _common import OUT_DIR, DAYS_PER_MONTH, load_act_timing_cohort, simplify_stage, \
    plot_forest, fmt_p

MIN_EVENTS = 5
STAGE_ORDER = ["I\u2013II", "III", "IV"]

m = load_act_timing_cohort()
m["stage_num"] = m["p_StagePath"].apply(simplify_stage)
m["age"] = pd.to_numeric(m["Age"], errors="coerce")
m["ecog_ge1"] = np.where(pd.to_numeric(m["ECOG"], errors="coerce") >= 1, 1.0, 0.0)
m.loc[m["ECOG"].isna(), "ecog_ge1"] = np.nan
m["male"] = np.where(m["Sex"] == "Male", 1.0, np.where(m["Sex"] == "Female", 0.0, np.nan))
m["pT4"] = m["pT"].apply(lambda x: 1.0 if str(x).startswith("pT4")
                         else (0.0 if str(x).startswith(("pT1", "pT2", "pT3")) else np.nan))
m["pN_pos"] = m["pN"].apply(lambda x: 0.0 if str(x).startswith("pN0")
                            else (1.0 if str(x).startswith(("pN1", "pN2")) else np.nan))
m["right_sided"] = np.where(m["LocationGrpA"] == "Right-sided colon", 1.0,
                            np.where(m["LocationGrpA"] == "Left-sided colon", 0.0, np.nan))
m["rectum"] = np.where(m["LocationGrpB"] == "Rectum", 1.0,
                       np.where(m["LocationGrpB"] == "Colon", 0.0, np.nan))

m["msi_h"]    = m["msi_call"].map({"MSS": 0.0, "MSI-High": 1.0})
m["ras_pos"]  = m["ras_call"].map({"WT": 0.0, "MUT": 1.0})
m["braf_pos"] = m["braf_call"].map({"WT": 0.0, "MUT": 1.0})
m["lvi_present"] = m["p_LVI"].apply(lambda x: np.nan if pd.isna(x)
                                    else (1.0 if ("Ly1" in str(x) or "V1" in str(x)) else 0.0))
m["pni_present"] = m["p_PNI"].apply(lambda x: 1.0 if str(x).startswith("Pn1")
                                    else (0.0 if str(x).startswith("Pn0") else np.nan))
m["mrd_pos_f"] = m["mrd_pos"].astype(float)
m["age10"] = m["age"] / 10.0
m["log1p_mtm"] = np.log1p(pd.to_numeric(m["mrd_mtm"], errors="coerce"))

LM_DAYS = 84
coh = m[m["arm"].isin(["delayed", "late"])].copy()
_dfs = pd.to_numeric(coh["dfs_days"], errors="coerce")
_evt = pd.to_numeric(coh["dfs_evt"], errors="coerce")
coh = coh[_dfs.notna() & _evt.notna() & (_dfs >= LM_DAYS)].copy()
coh["_t"] = (pd.to_numeric(coh["dfs_days"], errors="coerce") - LM_DAYS) / DAYS_PER_MONTH
coh["_e"] = pd.to_numeric(coh["dfs_evt"], errors="coerce")
_same = (coh["_t"] == 0) & (coh["_e"] == 1)
coh.loc[_same, "_t"] = 1.0 / DAYS_PER_MONTH
print(f"Late-starter cohort (ACT 8\u201324w), 12-week landmark: n={len(coh)}, DFS events={int(coh['_e'].sum())}")

def stage_group(d):
    g = pd.to_numeric(d["stage_num"], errors="coerce")
    return pd.Series(np.select([g.isin([1, 2]), g.eq(3), g.eq(4)], STAGE_ORDER, default=None),
                     index=d.index).where(lambda s: s.notna())

def build_stage_cols(d):
    grp = stage_group(d)
    if grp.dropna().nunique() < 2:
        return []
    ref = [s for s in STAGE_ORDER if (grp == s).any()][0]
    dd = pd.get_dummies(grp, prefix="stage").astype(float)
    dd = dd.drop(columns=[f"stage_{ref}"]) if f"stage_{ref}" in dd.columns else dd
    order = [f"stage_{s}" for s in STAGE_ORDER if f"stage_{s}" in dd.columns]
    dd = dd[order]
    for c in dd.columns:
        d[c] = dd[c]
    return list(dd.columns)

def stage_ref(d):
    present = [s for s in STAGE_ORDER if (stage_group(d) == s).any()]
    return present[0] if len(present) >= 2 else "reference"

def cox_fit(d, cols, penalizers=(0.0, 0.1, 0.5, 1.0)):
    use = d.dropna(subset=["_t", "_e"] + cols).copy()
    n, ev = len(use), int(use["_e"].sum())
    meta = {"n": n, "events": ev, "cindex": np.nan, "penalized": None}
    if n == 0 or ev < MIN_EVENTS or use[cols].nunique().min() < 2:
        return [], meta
    fit_df = use[["_t", "_e"] + cols].reset_index(drop=True)
    last = None
    for pen in penalizers:
        try:
            cph = CoxPHFitter(penalizer=pen).fit(fit_df, duration_col="_t", event_col="_e")
            rows = [{"covariate": c, "HR": float(np.exp(cph.params_[c])),
                     "CI_low": float(np.exp(cph.confidence_intervals_.loc[c, "95% lower-bound"])),
                     "CI_high": float(np.exp(cph.confidence_intervals_.loc[c, "95% upper-bound"])),
                     "p": float(cph.summary.loc[c, "p"])} for c in cols if c in cph.params_.index]
            meta.update({"cindex": float(cph.concordance_index_), "penalized": bool(pen > 0)})
            degenerate = any((abs(np.log(r["HR"])) > 5) or (not np.isfinite(r["CI_low"]))
                             or (not np.isfinite(r["CI_high"])) or r["CI_low"] <= 0 for r in rows)
            last = (rows, dict(meta))
            if not degenerate:
                return rows, meta
        except Exception:
            continue
    return last if last is not None else ([], meta)

UNI_VARS = [("MRD-window ctDNA positive", "mrd_pos_f"), ("Age (per 10 y)", "age10"),
            ("ECOG \u22651", "ecog_ge1"), ("Male sex", "male"), ("pT4 (vs T1\u2013T3)", "pT4"),
            ("Node positive (N1/N2)", "pN_pos"), ("Right-sided (vs left)", "right_sided"),
            ("Rectum (vs colon)", "rectum"), ("MSI-H (vs MSS/pMMR)", "msi_h"),
            ("RAS mutant", "ras_pos"), ("BRAF mutant", "braf_pos"),
            ("LVI present", "lvi_present"), ("PNI present", "pni_present"),
            ("log(1+MTM) [all]", "log1p_mtm")]

def label_of(col, ref_roman):
    if col.startswith("stage_"):
        return f"Stage {col.split('_', 1)[1]} (vs {ref_roman})"
    for lab, c in UNI_VARS:
        if c == col:
            return lab
    return col

stage_cols = build_stage_cols(coh)
ref = stage_ref(coh)
uni_rows, uni_forest = [], []
srows, smeta = cox_fit(coh, stage_cols)
for r in srows:
    uni_rows.append({"variable": label_of(r["covariate"], ref), "HR": round(r["HR"], 3),
                     "CI_low": round(r["CI_low"], 3), "CI_high": round(r["CI_high"], 3),
                     "p": round(r["p"], 4), "n": smeta["n"], "events": smeta["events"]})
    uni_forest.append((label_of(r["covariate"], ref), r["HR"], r["CI_low"], r["CI_high"], r["p"]))
for lab, col in UNI_VARS:
    rows, meta = cox_fit(coh, [col])
    if not rows:
        continue
    r = rows[0]
    uni_rows.append({"variable": lab, "HR": round(r["HR"], 3), "CI_low": round(r["CI_low"], 3),
                     "CI_high": round(r["CI_high"], 3), "p": round(r["p"], 4),
                     "n": meta["n"], "events": meta["events"]})
    uni_forest.append((lab, r["HR"], r["CI_low"], r["CI_high"], r["p"]))

uni_df = pd.DataFrame(uni_rows)
uni_df.to_csv(os.path.join(OUT_DIR, "figure_2C_univariable_DFS.csv"), index=False)
plot_forest(uni_forest, "Figure 2C. Univariable predictors of DFS (ACT 8\u201324w)",
            "figure_2C_univariable_forest", xlabel="Hazard ratio for DFS event (95% CI, log scale)",
            subtitle=f"n={len(coh)}, events={int(coh['_e'].sum())}; each row = separate model")
print("\nFIGURE 2C — univariable predictors of DFS")
print(uni_df.to_string(index=False))

core_cols = ["mrd_pos_f"] + stage_cols + ["age10", "ecog_ge1", "right_sided"]
crows, cmeta = cox_fit(coh, core_cols)
core_rows, core_forest = [], []
for r in crows:
    core_rows.append({"variable": label_of(r["covariate"], ref), "HR": round(r["HR"], 3),
                      "CI_low": round(r["CI_low"], 3), "CI_high": round(r["CI_high"], 3),
                      "p": round(r["p"], 4), "n_model": cmeta["n"], "events_model": cmeta["events"],
                      "C_index": round(cmeta["cindex"], 3)})
    core_forest.append((label_of(r["covariate"], ref), r["HR"], r["CI_low"], r["CI_high"], r["p"]))
core_df = pd.DataFrame(core_rows)
core_df.to_csv(os.path.join(OUT_DIR, "figure_2D_multivariable_CORE_DFS.csv"), index=False)
plot_forest(core_forest, "Figure 2D. Multivariable (CORE) predictors of DFS (ACT 8\u201324w)",
            "figure_2D_multivariable_forest", xlabel="Hazard ratio for DFS event (95% CI, log scale)",
            subtitle=f"n={cmeta['n']}, events={cmeta['events']}, C-index={cmeta['cindex']:.3f}; mutually adjusted")
print(f"\nFIGURE 2D — CORE multivariable model (n={cmeta['n']}, events={cmeta['events']}, "
      f"C-index={cmeta['cindex']:.3f})")
print(core_df.to_string(index=False))

pos = coh[coh["mrd_pos"] & (coh["stage_num"].isin([1, 2, 3]))].copy()
pos["stage_III"] = (pd.to_numeric(pos["stage_num"], errors="coerce") == 3).astype(float)
e2_cols = ["log1p_mtm", "stage_III", "age10", "ecog_ge1", "right_sided"]
erows, emeta = cox_fit(pos, e2_cols, penalizers=(0.0,))
LBL2E = {"log1p_mtm": "log(1+MTM) per unit", "stage_III": "Stage III (vs I\u2013II)",
         "age10": "Age (per 10 y)", "ecog_ge1": "ECOG \u22651", "right_sided": "Right-sided (vs left)"}
e2_rows, e2_forest = [], []
for r in erows:
    e2_rows.append({"variable": LBL2E.get(r["covariate"], r["covariate"]), "HR": round(r["HR"], 3),
                    "CI_low": round(r["CI_low"], 3), "CI_high": round(r["CI_high"], 3),
                    "p": round(r["p"], 4), "n_model": emeta["n"], "events_model": emeta["events"],
                    "C_index": round(emeta["cindex"], 3)})
    e2_forest.append((LBL2E.get(r["covariate"], r["covariate"]), r["HR"], r["CI_low"], r["CI_high"], r["p"]))
e2_df = pd.DataFrame(e2_rows)
e2_df.to_csv(os.path.join(OUT_DIR, "figure_2E_MRDpos_ctdna_burden_DFS.csv"), index=False)
plot_forest(e2_forest, "Figure 2E. Multivariate analysis with ctDNA levels log(1+MTM/mL) (ACT 8\u201324w) (Stage I\u2013III)",
            "figure_2E_ctdna_burden_forest", xlabel="Hazard ratio for DFS event (95% CI, log scale)",
            subtitle=f"MRD+ (Stage I\u2013III); n={emeta['n']}, events={emeta['events']}, C-index={emeta['cindex']:.3f}")
print(f"\nFIGURE 2E — ctDNA burden (MRD+, Stage I\u2013III; n={emeta['n']}, events={emeta['events']})")
print(e2_df.to_string(index=False))
