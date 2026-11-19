import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter

from _common import (OUT_DIR, DAYS_PER_MONTH, load_dynamics_cohort, stage_dummies,
                     fit_km, draw_km_curve, draw_risk_table, hide_spines, fmt_p)

MIN_N = 5
GROUP_ORDER = ["sustained_negative", "clearance", "decrease_detectable", "rising", "newly_detectable"]
LABELS = {"sustained_negative": "Sustained negative", "clearance": "Clearance",
          "decrease_detectable": "Decrease but detectable", "rising": "Rising",
          "newly_detectable": "Newly detectable"}
COLORS = {"sustained_negative": "#1B5E20", "clearance": "#8BC400",
          "decrease_detectable": "#008BCE", "rising": "#004677", "newly_detectable": "#E87722"}

df = load_dynamics_cohort()


def landmark_prep(d, dfs_col="p_drelDFS_Any", evt_col="p_evtDFS", lm_col="tp2_day"):
    d = d.dropna(subset=[dfs_col, evt_col, lm_col]).copy()
    d["_evt"] = d[evt_col].astype(int)
    d["_lm_days"] = d[dfs_col] - d[lm_col]
    d = d[d["_lm_days"] >= 0].copy()
    same = (d["_lm_days"] == 0) & (d["_evt"] == 1)
    d.loc[same, "_lm_days"] = 1
    d["_lm_months"] = d["_lm_days"] / DAYS_PER_MONTH
    return d


def cox_binary(ref_t, ref_e, cmp_t, cmp_e, ref_stage, cmp_stage):
    cdf = pd.DataFrame({"T": np.concatenate([ref_t, cmp_t]),
                        "E": np.concatenate([ref_e, cmp_e]),
                        "grp": np.array([0] * len(ref_t) + [1] * len(cmp_t), dtype=float),
                        "stage": np.concatenate([ref_stage, cmp_stage])})
    cdf = cdf.dropna(subset=["stage"])
    st = stage_dummies(cdf["stage"])
    cdf = cdf.drop(columns=["stage"])
    if st.shape[1] >= 1:
        cdf = pd.concat([cdf, st], axis=1)
    for pen in (0.0, 0.1):
        try:
            cph = CoxPHFitter(penalizer=pen).fit(cdf, "T", "E")
            return (float(np.exp(cph.params_["grp"])),
                    float(np.exp(cph.confidence_intervals_.loc["grp", "95% lower-bound"])),
                    float(np.exp(cph.confidence_intervals_.loc["grp", "95% upper-bound"])),
                    float(cph.summary.loc["grp", "p"]))
        except Exception:
            continue
    return np.nan, np.nan, np.nan, np.nan


def cox_multivariate(lm_df, ref_key="sustained_negative"):
    d = lm_df[["_lm_months", "_evt", "ctdna_dynamics", "stage_num"]].dropna(subset=["_lm_months", "_evt"]).copy()
    dummies = pd.get_dummies(d["ctdna_dynamics"], prefix="grp").astype(float)
    if f"grp_{ref_key}" in dummies.columns:
        dummies = dummies.drop(columns=[f"grp_{ref_key}"])
    cox_df = pd.concat([d[["_lm_months", "_evt", "stage_num"]], dummies], axis=1).dropna(subset=["stage_num"])
    st = stage_dummies(cox_df["stage_num"])
    fit_df = pd.concat([cox_df.drop(columns=["stage_num"]), st], axis=1)
    for pen in (0.0, 0.1):
        try:
            cph = CoxPHFitter(penalizer=pen).fit(fit_df, "_lm_months", "_evt")
            res = {}
            for gk in GROUP_ORDER:
                col = f"grp_{gk}"
                if col in cph.params_.index:
                    res[gk] = (float(np.exp(cph.params_[col])),
                               float(np.exp(cph.confidence_intervals_.loc[col, "95% lower-bound"])),
                               float(np.exp(cph.confidence_intervals_.loc[col, "95% upper-bound"])),
                               float(cph.summary.loc[col, "p"]))
            return res
        except Exception:
            continue
    return {}


t5 = []
for gk in GROUP_ORDER:
    sub = df[df["ctdna_dynamics"] == gk]
    t5.append({"Dynamic pattern": LABELS[gk], "All (N=1,010)": len(sub),
               "Long ACT": int((sub["cohort_def"] == "long").sum()),
               "Short ACT": int((sub["cohort_def"] == "short").sum())})
t5.append({"Dynamic pattern": "Total", "All (N=1,010)": len(df),
           "Long ACT": int((df["cohort_def"] == "long").sum()),
           "Short ACT": int((df["cohort_def"] == "short").sum())})
supp5 = pd.DataFrame(t5)
supp5.to_csv(os.path.join(OUT_DIR, "supplementary_table_5_cohort3_dynamics_patterns.csv"), index=False)
print("=" * 90)
print("SUPPLEMENTARY TABLE 5 — serial ctDNA dynamic patterns (N = 1,010)")
print("=" * 90)
print(supp5.to_string(index=False))


lm = landmark_prep(df)
print(f"\nLandmark-analysable cohort: n = {len(lm)}")

A = cox_multivariate(lm)
BE_PANEL = {"sustained_negative": "B", "clearance": "C", "rising": "D", "decrease_detectable": "E"}
BE = {}
for gk in BE_PANEL:
    sub = lm[lm["ctdna_dynamics"] == gk]
    L = sub[sub["cohort_def"] == "long"]; S = sub[sub["cohort_def"] == "short"]
    if len(L) >= MIN_N and len(S) >= MIN_N:
        hr, lo, hi, p = cox_binary(S["_lm_months"].values, S["_evt"].values,
                                   L["_lm_months"].values, L["_evt"].values,
                                   S["stage_num"].values, L["stage_num"].values)
    else:
        hr = lo = hi = p = np.nan
    BE[gk] = dict(hr=hr, lo=lo, hi=hi, p=p, nL=len(L), nS=len(S),
                  eL=int(L["_evt"].sum()), eS=int(S["_evt"].sum()), L=L, S=S)

ddl = lm[(lm["ctdna_dynamics"] == "decrease_detectable") & (lm["cohort_def"] == "long")]
med = ddl["MTM_tp2"].median()
lo_m = ddl[ddl["MTM_tp2"] < med]; hi_m = ddl[ddl["MTM_tp2"] >= med]
fhr, flo, fhi, fp = cox_binary(lo_m["_lm_months"].values, lo_m["_evt"].values,
                               hi_m["_lm_months"].values, hi_m["_evt"].values,
                               lo_m["stage_num"].values, hi_m["stage_num"].values)

rows = []
for gk in ["clearance", "decrease_detectable", "rising"]:
    if gk in A:
        hr, lo, hi, p = A[gk]
        rows.append({"panel": "A", "comparison": f"{LABELS[gk]} vs Sustained neg (MV stage-adj)",
                     "HR": round(hr, 3), "CI_low": round(lo, 3), "CI_high": round(hi, 3),
                     "p": p, "N": int((lm["ctdna_dynamics"] == gk).sum())})
for gk, panel in BE_PANEL.items():
    d = BE[gk]
    rows.append({"panel": panel, "comparison": f"{LABELS[gk]}: Long vs Short",
                 "HR": round(d["hr"], 3) if not np.isnan(d["hr"]) else "",
                 "CI_low": round(d["lo"], 3) if not np.isnan(d["hr"]) else "",
                 "CI_high": round(d["hi"], 3) if not np.isnan(d["hr"]) else "",
                 "p": d["p"], "N": f"L{d['nL']}/S{d['nS']}"})
rows.append({"panel": "F", "comparison": "Decrease (Long only): >=median vs <median TP#3 ctDNA",
             "HR": round(fhr, 3), "CI_low": round(flo, 3), "CI_high": round(fhi, 3),
             "p": fp, "N": len(ddl)})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT_DIR, "figure_4_duration_HRs.csv"), index=False)
print("\nFIGURE 4 — hazard ratios")
with pd.option_context("display.width", 160):
    print(res.to_string(index=False))


def fmt_hr(hr, lo, hi, p):
    return "n/a" if (hr is None or np.isnan(hr)) else f"HR={hr:.2f} ({lo:.2f}\u2013{hi:.2f}) p={fmt_p(p) or '<0.001'}"


max_t = lm["_lm_months"].max()
max_tick = int(np.ceil(max_t / 6) * 6)
xticks = list(np.arange(0, max_tick + 1, 6))
xlim = (-0.5, max_tick + 0.5)
XLAB = "Months since assessment timepoint"

fig = plt.figure(figsize=(20, 15))
gs = fig.add_gridspec(5, 3, height_ratios=[4.0, 1.35, 0.9, 4.0, 1.35],
                      hspace=0.18, wspace=0.18)
fig.suptitle("Figure 4. Serial ctDNA dynamics and adjuvant chemotherapy duration (DFS)",
             fontsize=15, fontweight="bold", y=0.995)


def finish_km(ax, title, ann_text, ylabel="Disease-free survival"):
    ax.set_xlim(xlim); ax.set_xticks(xticks); ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    hide_spines(ax); ax.tick_params(labelbottom=False)
    ax.text(0.02, 0.03, ann_text, transform=ax.transAxes, fontsize=8, va="bottom",
            family="monospace", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.9))
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc="upper right")


axA = fig.add_subplot(gs[0, 0]); rA = fig.add_subplot(gs[1, 0], sharex=axA)
strataA = []
for gk in ["sustained_negative", "clearance", "decrease_detectable", "rising"]:
    sub = lm[lm["ctdna_dynamics"] == gk]
    if len(sub) >= MIN_N:
        draw_km_curve(axA, fit_km(sub["_lm_months"].values, sub["_evt"].values), COLORS[gk],
                      label=f"{LABELS[gk]} (n={len(sub)})")
        strataA.append((LABELS[gk], COLORS[gk], sub["_lm_months"].values))
txtA = [f"Sustained neg (n={int((lm['ctdna_dynamics']=='sustained_negative').sum())}): ref"]
for gk in ["clearance", "decrease_detectable", "rising"]:
    if gk in A:
        txtA.append(f"{LABELS[gk]}: {fmt_hr(*A[gk])}")
finish_km(axA, "A. DFS by ctDNA dynamics\nMultivariable stage-adjusted Cox (ref = Sustained neg)",
          "\n".join(txtA))
draw_risk_table(rA, strataA, xticks, xlim, xlabel=XLAB, fontsize=7.2)

pos = {"sustained_negative": (0, 1), "clearance": (0, 2), "rising": (3, 0), "decrease_detectable": (3, 1)}
for gk, (r, c) in pos.items():
    ax = fig.add_subplot(gs[r, c]); rx = fig.add_subplot(gs[r + 1, c], sharex=ax)
    d = BE[gk]; col = COLORS[gk]; strata = []
    if len(d["L"]) >= MIN_N:
        draw_km_curve(ax, fit_km(d["L"]["_lm_months"].values, d["L"]["_evt"].values), col, ls="-")
        strata.append(("Long", col, d["L"]["_lm_months"].values))
    if len(d["S"]) >= MIN_N:
        draw_km_curve(ax, fit_km(d["S"]["_lm_months"].values, d["S"]["_evt"].values), col, ls="--")
        strata.append(("Short", col, d["S"]["_lm_months"].values))
    finish_km(ax, f"{BE_PANEL[gk]}. {LABELS[gk]}: Long(\u2014) vs Short(--)",
              f"Long vs Short: {fmt_hr(d['hr'], d['lo'], d['hi'], d['p'])}\n"
              f"Long n={d['nL']} (ev {d['eL']}) | Short n={d['nS']} (ev {d['eS']})")
    ax.title.set_color(col)
    draw_risk_table(rx, strata, xticks, xlim, xlabel=XLAB, fontsize=7.2)

axF = fig.add_subplot(gs[3, 2]); rF = fig.add_subplot(gs[4, 2], sharex=axF)
strataF = []
if len(lo_m) and len(hi_m):
    draw_km_curve(axF, fit_km(lo_m["_lm_months"].values, lo_m["_evt"].values), "#1B5E20", ls="-")
    draw_km_curve(axF, fit_km(hi_m["_lm_months"].values, hi_m["_evt"].values), "#008BCE", ls="-")
    strataF = [("< median", "#1B5E20", lo_m["_lm_months"].values),
               ("\u2265 median", "#008BCE", hi_m["_lm_months"].values)]
finish_km(axF, f"F. Decrease-but-detectable (Long only)\nDFS by TP#3 ctDNA (median {med:.3f})",
          f">=med vs <med: {fmt_hr(fhr, flo, fhi, fp)}\n<med n={len(lo_m)} | >=med n={len(hi_m)}")
draw_risk_table(rF, strataF, xticks, xlim, xlabel=XLAB, fontsize=7.2)

fig.subplots_adjust(top=0.95)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT_DIR, f"figure_4_duration_km_panels.{ext}"), dpi=200,
                bbox_inches="tight", facecolor="white")
plt.close(fig)
print("\nSaved Figure 4 panels and all Cohort 3 outputs to:", OUT_DIR)
