import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _common import (OUT_DIR, DAYS_PER_MONTH, load_act_timing_cohort, simplify_stage,
                     fit_km, draw_km_curve, draw_risk_table, landmark_frame,
                     landmark_cox_adjusted, adj_rmst_contrasts, rmst_difference,
                     mtm_hi_lo_cox, fmt_p, hide_spines)

LM = 84
COVS = ("stage", "age", "ecog", "mtm")
ARM_LBL = {"no_ACT": "No ACT", "early": "ACT 0\u20138w", "delayed": "ACT 8\u201312w"}
ARM_COL = {"no_ACT": "#9E9E9E", "early": "#1B5E20", "delayed": "#008BCE"}
ARMS = ["no_ACT", "early", "delayed"]

m = load_act_timing_cohort()
m["stage_num"] = m["p_StagePath"].apply(simplify_stage)
m["age"] = pd.to_numeric(m["Age"], errors="coerce")
m["ecog_ge1"] = np.where(pd.to_numeric(m["ECOG"], errors="coerce") >= 1, 1.0, 0.0)
m.loc[m["ECOG"].isna(), "ecog_ge1"] = np.nan
pos = m[m["mrd_pos"]].copy()
print(f"TP#1 MRD-positive cohort: N = {len(pos)}")


def km_panel(endpoint, panel, title):
    d = landmark_frame(pos, LM, endpoint)
    hr_vs_noACT = landmark_cox_adjusted(d, ARMS, "no_ACT", COVS)
    hr_e_vs_d   = landmark_cox_adjusted(d, ["early", "delayed"], "early", COVS)

    xmax = int(np.ceil(d["_t"].max() / 6) * 6)
    xticks = list(range(0, xmax + 1, 6))
    fig = plt.figure(figsize=(8.2, 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[4.0, 1.15], hspace=0.07)
    ax = fig.add_subplot(gs[0]); rax = fig.add_subplot(gs[1], sharex=ax)

    strata = []
    for a in ARMS:
        sub = d[d["arm"] == a]
        if len(sub) == 0:
            continue
        kmf = fit_km(sub["_t"].values, sub["_e"].values,
                     label=f"{ARM_LBL[a]} (n={len(sub)}, ev={int(sub['_e'].sum())})")
        draw_km_curve(ax, kmf, ARM_COL[a], label=f"{ARM_LBL[a]} (n={len(sub)}, ev={int(sub['_e'].sum())})")
        strata.append((ARM_LBL[a], ARM_COL[a], sub["_t"].values))
    ax.set_xlim(-1, xmax + 1)
    ax.set_ylabel(f"{endpoint} probability")
    ax.set_ylim(-0.02, 1.03)
    ax.set_title(title, fontsize=12, fontweight="bold")
    hide_spines(ax)
    ax.tick_params(labelbottom=False)
    ax.legend(loc="lower left", fontsize=8.5, frameon=False)

    de = hr_e_vs_d.get("delayed")
    e_vs_d = (1 / de[0], 1 / de[2], 1 / de[1], de[3]) if de else None
    e = hr_vs_noACT.get("early"); dd = hr_vs_noACT.get("delayed")
    lines = []
    if e:
        lines.append(f"ACT 0\u20138w vs No ACT : HR {e[0]:.2f} ({e[1]:.2f}\u2013{e[2]:.2f}), p={fmt_p(e[3]) or '<0.001'}")
    if dd:
        lines.append(f"ACT 8\u201312w vs No ACT: HR {dd[0]:.2f} ({dd[1]:.2f}\u2013{dd[2]:.2f}), p={fmt_p(dd[3]) or '<0.001'}")
    if e_vs_d:
        lines.append(f"ACT 0\u20138w vs 8\u201312w : HR {e_vs_d[0]:.2f} ({e_vs_d[1]:.2f}\u2013{e_vs_d[2]:.2f}), p={fmt_p(e_vs_d[3]) or '<0.001'}")
    ax.text(0.97, 0.52, "Adjusted HR (stage+age+ECOG+log(1+MTM))\n" + "\n".join(lines),
            transform=ax.transAxes, fontsize=8.2, va="top", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.9))

    draw_risk_table(rax, strata, xticks, (-1, xmax + 1),
                    xlabel="Months from 12-week landmark")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"figure_2{panel}_{endpoint}_landmark_km.{ext}"),
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    rows = []
    for ref, cmp, tup in [("No ACT", "ACT 0\u20138w", e), ("No ACT", "ACT 8\u201312w", dd),
                          ("ACT 8\u201312w", "ACT 0\u20138w", e_vs_d)]:
        if tup:
            rows.append({"panel": f"2{panel}", "endpoint": endpoint, "reference": ref,
                         "comparison": cmp, "HR": round(tup[0], 3), "CI_low": round(tup[1], 3),
                         "CI_high": round(tup[2], 3), "p": round(tup[3], 5)})
    return pd.DataFrame(rows)


tab_dfs = km_panel("DFS", "A", "Figure 2A. DFS by ACT timing (MRD-positive)")
tab_os  = km_panel("OS",  "B", "Figure 2B. OS by ACT timing (MRD-positive)")
hr_tab = pd.concat([tab_dfs, tab_os], ignore_index=True)
hr_tab.to_csv(os.path.join(OUT_DIR, "figure_2AB_landmark_adjusted_HRs.csv"), index=False)
print("\nFigure 2A / 2B — adjusted landmark hazard ratios")
print(hr_tab.to_string(index=False))


d_dfs = landmark_frame(pos, LM, "DFS")
present = [a for a in ["no_ACT", "early", "delayed", "late"]
           if int((d_dfs["arm"] == a).sum()) >= 5]
grid_max = min(float(d_dfs[d_dfs["arm"] == a]["_t"].max()) for a in present)
horizons = list(range(6, int(grid_max // 6) * 6 + 1, 6))
ref_t = d_dfs[d_dfs["arm"] == "no_ACT"]["_t"].values
ref_e = d_dfs[d_dfs["arm"] == "no_ACT"]["_e"].values

CURVES = [("early", "ACT 0\u20138w", "#1B5E20"), ("delayed", "ACT 8\u201312w", "#008BCE")]
gain = {a: {"adj": [], "un": []} for a, _, _ in CURVES}
direct = {"Early vs No ACT": {}, "Delayed vs No ACT": {}, "Early vs Delayed": {}}

for h in horizons:
    vs_no = adj_rmst_contrasts(d_dfs, present, "no_ACT", COVS, h)
    for a, _, _ in CURVES:
        sub = d_dfs[d_dfs["arm"] == a]
        gain[a]["adj"].append(vs_no[a][0])
        gain[a]["un"].append(rmst_difference(ref_t, ref_e, sub["_t"].values, sub["_e"].values, h)[0])
    direct["Early vs No ACT"][h] = vs_no["early"]
    direct["Delayed vs No ACT"][h] = vs_no["delayed"]
    direct["Early vs Delayed"][h] = adj_rmst_contrasts(d_dfs, ["delayed", "early"], "delayed", COVS, h)["early"]

fig, ax = plt.subplots(figsize=(10.5, 5.6))
for a, lbl, color in CURVES:
    ax.plot(horizons, gain[a]["adj"], "-o", color=color, lw=2.6, ms=5, label=f"{lbl} (adjusted)")
    ax.plot(horizons, gain[a]["un"], ":", color=color, lw=1.6, alpha=0.7, label=f"{lbl} (unadjusted)")
ax.axhline(0, color="#555555", lw=1.0, ls="--")
ax.set_xlabel("Months from 12-week landmark")
ax.set_ylabel("RMST gain vs No ACT (DFS, months)")
ax.set_title("Supplementary Figure 1. RMST Difference in Disease-Free Survival by ACT Initiation Timing",
             fontsize=11.5, fontweight="bold")
ax.set_xticks(horizons); ax.grid(axis="y", color="#DDDDDD", lw=0.8)
hide_spines(ax)
ax.legend(loc="upper left", fontsize=9, frameon=False)

blocks = [("Early vs Delayed ACT: adj. RMST (direct):", "Early vs Delayed"),
          ("Early vs No ACT: adj. RMST (direct):", "Early vs No ACT"),
          ("Delayed vs No ACT: adj. RMST (direct):", "Delayed vs No ACT")]
ann = []
for head, key in blocks:
    ann.append(head)
    for H in (24, 36):
        diff, lo, hi, p = direct[key][H]
        pp = "P<0.001" if (p is not None and p < 0.001) else f"p = {p:.3f}"
        ann.append(f"  {H} mo: {diff:+.2f} ({lo:.2f} to {hi:.2f}), {pp}")
    ann.append("")
ax.text(1.02, 0.5, "\n".join(ann).rstrip(), transform=ax.transAxes, fontsize=8.4,
        va="center", ha="left", family="monospace")
fig.subplots_adjust(right=0.63)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT_DIR, f"supplementary_figure_1_rmst_gain_DFS.{ext}"),
                dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

rows24_36 = []
for head, key in blocks:
    for H in (24, 36):
        diff, lo, hi, p = direct[key][H]
        rows24_36.append({"contrast": key, "horizon_mo": H, "RMST_diff_mo": round(diff, 2),
                          "CI_low": round(lo, 2), "CI_high": round(hi, 2), "p": round(p, 4)})
rmst_df = pd.DataFrame(rows24_36)
rmst_df.to_csv(os.path.join(OUT_DIR, "supplementary_figure_1_rmst_contrasts.csv"), index=False)
print("\nSupplementary Figure 1 — direct adjusted RMST contrasts (24 & 36 months)")
print(rmst_df.to_string(index=False))


mtm_median = pos["mrd_mtm"].median()
pos["mtm_grp"] = np.where(pos["mrd_mtm"] >= mtm_median, "high", "low")
print(f"\nMedian TP#1 ctDNA burden across the MRD-positive cohort = {mtm_median:.3f} MTM/mL")

STRATA = [("no_ACT", "low"), ("no_ACT", "high"),
          ("early", "low"), ("early", "high"),
          ("delayed", "low"), ("delayed", "high")]
BURDEN_LS = {"low": "-", "high": "--"}
BURDEN_TAG = {"low": "Low burden", "high": "High burden"}
ROW_LBL = {"no_ACT": "No ACT", "early": "ACT 0\u20138w", "delayed": "ACT 8\u201312w"}

fig = plt.figure(figsize=(21, 9.2))
gs = fig.add_gridspec(2, 2, height_ratios=[4.0, 2.1], hspace=0.08, wspace=0.14)
fig.suptitle(f"Supplementary Figure 2. DFS (A) and OS (B) by ACT initiation timing and TP#1 ctDNA burden "
             f"(median split at {mtm_median:.3f} MTM/mL; within-arm high-vs-low HR adjusted for stage + age + ECOG\u22651)",
             fontsize=12, fontweight="bold")

mtm_rows = []
for col, (endpoint, panel) in enumerate([("DFS", "A"), ("OS", "B")]):
    d2 = landmark_frame(pos, LM, endpoint)
    xmax = int(np.ceil(d2["_t"].max() / 6) * 6)
    xticks = list(range(0, xmax + 1, 6))
    ax = fig.add_subplot(gs[0, col]); rax = fig.add_subplot(gs[1, col], sharex=ax)

    risk_strata = []
    for arm, grp in STRATA:
        sub = d2[(d2["arm"] == arm) & (d2["mtm_grp"] == grp)]
        if len(sub) == 0:
            continue
        lbl = f"{ROW_LBL[arm]} / {BURDEN_TAG[grp]} (n={len(sub)})"
        draw_km_curve(ax, fit_km(sub["_t"].values, sub["_e"].values), ARM_COL[arm],
                      ls=BURDEN_LS[grp], label=lbl)
        risk_strata.append((f"{ROW_LBL[arm]} / {grp[0].upper()}", ARM_COL[arm], sub["_t"].values))

    ann_lines = []
    for arm in ARMS:
        hr, lo, hi, p, pen, n_lo, n_hi, ev_lo, ev_hi, lr_p = mtm_hi_lo_cox(d2[d2["arm"] == arm])
        if not np.isnan(hr):
            ann_lines.append((ARM_COL[arm],
                              f"{ROW_LBL[arm]}: high vs low ctDNA burden  HR {hr:.2f}, 95% CI {lo:.2f}\u2013{hi:.2f}, P={p:.3f}"))
        mtm_rows.append({"arm": ROW_LBL[arm], "endpoint": endpoint, "n_low": n_lo, "n_high": n_hi,
                         "ev_low": ev_lo, "ev_high": ev_hi,
                         "HR_high_vs_low": round(hr, 3) if not np.isnan(hr) else "",
                         "CI_low": round(lo, 3) if not np.isnan(hr) else "",
                         "CI_high": round(hi, 3) if not np.isnan(hr) else "",
                         "adj_p": round(p, 4) if not np.isnan(p) else "",
                         "logrank_p": round(lr_p, 4) if not np.isnan(lr_p) else ""})
    for i, (c, txt) in enumerate(ann_lines):
        ax.text(0.5, 0.97 - i * 0.055, txt, transform=ax.transAxes, fontsize=8.6,
                va="top", ha="center", color=c, family="monospace")

    ax.set_xlim(-1, xmax + 1); ax.set_ylim(-0.02, 1.03)
    ax.set_ylabel(f"{endpoint} probability", fontsize=12)
    ax.set_title(f"{panel}. MRD-positive patients (all stages) \u2013 {endpoint} by ACT timing and TP#1 ctDNA burden",
                 fontsize=10.5, fontweight="bold")
    hide_spines(ax); ax.tick_params(labelbottom=False)
    ax.legend(loc="lower left", fontsize=7.6, frameon=False, ncol=1)
    draw_risk_table(rax, risk_strata, xticks, (-1, xmax + 1),
                    xlabel="Months from 12-week landmark", fontsize=7.5)

fig.subplots_adjust(top=0.91)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT_DIR, f"supplementary_figure_2_ctdna_burden_km.{ext}"),
                dpi=250, bbox_inches="tight", facecolor="white")
plt.close(fig)

mtm_df = pd.DataFrame(mtm_rows)
mtm_df.to_csv(os.path.join(OUT_DIR, "supplementary_figure_2_ctdna_burden_HRs.csv"), index=False)
print("\nSupplementary Figure 2 — within-arm high vs low ctDNA-burden HRs (DFS and OS)")
print(mtm_df.to_string(index=False))
