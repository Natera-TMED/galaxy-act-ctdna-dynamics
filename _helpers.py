import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

warnings.filterwarnings("ignore")

BASE          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR       = os.path.join(BASE, "manuscript_final_code", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

DAYS_PER_MONTH = 30.44
ID_COLS = ["Bundling.ID", "GalaxyID"]


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def hide_spines(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def simplify_stage(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    if s.startswith("IV") or s == "Mts/Rec":
        return 4
    if s.startswith("III"):
        return 3
    if s.startswith("II"):
        return 2
    if s.startswith("I"):
        return 1
    return np.nan


def stage_dummies(stage_series, prefix="stage"):
    STAGE_I_MERGE_MIN = 10
    s_num = pd.to_numeric(stage_series, errors="coerce")
    if 0 < int((s_num == 1).sum()) < STAGE_I_MERGE_MIN:
        stage_series = s_num.replace({1: 2})
    s = stage_series.dropna().astype(int)
    if s.nunique() < 2:
        return pd.DataFrame(index=stage_series.index)
    dd = pd.get_dummies(stage_series.astype("Int64"), prefix=prefix).astype(float)
    ref_col = sorted(dd.columns, key=lambda c: int(c.split("_")[-1]))[0]
    return dd.drop(columns=[ref_col])


def _pt_grp(x):
    s = str(x)
    if s.startswith("pT4"):
        return "T4"
    for k in ("pT1", "pT2", "pT3"):
        if s.startswith(k):
            return k[1:]
    return np.nan


def _pn_grp(x):
    s = str(x)
    for k in ("pN0", "pN1", "pN2"):
        if s.startswith(k):
            return k[1:]
    return np.nan


def _lvi(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    return "Present" if ("Ly1" in s or "V1" in s) else "Absent"


def _pni(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    if s.startswith("Pn1"):
        return "Present"
    if s.startswith("Pn0"):
        return "Absent"
    return np.nan


def derive_baseline(m, mtm_col):
    m = m.copy()
    m["age"] = pd.to_numeric(m["Age"], errors="coerce")
    m["age_grp"] = pd.cut(m["age"], [-np.inf, 65, 75, np.inf],
                          labels=["<65", "65\u201374", "\u226575"], right=False)
    m["sex"] = m["Sex"]
    ecog_n = pd.to_numeric(m["ECOG"], errors="coerce")
    m["ecog"] = np.where(ecog_n >= 1, "\u22651", "0")
    m.loc[ecog_n.isna(), "ecog"] = np.nan
    m["stage"] = m["stage_num"].map({1: "I", 2: "II", 3: "III", 4: "IV"})
    m["pT_grp"] = m["pT"].apply(_pt_grp)
    m["pN_grp"] = m["pN"].apply(_pn_grp)
    m["sidedness"] = m["LocationGrpA"].replace("Unknown", np.nan)
    m["primary"]   = m["LocationGrpB"]
    m["msi"]  = m["msi_call"].map({"MSS": "MSS", "MSI-High": "MSI-H"})
    m["ras"]  = m["ras_call"].map({"WT": "Negative", "MUT": "Positive"})
    m["braf"] = m["braf_call"].map({"WT": "Negative", "MUT": "Positive"})
    m["lvi"] = m["p_LVI"].apply(_lvi)
    m["pni"] = m["p_PNI"].apply(_pni)
    m["mrd"] = np.where(m["mrd_pos"], "Positive", "Negative")
    mtm = pd.to_numeric(m[mtm_col], errors="coerce")
    m["mrd_mtm_pos"] = np.where(m["mrd_pos"], mtm, np.nan)
    m["fu_months"] = pd.to_numeric(m["p_drelOS"], errors="coerce") / DAYS_PER_MONTH
    m["os_evt"] = pd.to_numeric(m["p_evtOS"], errors="coerce")
    return m


def _cont_p(samples):
    samples = [x for x in samples if len(x) > 0]
    if len(samples) < 2:
        return np.nan
    try:
        if len(samples) == 2:
            return sps.mannwhitneyu(samples[0], samples[1], alternative="two-sided").pvalue
        return sps.kruskal(*samples).pvalue
    except Exception:
        return np.nan


def _cat_p(tab):
    if tab.shape[0] < 2 or tab.shape[1] < 2 or tab.values.sum() == 0:
        return np.nan
    try:
        if tab.shape == (2, 2):
            return sps.fisher_exact(tab.values)[1]
        return sps.chi2_contingency(tab.values)[1]
    except Exception:
        return np.nan


def _cont_stat(sr, dec=0):
    sr = sr.dropna()
    if sr.empty:
        return ""
    return f"{sr.median():.{dec}f} [{sr.quantile(.25):.{dec}f}\u2013{sr.quantile(.75):.{dec}f}]"


def _cont_row(rows, cols, groups, label, col, dec=0, show_p=True):
    r = {"Characteristic": label}
    for name, gdf in cols:
        r[name] = _cont_stat(gdf[col], dec)
    r["p"] = fmt_p(_cont_p([gdf[col].dropna().values for gdf in groups])) if show_p else "\u2013"
    rows.append(r)


def _cat_rows(rows, cols, groups, label, col, order, show_missing=True):
    rows.append({"Characteristic": label, **{c[0]: "" for c in cols}, "p": ""})
    header_idx = len(rows) - 1
    for lv in order:
        r = {"Characteristic": f"  {lv}"}
        for name, gdf in cols:
            n_tot = len(gdf)
            n_lv = int((gdf[col] == lv).sum())
            r[name] = f"{n_lv} ({100*n_lv/n_tot:.1f}%)" if n_tot else "0"
        r["p"] = ""
        rows.append(r)
    tab = pd.DataFrame({i: groups[i][col].value_counts() for i in range(len(groups))}).fillna(0)
    tab = tab.loc[[lv for lv in order if lv in tab.index]]
    rows[header_idx]["p"] = fmt_p(_cat_p(tab))
    if show_missing:
        r = {"Characteristic": "  Missing"}
        for name, gdf in cols:
            n_tot = len(gdf)
            n_miss = int(gdf[col].isna().sum())
            r[name] = f"{n_miss} ({100*n_miss/n_tot:.1f}%)" if n_tot else "0"
        r["p"] = ""
        rows.append(r)


def build_demographics_table(cols, groups, include_mrd_row):
    rows = []
    rows.append({"Characteristic": "N", **{name: f"{len(gdf):,}" for name, gdf in cols}, "p": ""})
    _cont_row(rows, cols, groups, "Follow-up time in months, median [IQR]", "fu_months", dec=1, show_p=False)
    if include_mrd_row:
        _cat_rows(rows, cols, groups, "MRD-window ctDNA status", "mrd",
                  order=["Negative", "Positive"], show_missing=False)
    _cont_row(rows, cols, groups, "Age, median [IQR], y", "age")
    _cat_rows(rows, cols, groups, "Age group", "age_grp",
              order=["<65", "65\u201374", "\u226575"], show_missing=False)
    _cat_rows(rows, cols, groups, "Sex", "sex", order=["Female", "Male"], show_missing=False)
    _cat_rows(rows, cols, groups, "ECOG performance status", "ecog", order=["0", "\u22651"])
    _cat_rows(rows, cols, groups, "Pathologic stage", "stage", order=["I", "II", "III", "IV"])
    _cat_rows(rows, cols, groups, "Primary tumor (pT)", "pT_grp", order=["T1", "T2", "T3", "T4"])
    _cat_rows(rows, cols, groups, "Nodal status (pN)", "pN_grp", order=["N0", "N1", "N2"])
    _cat_rows(rows, cols, groups, "Tumor sidedness", "sidedness",
              order=["Left-sided colon", "Right-sided colon"])
    _cat_rows(rows, cols, groups, "Primary site", "primary", order=["Colon", "Rectum"])
    _cat_rows(rows, cols, groups, "MMR/MSI status", "msi", order=["MSS", "MSI-H"])
    _cat_rows(rows, cols, groups, "RAS", "ras", order=["Negative", "Positive"])
    _cat_rows(rows, cols, groups, "BRAF", "braf", order=["Negative", "Positive"])
    _cat_rows(rows, cols, groups, "Lymphovascular invasion", "lvi", order=["Absent", "Present"])
    _cat_rows(rows, cols, groups, "Perineural invasion", "pni", order=["Absent", "Present"])
    _cont_row(rows, cols, groups, "MRD-window MTM among MRD+, median [IQR]", "mrd_mtm_pos", dec=2)
    return pd.DataFrame(rows)[["Characteristic"] + [c[0] for c in cols] + ["p"]]


def fit_km(t, e, label=""):
    return KaplanMeierFitter(label=label).fit(t, event_observed=e)


def draw_risk_table(risk_ax, strata, xticks, xlim, show_xaxis=True, xlabel=None,
                    fontsize=7.5, title="No. at risk"):
    n = len(strata)
    risk_ax.set_xlim(xlim)
    risk_ax.set_ylim(0, n)
    for i, (label, color, t) in enumerate(strata):
        y = n - i - 0.5
        tarr = np.asarray([x for x in t if np.isfinite(x)], float)
        for xt in xticks:
            risk_ax.text(xt, y, str(int(np.sum(tarr >= xt))),
                         ha="center", va="center", fontsize=fontsize, color="#222222")
    risk_ax.set_yticks([n - i - 0.5 for i in range(n)])
    risk_ax.set_yticklabels([s[0] for s in strata], fontsize=fontsize)
    for tick, (label, color, t) in zip(risk_ax.get_yticklabels(), strata):
        tick.set_color(color)
    risk_ax.set_xticks(xticks)
    if show_xaxis:
        risk_ax.set_xticklabels([str(int(x)) for x in xticks], fontsize=fontsize + 0.5)
        if xlabel:
            risk_ax.set_xlabel(xlabel)
    else:
        risk_ax.set_xticklabels([])
    risk_ax.tick_params(axis="both", length=0)
    for s in ("top", "right", "left", "bottom"):
        risk_ax.spines[s].set_visible(False)
    risk_ax.set_title(title, loc="left", fontsize=fontsize + 1, fontweight="bold", pad=3)


def draw_km_curve(ax, kmf, color, lw=2.2, ls="-", alpha=1.0, label=None):
    sf = kmf.survival_function_
    ax.step(np.array(sf.index), np.array(sf.iloc[:, 0]), where="post",
            color=color, lw=lw, ls=ls, alpha=alpha, solid_capstyle="butt",
            label=label)
    for c in kmf.durations[kmf.event_observed == 0]:
        try:
            ax.plot(c, float(kmf.predict(c)), "|", color=color, ms=7, mew=1.3, alpha=alpha)
        except Exception:
            pass


def landmark_frame(df, landmark_d, endpoint):
    dur_col = "dfs_days" if endpoint == "DFS" else "os_days"
    evt_col = "dfs_evt" if endpoint == "DFS" else "os_evt"
    d = df[df[dur_col].notna()].copy()
    d = d[d[dur_col] >= landmark_d].copy()
    d["_t"] = (d[dur_col] - landmark_d) / DAYS_PER_MONTH
    d["_e"] = pd.to_numeric(d[evt_col], errors="coerce").astype(int)
    same = (d["_t"] == 0) & (d["_e"] == 1)
    d.loc[same, "_t"] = 1.0 / DAYS_PER_MONTH
    return d


def landmark_cox_adjusted(d, present_arms, ref_key, covs=("stage", "age", "ecog", "mtm")):
    dd = d[d["arm"].isin(present_arms)].copy()
    if "mtm" in covs:
        dd["log_mtm"] = np.log1p(pd.to_numeric(dd["mrd_mtm"], errors="coerce"))
    req = []
    if "stage" in covs: req.append("stage_num")
    if "age" in covs:   req.append("age")
    if "ecog" in covs:  req.append("ecog_ge1")
    if "mtm" in covs:   req.append("log_mtm")
    dd = dd.dropna(subset=["_t", "_e"] + req).copy()

    parts = []
    if "stage" in covs: parts.append(stage_dummies(dd["stage_num"], prefix="stage"))
    if "age" in covs:   parts.append(dd[["age"]].astype(float))
    if "ecog" in covs:  parts.append(dd[["ecog_ge1"]].astype(float))
    if "mtm" in covs:   parts.append(dd[["log_mtm"]].astype(float))
    adj = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=dd.index)

    dummies = pd.get_dummies(dd["arm"], prefix="g").astype(float)
    ref_col = f"g_{ref_key}"
    if ref_col in dummies.columns:
        dummies = dummies.drop(columns=[ref_col])
    n_arm = dd["arm"].value_counts().to_dict()
    ev_arm = dd.groupby("arm")["_e"].sum().to_dict()
    fit_df = pd.concat([dd[["_t", "_e"]].reset_index(drop=True),
                        dummies.reset_index(drop=True),
                        adj.reset_index(drop=True)], axis=1)
    results = {}
    for pen in (0.0, 0.1, 0.5):
        try:
            cph = CoxPHFitter(penalizer=pen).fit(fit_df, duration_col="_t", event_col="_e")
            for col in dummies.columns:
                lvl = col[2:]
                hr = float(np.exp(cph.params_[col]))
                lo = float(np.exp(cph.confidence_intervals_.loc[col, "95% lower-bound"]))
                hi = float(np.exp(cph.confidence_intervals_.loc[col, "95% upper-bound"]))
                p = float(cph.summary.loc[col, "p"])
                results[lvl] = (hr, lo, hi, p, pen > 0,
                                int(n_arm.get(lvl, 0)), int(ev_arm.get(lvl, 0)))
            return results
        except Exception:
            continue
    return results


def _km_area(t, e, tau):
    t = np.asarray(t, float); e = np.asarray(e, int)
    if len(t) == 0:
        return 0.0
    order = np.argsort(t, kind="mergesort")
    ts, es = t[order], e[order]
    n = len(ts)
    et = np.unique(ts[(es == 1) & (ts <= tau)])
    if len(et) == 0:
        return float(tau)
    nrisk = n - np.searchsorted(ts, et, side="left")
    ev_all = ts[es == 1]
    d = (np.searchsorted(ev_all, et, side="right")
         - np.searchsorted(ev_all, et, side="left"))
    S_steps = np.cumprod(1.0 - d / nrisk)
    Svals = np.concatenate([[1.0], S_steps])
    bounds = np.concatenate([[0.0], et, [tau]])
    seg = np.clip(bounds[1:], None, tau) - np.clip(bounds[:-1], None, tau)
    return float(np.sum(Svals * np.maximum(seg, 0.0)))


def _pseudo_rmst(t, e, tau):
    t = np.asarray(t, float); e = np.asarray(e, int); n = len(t)
    theta = _km_area(t, e, tau)
    ps = np.empty(n); keep = np.ones(n, bool)
    for i in range(n):
        keep[i] = False
        ps[i] = n * theta - (n - 1) * _km_area(t[keep], e[keep], tau)
        keep[i] = True
    return ps


def _design(dd, arms, ref_key, covs):
    order = [a for a in arms if a != ref_key]
    cols = [np.ones(len(dd))]
    for a in order:
        cols.append((dd["arm"].values == a).astype(float))
    if "stage" in covs:
        sd = stage_dummies(dd["stage_num"], prefix="stage")
        for c in sd.columns:
            cols.append(sd[c].values.astype(float))
    if "age" in covs:
        cols.append(dd["age"].values.astype(float))
    if "ecog" in covs:
        cols.append(dd["ecog_ge1"].values.astype(float))
    if "mtm" in covs:
        cols.append(np.log1p(dd["mrd_mtm"].values.astype(float)))
    return np.column_stack(cols), order


def adj_rmst_contrasts(d, arms, ref_key, covs, horizon):
    req = ["_t", "_e"] + (["stage_num"] if "stage" in covs else []) \
        + (["age"] if "age" in covs else []) + (["ecog_ge1"] if "ecog" in covs else []) \
        + (["mrd_mtm"] if "mtm" in covs else [])
    dd = d[d["arm"].isin(arms)].dropna(subset=req).reset_index(drop=True)
    y = _pseudo_rmst(dd["_t"].values, dd["_e"].values, horizon)
    X, order = _design(dd, arms, ref_key, covs)
    n, k = X.shape
    XtXinv = np.linalg.pinv(X.T @ X)
    beta = XtXinv @ (X.T @ y)
    resid = y - X @ beta
    meat = X.T @ (X * (resid ** 2)[:, None])
    scale = n / max(n - k, 1)
    cov = XtXinv @ meat @ XtXinv * scale
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    out = {}
    for i, a in enumerate(order):
        est = float(beta[1 + i]); s = float(se[1 + i])
        n_a = int((dd["arm"] == a).sum()); ev_a = int(dd.loc[dd["arm"] == a, "_e"].sum())
        if n_a < 8 or ev_a < 3 or s <= 0:
            out[a] = (est, np.nan, np.nan, np.nan)
        else:
            out[a] = (est, est - 1.96 * s, est + 1.96 * s, 2 * sps.norm.sf(abs(est) / s))
    return out


def _rmst_arm(t, e, horizon):
    t = np.asarray(t, float); e = np.asarray(e, int)
    ev_times = np.unique(t[e == 1]); ev_times = ev_times[ev_times <= horizon]
    segs, pt, S = [], 0.0, 1.0
    d_arr, n_arr, ct_arr = [], [], []
    for ut in ev_times:
        n_i = int(np.sum(t >= ut)); d_i = int(np.sum((t == ut) & (e == 1)))
        segs.append((pt, ut, S))
        S = S * (1 - d_i / n_i) if n_i > 0 else S
        pt = ut; ct_arr.append(ut); d_arr.append(d_i); n_arr.append(n_i)
    segs.append((pt, horizon, S))

    def area_from(a):
        tot = 0.0
        for s0, s1, sv in segs:
            lo, hi = max(s0, a), min(s1, horizon)
            if hi > lo:
                tot += sv * (hi - lo)
        return tot

    rmst = area_from(0.0); var = 0.0
    for k2, ct in enumerate(ct_arr):
        n_i, d_i = n_arr[k2], d_arr[k2]
        if d_i > 0 and n_i > d_i:
            A = area_from(ct); var += (A ** 2) * d_i / (n_i * (n_i - d_i))
    return float(rmst), float(var)


def rmst_difference(ref_t, ref_e, cmp_t, cmp_e, horizon):
    r_ref, v_ref = _rmst_arm(ref_t, ref_e, horizon)
    r_cmp, v_cmp = _rmst_arm(cmp_t, cmp_e, horizon)
    diff = r_cmp - r_ref
    if np.isnan(v_ref) or np.isnan(v_cmp) or (v_ref + v_cmp) <= 0:
        return diff, np.nan, np.nan, np.nan
    se = np.sqrt(v_ref + v_cmp)
    return diff, diff - 1.96 * se, diff + 1.96 * se, 2 * sps.norm.sf(abs(diff) / se)


def mtm_hi_lo_cox(dsub):
    d = dsub.dropna(subset=["_t", "_e", "stage_num", "age", "ecog_ge1"]).copy()
    d["hi"] = (d["mtm_grp"] == "high").astype(float)
    lo_m = d[d["hi"] == 0]; hi_m = d[d["hi"] == 1]
    n_lo, n_hi = len(lo_m), len(hi_m)
    ev_lo, ev_hi = int(lo_m["_e"].sum()), int(hi_m["_e"].sum())
    lr_p = np.nan
    if n_lo >= 3 and n_hi >= 3:
        try:
            lr_p = float(logrank_test(lo_m["_t"], hi_m["_t"], lo_m["_e"], hi_m["_e"]).p_value)
        except Exception:
            pass
    if n_lo < 8 or n_hi < 8 or (ev_lo + ev_hi) < 3 or d["hi"].nunique() < 2:
        return (np.nan,) * 5 + (n_lo, n_hi, ev_lo, ev_hi, lr_p)
    sd = stage_dummies(d["stage_num"], prefix="stage")
    fit_df = pd.concat([d[["_t", "_e", "hi", "age", "ecog_ge1"]].reset_index(drop=True),
                        sd.reset_index(drop=True)], axis=1)
    for pen in (0.0, 0.1, 0.5, 1.0):
        try:
            cph = CoxPHFitter(penalizer=pen).fit(fit_df, duration_col="_t", event_col="_e")
            hr = float(np.exp(cph.params_["hi"]))
            lo = float(np.exp(cph.confidence_intervals_.loc["hi", "95% lower-bound"]))
            hi = float(np.exp(cph.confidence_intervals_.loc["hi", "95% upper-bound"]))
            p = float(cph.summary.loc["hi", "p"])
            if np.isfinite(hr) and np.isfinite(lo) and np.isfinite(hi) and lo > 0:
                return hr, lo, hi, p, pen > 0, n_lo, n_hi, ev_lo, ev_hi, lr_p
        except Exception:
            continue
    return (np.nan,) * 5 + (n_lo, n_hi, ev_lo, ev_hi, lr_p)


def plot_forest(items, title, out_name, xlabel="Hazard ratio (95% CI, log scale)",
                subtitle="", footer=""):
    items = [it for it in items if it[1] is not None and not np.isnan(it[1])]
    if not items:
        return None
    n = len(items)
    fig, ax = plt.subplots(figsize=(9.6, 0.52 * n + 2.0))
    ypos = np.arange(n)[::-1]
    for y, (lab, hr, lo, hi, p) in zip(ypos, items):
        lo_p = lo if (lo and lo > 0 and not np.isnan(lo)) else hr
        hi_p = hi if (hi and not np.isnan(hi)) else hr
        col = "#B00020" if (p is not None and p < 0.05) else "#333333"
        ax.plot([lo_p, hi_p], [y, y], "-", color=col, lw=1.8)
        ax.plot(hr, y, "s", color=col, ms=7)
    ax.axvline(1.0, color="#888888", lw=1.0, ls="--")
    ax.set_ylim(-1.4, n - 0.4)
    ax.set_yticks(ypos)
    ax.set_yticklabels([it[0] for it in items], fontsize=9)
    ax.set_xscale("log")
    xr = [it[2] for it in items if it[2] and it[2] > 0] + [it[3] for it in items if it[3]]
    lo_lim = max(0.02, min(xr) * 0.7) if xr else 0.1
    hi_lim = min(60, max(xr) * 1.3) if xr else 10
    ax.set_xlim(lo_lim, hi_lim)
    ax.set_xlabel(xlabel)
    ax.set_title(title + ("\n" + subtitle if subtitle else ""), fontsize=11)
    hide_spines(ax)
    xtext = hi_lim * 1.02
    for y, (lab, hr, lo, hi, p) in zip(ypos, items):
        ax.text(xtext, y, f"{hr:.2f} ({lo:.2f}\u2013{hi:.2f})  p={fmt_p(p) or '<0.001'}",
                va="center", ha="left", fontsize=8, family="monospace")
    if footer:
        fig.text(0.01, 0.005, footer, fontsize=6.8, color="grey", va="bottom")
    fig.subplots_adjust(right=0.62, left=0.32, top=0.86, bottom=0.14)
    paths = []
    for e in ("png", "pdf"):
        p_out = os.path.join(OUT_DIR, f"{out_name}.{e}")
        fig.savefig(p_out, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(p_out)
    plt.close(fig)
    return paths[0]
