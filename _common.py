import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PATIENTS_FILE = os.path.join(DATA, "galaxy_cohort_demographics.csv")
SAMPLES_FILE = os.path.join(DATA, "galaxy_cohort_testResult.csv")

MRD_LO, MRD_HI = 14, 70
EARLY_HI, DELAYED_HI, LATE_HI = 56, 84, 168
LONG_ACT_MIN_DAYS = 90
MIN_ACT_DAYS = 14
ACT_START_MAX_DAYS = 120
MIN_FU_AFTER_TP2 = 180
ACT_ARMS = ["early", "delayed", "late", "no_ACT"]

_EVALUABLE = None
_COHORT3 = None


def _select_tp1(ps, adj_start, has_act):
    c = ps[(ps["days_after_surgery"] >= MRD_LO) & (ps["days_after_surgery"] <= MRD_HI)]
    if has_act and not pd.isna(adj_start):
        c = c[c["days_after_surgery"] <= adj_start]
    if c.empty:
        return None
    return c.loc[c["days_after_surgery"].idxmax()]


def _assign_arm(has_act, adj_start_days):
    if not has_act:
        return "no_ACT"
    if pd.isna(adj_start_days):
        return "excluded"
    if adj_start_days < EARLY_HI:
        return "early"
    if adj_start_days < DELAYED_HI:
        return "delayed"
    if adj_start_days <= LATE_HI:
        return "late"
    return "excluded"


def _build():
    patients = pd.read_csv(PATIENTS_FILE, low_memory=False)
    patients["Bundling.ID"] = patients["Bundling.ID"].astype(str)
    samples = pd.read_csv(SAMPLES_FILE, low_memory=False)
    samples["Bundling.ID"] = samples["Bundling.ID"].astype(str)
    samples = samples[samples["days_after_surgery"] >= MIN_ACT_DAYS].copy()
    samp_by_id = {pid: s for pid, s in samples.groupby("Bundling.ID")}

    mrd = []
    for _, p in patients.iterrows():
        ps = samp_by_id.get(p["Bundling.ID"])
        if ps is None:
            continue
        r = _select_tp1(ps, p["adj_start_days"], bool(p["has_act"]))
        if r is None:
            continue
        mrd.append({"Bundling.ID": p["Bundling.ID"], "mrd_day": r["days_after_surgery"],
                    "mrd_mtm": r["MTM"]})
    mrd = pd.DataFrame(mrd)
    evaluable = patients.merge(mrd, on="Bundling.ID", how="inner")
    evaluable["mrd_pos"] = evaluable["mrd_mtm"] > 0
    evaluable["arm"] = [_assign_arm(h, d) for h, d in
                        zip(evaluable["has_act"], evaluable["adj_start_days"])]

    adj = patients[patients["has_act"] == True].copy()
    adj = adj[adj["adj_start_days"].notna() & adj["adj_end_days"].notna() &
              adj["adj_duration"].notna()]
    adj = adj[adj["adj_duration"] >= MIN_ACT_DAYS]
    adj = adj[~(adj["adj_start_days"] > ACT_START_MAX_DAYS)]
    adj["cohort_def"] = np.where(adj["adj_duration"] >= LONG_ACT_MIN_DAYS, "long", "short")

    records = []
    for _, pat in adj.iterrows():
        pid, adj_start, adj_end = pat["Bundling.ID"], pat["adj_start_days"], pat["adj_end_days"]
        ps = samp_by_id.get(pid)
        if ps is None:
            continue
        tp1 = _select_tp1(ps, adj_start, True)
        if tp1 is None:
            continue
        tp1_day = tp1["days_after_surgery"]

        if pat["cohort_def"] == "long":
            midpoint = adj_start + 90
            cand = ps[(ps["days_after_surgery"] >= midpoint - 45) &
                      (ps["days_after_surgery"] <= midpoint + 45) &
                      (ps["days_after_surgery"] >= adj_start + 30) &
                      (ps["days_after_surgery"] <= adj_end - 30) &
                      (ps["days_after_surgery"] > tp1_day)]
            cand = cand[cand.index != tp1.name]
            if cand.empty:
                continue
            dist = (cand["days_after_surgery"] - midpoint).abs()
            closest = cand[dist == dist.min()]
            tp2 = closest.loc[closest["days_after_surgery"].idxmax()]
            tp2_day = tp2["days_after_surgery"]
            if (tp2_day - adj_start) > 600:
                continue
        else:
            cand = ps[ps["days_after_surgery"] > adj_end]
            cand = cand[cand.index != tp1.name]
            if cand.empty:
                continue
            dist = cand["days_after_surgery"] - adj_end
            closest = cand[dist == dist.min()]
            tp2 = closest.loc[closest["days_after_surgery"].idxmin()]
            tp2_day = tp2["days_after_surgery"]
            tp2_rel = tp2_day - adj_start
            if tp2_rel < 14 or tp2_rel > 180:
                continue

        records.append({"Bundling.ID": pid, "cohort_def": pat["cohort_def"],
                        "adj_duration": pat["adj_duration"], "adj_start_days": adj_start,
                        "adj_end_days": adj_end, "tp1_day": tp1_day, "MTM_tp1": tp1["MTM"],
                        "tp2_day": tp2_day, "MTM_tp2": tp2["MTM"],
                        "tp2_days_after_act_start": tp2_day - adj_start,
                        "p_drelDFS_Any": pat["p_drelDFS_Any"], "p_evtDFS": pat["p_evtDFS"]})

    combined = pd.DataFrame(records)
    combined["fu_from_tp2"] = combined["p_drelDFS_Any"] - combined["tp2_day"]
    had_event = combined["p_evtDFS"].astype(bool)
    insufficient = (combined["fu_from_tp2"] < MIN_FU_AFTER_TP2) & ~had_event
    combined = combined[~insufficient].copy()

    m1 = combined["MTM_tp1"].fillna(0)
    m2 = combined["MTM_tp2"].fillna(0)
    conditions = [(m1 == 0) & (m2 == 0), (m1 == 0) & (m2 > 0), (m1 > 0) & (m2 == 0),
                  (m1 > 0) & (m2 > 0) & (m2 < m1), (m1 > 0) & (m2 > m1)]
    choices = ["sustained_negative", "newly_detectable", "clearance",
               "decrease_detectable", "rising"]
    combined["ctdna_dynamics"] = np.select(conditions, choices, default="unassigned")

    demo_merge = adj.drop(columns=["adj_start_days", "adj_end_days", "adj_duration",
                                   "p_drelDFS_Any", "p_evtDFS"], errors="ignore")
    dyn = combined.merge(demo_merge, on="Bundling.ID", how="left", suffixes=("", "_demo"))
    dyn["age"] = pd.to_numeric(dyn["Age"], errors="coerce")
    dyn["ecog_ge1"] = np.where(pd.to_numeric(dyn["ECOG"], errors="coerce") >= 1, 1.0, 0.0)
    dyn.loc[dyn["ECOG"].isna(), "ecog_ge1"] = np.nan

    return evaluable, dyn


def _ensure():
    global _EVALUABLE, _COHORT3
    if _EVALUABLE is None or _COHORT3 is None:
        _EVALUABLE, _COHORT3 = _build()


def load_evaluable_cohort():
    _ensure()
    return _EVALUABLE.copy()


def load_act_timing_cohort():
    _ensure()
    return _EVALUABLE[_EVALUABLE["arm"].isin(ACT_ARMS)].copy()


def load_dynamics_cohort():
    _ensure()
    return _COHORT3.copy()
