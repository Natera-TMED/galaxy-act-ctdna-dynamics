from _common import (OUT_DIR, derive_baseline, build_demographics_table,
                     load_act_timing_cohort, simplify_stage)
import os

m = load_act_timing_cohort()
m["stage_num"] = m["p_StagePath"].apply(simplify_stage)
m = derive_baseline(m, "mrd_mtm")

ARM = {"no_ACT": "No ACT", "early": "ACT 0\u20138w", "delayed": "ACT 8\u201312w",
       "late": "ACT 12\u201324w"}

pos = m[m["mrd_pos"]].copy()
cols1 = [
    ("All",            pos),
    ("No ACT",         pos[pos["arm"] == "no_ACT"]),
    ("ACT 0\u20138w",  pos[pos["arm"] == "early"]),
    ("ACT 8\u201312w", pos[pos["arm"] == "delayed"]),
    ("ACT 12\u201324w", pos[pos["arm"] == "late"]),
]
P_OMNIBUS = "p (omnibus: No ACT / 0\u20138w / 8\u201312w / 12\u201324w)"
P_PAIR    = "p (ACT 0\u20138w vs ACT 8\u201312w)"
groups_omni = [pos[pos["arm"] == a] for a in ["no_ACT", "early", "delayed", "late"]]
groups_pair = [pos[pos["arm"] == a] for a in ["early", "delayed"]]

supp1 = build_demographics_table(cols1, groups_omni, include_mrd_row=False).rename(columns={"p": P_OMNIBUS})
supp1[P_PAIR] = build_demographics_table(cols1, groups_pair, include_mrd_row=False)["p"].values

out1 = os.path.join(OUT_DIR, "supplementary_table_1_cohort1_MRDpos_466_by_arm.csv")
supp1.to_csv(out1, index=False)

late = m[m["arm"].isin(["delayed", "late"])].copy()
cols2 = [
    ("All",  late),
    ("MRD-", late[~late["mrd_pos"]]),
    ("MRD+", late[late["mrd_pos"]]),
]
groups2 = [late[~late["mrd_pos"]], late[late["mrd_pos"]]]
supp2 = build_demographics_table(cols2, groups2, include_mrd_row=False)

out2 = os.path.join(OUT_DIR, "supplementary_table_2_cohort1_lateACT_292_MRDnegVsPos.csv")
supp2.to_csv(out2, index=False)

import pandas as pd
pd.set_option("display.max_rows", None, "display.width", 220)
print("=" * 110)
print("SUPPLEMENTARY TABLE 1 — TP#1 MRD-positive ACT-timing cohort (N = 466), by ACT timing arm")
print("=" * 110)
print(supp1.to_string(index=False))
print("\n" + "=" * 110)
print("SUPPLEMENTARY TABLE 2 — Prognostic late-starter cohort (ACT 8\u201324w, N = 292): MRD- vs MRD+")
print("=" * 110)
print(supp2.to_string(index=False))
print(f"\nSaved:\n  {out1}\n  {out2}")
