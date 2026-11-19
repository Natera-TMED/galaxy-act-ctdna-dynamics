import os
import numpy as np
import pandas as pd

from _common import OUT_DIR, load_dynamics_cohort, derive_baseline, build_demographics_table

d = load_dynamics_cohort()
d["mrd_pos"] = pd.to_numeric(d["MTM_tp1"], errors="coerce") > 0
m = derive_baseline(d, "MTM_tp1")

cols = [
    ("All",       m),
    ("Long ACT",  m[m["cohort_def"] == "long"]),
    ("Short ACT", m[m["cohort_def"] == "short"]),
]
groups = [m[m["cohort_def"] == "long"], m[m["cohort_def"] == "short"]]
supp4 = build_demographics_table(cols, groups, include_mrd_row=True)

out = os.path.join(OUT_DIR, "supplementary_table_4_cohort3_1010_longVsShort.csv")
supp4.to_csv(out, index=False)

pd.set_option("display.max_rows", None, "display.width", 220)
print("=" * 110)
print("SUPPLEMENTARY TABLE 4 — ctDNA-dynamics / ACT-duration cohort (N = 1,010): Long vs Short ACT")
print("=" * 110)
print(supp4.to_string(index=False))
print(f"\nSaved -> {out}")
