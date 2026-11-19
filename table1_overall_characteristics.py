import os
import pandas as pd

from _common import OUT_DIR, load_evaluable_cohort, derive_baseline, build_demographics_table

g = load_evaluable_cohort()
m = derive_baseline(g, "mrd_mtm")
print(f"Evaluable cohort: N = {len(m):,}  "
      f"(MRD\u2212 {int((~m['mrd_pos']).sum()):,}; MRD+ {int(m['mrd_pos'].sum()):,})")

cols = [("All", m), ("MRD\u2212", m[~m["mrd_pos"]]), ("MRD+", m[m["mrd_pos"]])]
groups = [m[~m["mrd_pos"]], m[m["mrd_pos"]]]
table1 = build_demographics_table(cols, groups, include_mrd_row=False)

out = os.path.join(OUT_DIR, "table_1_evaluable_3650_MRDposVsNeg.csv")
table1.to_csv(out, index=False)

pd.set_option("display.max_rows", None, "display.width", 220)
print("=" * 110)
print("TABLE 1 — Patient characteristics, evaluable GALAXY cohort (N = 3,650): MRD\u2212 vs MRD+")
print("=" * 110)
print(table1.to_string(index=False))
print(f"\nSaved -> {out}")
