"""
Step 7: Non-food for (almost) all countries.

Problem : Eurostat publishes the non-food INDEX for only 6 countries, but the yearly % change (calendar adjusted)
          exists for 26 countries with a full pre-COVID history.
Idea    : rebuild a level index from the yearly % changes:
              level(month) = level(same month one year earlier) x (1 + change / 100)
          All 12 months of 2010 are set to 100, then every later month is chained from the same month one year earlier.
          This is exact up to rounding. The script PROVES it on the 91 series where Eurostat also gives the true index.
Then    : the rebuilt series (called CAR) is modelled with the same seasonal models as NSA (SARIMA is the headline),
          and step 06 is run again so the results land in results/final.
Run     : python 07_nonfood_extension.py             (needs steps 01, 04, 05 and 06 done first; takes a few minutes)
          python 07_nonfood_extension.py --no-models  (only build and validate the rebuilt series)
Out     : results/nonfood_ext/*.csv, plus CAR rows added to results/models and results/final
Left out: Ireland (history only from 2016) and Hungary, Poland, United Kingdom (data stops in late 2020, so the
          22-month COVID window is incomplete). They stay in the files but are flagged and not used in category results.
"""
import os, sys, subprocess, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--no-models", action="store_true", help="only build and validate, do not run steps 05 and 06")
args = ap.parse_args()
OUT = "results/nonfood_ext"
os.makedirs(OUT, exist_ok=True)
PRE = pd.date_range("2011-01-01", "2020-02-01", freq="MS")     # yearly changes needed to rebuild up to Feb 2020
pd.set_option("display.width", 200)

df = pd.read_csv("sts_retail_clean_long.csv", parse_dates=["month"])
ca = df[(df.indicator == "VOL_SLS") & (df.s_adj == "CA")]


def rebuild(g):
    """g = yearly % change (index = month). Returns the rebuilt level index; every month of 2010 = 100."""
    g = g.sort_index()
    idx = pd.date_range("2010-01-01", g.index.max(), freq="MS")
    L = pd.Series(np.nan, index=idx)
    L.iloc[:12] = 100.0
    for i in range(12, len(idx)):
        gt = g.get(idx[i], np.nan)
        L.iloc[i] = L.iloc[i - 12] * (1 + gt / 100) if pd.notna(gt) else np.nan
    return L


# ------------------------------------------------------------------ 1. proof: rebuild vs the true Eurostat index
rows = []
for nace in ["G47", "G47_FOOD", "G4791", "G47_NFOOD"]:
    for ctry, gg in ca[(ca.nace == nace) & (ca.unit == "PCH_SM")].groupby("country"):
        true = ca[(ca.nace == nace) & (ca.unit == "I21") & (ca.country == ctry)].set_index("month").value.sort_index()
        base = true.reindex(pd.date_range("2010-01-01", "2010-12-01", freq="MS"))
        if true.empty or base.isna().any():
            continue
        L = rebuild(gg.set_index("month").value)
        # the rebuilt series equals the true one times (100 / true value of the same month in 2010)
        scaled = true * (100.0 / base.values)[true.index.month - 1]
        e = (L / scaled.reindex(L.index) - 1).dropna() * 100
        rows.append((nace, ctry, len(e), e.abs().max(), e.abs().mean()))
val = pd.DataFrame(rows, columns=["nace", "country", "months", "max_abs_error_pct", "mean_abs_error_pct"]).round(3)
val.to_csv(f"{OUT}/validation_rebuild_vs_true.csv", index=False)
print("PROOF: rebuilt index vs the true Eurostat calendar-adjusted index (error in %):")
print(val.groupby("nace").agg(series=("country", "count"), typical_max_error=("max_abs_error_pct", "median"),
                              worst_error=("max_abs_error_pct", "max")).round(2))

# ------------------------------------------------------------------ 2. rebuild non-food for every country with a full pre-COVID history
g_all = ca[(ca.nace == "G47_NFOOD") & (ca.unit == "PCH_SM")]
built, skipped = {}, []
for ctry, g in g_all.groupby("country"):
    s = g.set_index("month").value
    if s.reindex(PRE).notna().all():
        built[ctry] = rebuild(s).dropna()
    else:
        skipped.append(ctry)
long = pd.concat([pd.DataFrame({"country": c, "indicator": "VOL_SLS", "nace": "G47_NFOOD", "s_adj": "CAR",
                                "unit": "I21", "month": L.index, "value": L.round(3).values, "flag": "rebuilt"})
                  for c, L in built.items()], ignore_index=True)
long.to_csv(f"{OUT}/rebuilt_nonfood_CAR.csv", index=False)
last = pd.Series({c: L.index.max() for c, L in built.items()})
partial = sorted(last[last < "2021-12-01"].index)
print(f"\nRebuilt non-food for {len(built)} countries. Full 2020-21 window: {len(built) - len(partial)}. "
      f"Data stops before Dec 2021 (kept, but not used in category results): {partial}. "
      f"No full pre-COVID history: {sorted(skipped)}")

# ------------------------------------------------------------------ 3. is the rebuilt series still seasonal? (decides the model family)
try:
    from statsmodels.tsa.seasonal import STL

    def strength(y):
        r = STL(np.log(np.asarray(y, float)), period=12, robust=True).fit()
        return max(0.0, 1 - np.var(r.resid) / np.var(r.seasonal + r.resid))
    fs_rb = [strength(L.loc[:"2020-02-01"]) for L in built.values()]
    tr = ca[(ca.nace == "G47_NFOOD") & (ca.unit == "I21")]
    fs_tr = [strength(g.set_index("month").value.loc[:"2020-02-01"]) for _, g in tr.groupby("country")]
    print(f"\nSeasonal strength (STL, pre-COVID, median): rebuilt = {np.median(fs_rb):.2f}, "
          f"true calendar-adjusted index (6 countries) = {np.median(fs_tr):.2f}. "
          f"Above 0.64 means seasonal models, so CAR uses the same models as NSA.")
except Exception as e:
    print("\n(Seasonality check skipped:", e, ")")

if args.no_models:
    print("\n--no-models: stopping here. Rebuilt file:", f"{OUT}/rebuilt_nonfood_CAR.csv")
    sys.exit(0)

# ------------------------------------------------------------------ 4. run the models (step 05, appended) and the final numbers (step 06)
py = sys.executable
for cmd in ([py, "05_models_backtest.py", "--data", f"{OUT}/rebuilt_nonfood_CAR.csv", "--adj", "CAR", "--append"],
            [py, "06_final_impact.py"]):
    print("\n>>", " ".join(cmd), flush=True)
    if subprocess.run(cmd).returncode:
        sys.exit("A step failed, see the message above: " + " ".join(cmd))

# ------------------------------------------------------------------ 5. compare with the 6-country result
imp = pd.read_csv("results/final/impact_by_series.csv")
nf = imp[(imp.nace == "G47_NFOOD") & (imp.months_observed >= 22)]
six = set(nf[nf.adj == "SCA"].country)
parts = [("SCA: Eurostat index, 6 countries", nf[nf.adj == "SCA"]),
         ("NSA: Eurostat index, 6 countries", nf[nf.adj == "NSA"]),
         ("CAR: rebuilt, the same 6 countries", nf[(nf.adj == "CAR") & nf.country.isin(six)]),
         ("CAR: rebuilt, all countries with full window", nf[nf.adj == "CAR"])]
cmp = pd.DataFrame([(n, len(x), x.avg_gap_pct_2020_21.median(), x.peak_gap_pct.median(), x.months_to_recover.median())
                    for n, x in parts], columns=["basis", "countries", "median_avg_gap_pct", "median_peak_gap_pct",
                                                 "median_months_to_recover"]).round(1)
cmp.to_csv(f"{OUT}/nonfood_comparison.csv", index=False)
print("\nNON-FOOD, window 2020-03 to 2021-12:\n", cmp.to_string(index=False))
car = nf[nf.adj == "CAR"].sort_values("avg_gap_pct_2020_21")[["country", "avg_gap_pct_2020_21", "peak_gap_pct", "months_to_recover"]]
print("\nBy country (CAR):\n", car.to_string(index=False))
cat = pd.read_csv("results/final/impact_by_category.csv")
print("\nVerdict (CAR):\n", cat[(cat.nace == "G47_NFOOD") & (cat.adj == "CAR")][
    ["n_countries", "median_avg_gap_pct", "noise_floor_pct", "verdict", "share_countries_same_direction"]].to_string(index=False))
print("\nDone. Restart the dashboard to see the new non-food results (choose Non-food, basis CAR).")