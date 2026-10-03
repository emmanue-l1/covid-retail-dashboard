"""
Step 10: Value versus volume (net turnover).
Run : python 10_net_turnover.py             (needs steps 01, 05, 06 done for volume first; takes as long as step 05, a few minutes)
      python 10_net_turnover.py --compare-only   (if the net-turnover models already ran)
Out : results/models_nettur/*, results/final_nettur/*, results/nettur/*.csv

Idea
 - VOLUME of sales (what we used so far) = how much people bought.  NET TURNOVER = how much money shops took, in current prices.
 - Same method for both: model trained to Feb 2020, forecast, gap = actual / forecast - 1.
 - value gap - volume gap = the PRICE effect (percentage points): sales value rose more (or fell less) than volume because prices rose.
 - It is also a robustness test: if the volume story and the value story agree on direction, the finding does not depend on the measure.
Notes
 - Prices barely moved before 2021, so in 2020 the two gaps should be close; in late 2021 inflation starts and value runs ahead.
 - Net turnover has no rebuilt non-food version (CAR), so non-food compares on the 6 countries with a published index (SCA / NSA).
"""
import os, sys, subprocess, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--compare-only", action="store_true")
args = ap.parse_args()
py = sys.executable
NAMES = {"G47": "Total retail", "G47_FOOD": "Food, drink, tobacco", "G47_NFOOD": "Non-food (incl. fuel)", "G4791": "Online / mail order"}
OUT = "results/nettur/"; os.makedirs(OUT, exist_ok=True)

if not args.compare_only:
    for cmd in ([py, "05_models_backtest.py", "--indicator", "NETTUR", "--adj", "SCA", "NSA", "--out", "results/models_nettur"],
                [py, "06_final_impact.py", "--models", "results/models_nettur", "--out", "results/final_nettur"]):
        print("\n>>", " ".join(cmd), flush=True)
        if subprocess.run(cmd).returncode:
            sys.exit("A step failed, see the message above: " + " ".join(cmd))

V, N = "results/final/", "results/final_nettur/"
vs, ns = pd.read_csv(V + "impact_by_series.csv"), pd.read_csv(N + "impact_by_series.csv")
vc, nc = pd.read_csv(V + "impact_by_category.csv"), pd.read_csv(N + "impact_by_category.csv")
vs, ns = vs[(vs.months_observed >= 22) & vs.adj.isin(["SCA", "NSA"])], ns[(ns.months_observed >= 22) & ns.adj.isin(["SCA", "NSA"])]

# ---- country level: same countries in both
cty = vs[["adj", "nace", "country", "avg_gap_pct_2020_21", "peak_gap_pct"]].merge(
    ns[["adj", "nace", "country", "avg_gap_pct_2020_21", "peak_gap_pct"]], on=["adj", "nace", "country"], suffixes=("_volume", "_value"))
cty["price_effect_pp"] = cty.avg_gap_pct_2020_21_value - cty.avg_gap_pct_2020_21_volume
cty["category"] = cty.nace.map(NAMES)
cty.round(2).to_csv(OUT + "value_vs_volume_country.csv", index=False)

# ---- category level (median over the countries present in BOTH)
rows = []
for (adj, nace), g in cty.groupby(["adj", "nace"]):
    a = vc[(vc.adj == adj) & (vc.nace == nace)].iloc[0]; b = nc[(nc.adj == adj) & (nc.nace == nace)].iloc[0]
    rows.append((adj, nace, NAMES[nace], len(g), g.avg_gap_pct_2020_21_volume.median(), g.avg_gap_pct_2020_21_value.median(),
                 g.price_effect_pp.median(), a.noise_floor_pct, b.noise_floor_pct, a.verdict, b.verdict,
                 (np.sign(g.avg_gap_pct_2020_21_volume) == np.sign(g.avg_gap_pct_2020_21_value)).mean() * 100))
cat = pd.DataFrame(rows, columns=["adj", "nace", "category", "countries", "volume_gap_pct", "value_gap_pct", "price_effect_pp",
                                  "volume_noise_floor_pct", "value_noise_floor_pct", "volume_verdict", "value_verdict",
                                  "pct_countries_same_sign"]).round(2)
cat.to_csv(OUT + "value_vs_volume_category.csv", index=False)

# ---- monthly (median country), same countries
mv = pd.read_csv(V + "monthly_median_gap.csv", parse_dates=["month"]); mn = pd.read_csv(N + "monthly_median_gap.csv", parse_dates=["month"])
mm = mv[mv.adj.isin(["SCA", "NSA"])].merge(mn, on=["adj", "nace", "month"], suffixes=("_volume", "_value"))
mm = mm[["adj", "nace", "month", "median_gap_pct_volume", "median_gap_pct_value"]]
mm["price_effect_pp"] = mm.median_gap_pct_value - mm.median_gap_pct_volume
mm["category"] = mm.nace.map(NAMES)
mm.round(2).to_csv(OUT + "monthly_value_vs_volume.csv", index=False)

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print("\nVALUE vs VOLUME, average gap Mar 2020 - Dec 2021 (median country):\n", cat.drop(columns=["nace"]).to_string(index=False))
print("\nPrice effect by month (percentage points, SCA, median country):")
print(mm[mm.adj == "SCA"].pivot(index="month", columns="category", values="price_effect_pp").iloc[::3].round(1).to_string())
print("\nDone. Files in", OUT, "(restart the dashboard to see them).")