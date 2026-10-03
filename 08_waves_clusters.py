"""
Step 8: Waves, country groups (clusters) and map data.   No new models, uses the results of step 06 (and 07).
Run : python 08_waves_clusters.py
Out : results/extra/gap_by_wave_country.csv, gap_by_wave_category.csv, country_clusters.csv, cluster_summary.csv, map_data.csv

Decisions (and why):
 1. Four waves, chosen by what happened, NOT by looking at the gaps:
      First lockdown  Mar-Jun 2020 | Summer 2020  Jul-Oct 2020 | Winter wave  Nov 2020-Apr 2021 | Reopening  May-Dec 2021
 2. Main basis per category = the same as the story page (SCA; non-food = rebuilt CAR when available).
 3. Countries are grouped with Ward hierarchical clustering on the RANKS of the 4 wave gaps (robust to extreme values), into 3 groups.
    3 groups = "hit hard", "middle", "little hit / gained". Simple to explain; with 19-33 countries more groups would be tiny.
    Groups are named by their average gap (lowest / middle / highest), so the names always match the data, also for online where every gap is positive.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster

F, OUT = "results/final/", "results/extra/"
os.makedirs(OUT, exist_ok=True)
NAMES = {"G47": "Total retail", "G47_FOOD": "Food, drink, tobacco", "G47_NFOOD": "Non-food (incl. fuel)", "G4791": "Online / mail order"}
WAVES = [("First lockdown", "2020-03-01", "2020-06-01"), ("Summer 2020", "2020-07-01", "2020-10-01"),
         ("Winter wave", "2020-11-01", "2021-04-01"), ("Reopening 2021", "2021-05-01", "2021-12-01")]
ISO = {"Albania": "ALB", "Austria": "AUT", "Belgium": "BEL", "Bosnia and Herzegovina": "BIH", "Bulgaria": "BGR", "Croatia": "HRV",
       "Cyprus": "CYP", "Czechia": "CZE", "Denmark": "DNK", "Estonia": "EST", "Finland": "FIN", "France": "FRA", "Germany": "DEU",
       "Greece": "GRC", "Hungary": "HUN", "Ireland": "IRL", "Italy": "ITA", "Latvia": "LVA", "Lithuania": "LTU", "Luxembourg": "LUX",
       "Malta": "MLT", "Moldova": "MDA", "Montenegro": "MNE", "Netherlands": "NLD", "North Macedonia": "MKD", "Norway": "NOR",
       "Poland": "POL", "Portugal": "PRT", "Romania": "ROU", "Serbia": "SRB", "Slovakia": "SVK", "Slovenia": "SVN", "Spain": "ESP",
       "Sweden": "SWE", "Switzerland": "CHE", "Türkiye": "TUR", "United Kingdom": "GBR", "Iceland": "ISL", "Liechtenstein": "LIE", "Kosovo": "XKX"}

hf = pd.read_csv(F + "headline_forecasts.csv", parse_dates=["month"])
imp = pd.read_csv(F + "impact_by_series.csv")
has_car = ((hf.adj == "CAR") & (hf.nace == "G47_NFOOD")).any()
PRIMARY = {n: ("CAR" if n == "G47_NFOOD" and has_car else "SCA") for n in NAMES}
hf = hf[hf.adj == hf.nace.map(PRIMARY)]
full = imp[(imp.months_observed >= 22) & (imp.adj == imp.nace.map(PRIMARY))][["nace", "country"]]
hf = hf.merge(full, on=["nace", "country"])               # only countries with the full 22-month window

rows = []
for w, a, b in WAVES:
    x = hf[(hf.month >= a) & (hf.month <= b)].dropna(subset=["gap_pct"])
    rows.append(x.groupby(["nace", "country"]).gap_pct.mean().rename("avg_gap_pct").reset_index().assign(wave=w))
wc = pd.concat(rows)
wc["category"] = wc.nace.map(NAMES)
wc.round(2).to_csv(OUT + "gap_by_wave_country.csv", index=False)
wcat = wc.groupby(["nace", "category", "wave"]).avg_gap_pct.agg(median="median", q25=lambda x: x.quantile(.25),
                                                                q75=lambda x: x.quantile(.75), countries="count").reset_index()
wcat["wave"] = pd.Categorical(wcat.wave, [w[0] for w in WAVES], ordered=True)
wcat.sort_values(["nace", "wave"]).round(2).to_csv(OUT + "gap_by_wave_category.csv", index=False)

# ---- clusters
cl_rows, sum_rows = [], []
for n, g in wc.groupby("nace"):
    M = g.pivot(index="country", columns="wave", values="avg_gap_pct")[[w[0] for w in WAVES]].dropna()
    if len(M) < 9:
        continue
    Z = M.rank(pct=True)                                   # ranks, so one extreme country (e.g. Greece online) cannot dominate
    lab = fcluster(linkage(Z.values, "ward"), 3, "maxclust")
    mean_gap = pd.Series(M.mean(axis=1).values).groupby(lab).mean().sort_values()   # lowest first
    names = dict(zip(mean_gap.index, ["Lowest gap (hit hardest)", "Middle", "Highest gap (least hit / gained)"]))
    if len(mean_gap) < 3:
        names = {k: ["Lowest gap (hit hardest)", "Highest gap (least hit / gained)"][i] for i, k in enumerate(mean_gap.index)}
    for c, l in zip(M.index, lab):
        cl_rows.append((n, NAMES[n], c, names[l], M.loc[c].mean(), *M.loc[c].values))
cl = pd.DataFrame(cl_rows, columns=["nace", "category", "country", "group", "avg_gap_pct"] + [w[0] for w in WAVES]).round(2)
cl.to_csv(OUT + "country_clusters.csv", index=False)
sm = cl.groupby(["nace", "category", "group"]).agg(countries=("country", "count"), avg_gap_pct=("avg_gap_pct", "mean"),
        members=("country", lambda x: ", ".join(sorted(x)))).reset_index().round(2)
sm.to_csv(OUT + "cluster_summary.csv", index=False)

# ---- map data (average gap 2020-21 per country and category)
mp = imp[(imp.months_observed >= 22) & (imp.adj == imp.nace.map(PRIMARY))][["nace", "country", "avg_gap_pct_2020_21"]].copy()
mp["category"] = mp.nace.map(NAMES); mp["iso3"] = mp.country.map(ISO)
mp.to_csv(OUT + "map_data.csv", index=False)
print("Median gap by wave (%):")
print(wcat.pivot(index="category", columns="wave", values="median").round(1)[[w[0] for w in WAVES]].to_string())
print("\nCountry groups:"); print(sm[["category", "group", "countries", "avg_gap_pct", "members"]].to_string(index=False))
print("\nMissing ISO codes:", sorted(set(mp[mp.iso3.isna()].country)))