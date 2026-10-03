"""
Step 6: Final impact numbers. These are the numbers for the report and the dashboard.
Run  : python 06_final_impact.py        (needs results/models/ from step 05)
       python 06_final_impact.py --models results/models_nettur --out results/final_nettur     (net turnover, see step 10)
Out  : results/final/*.csv

Decisions made here (and why):
 1. Headline model = the best one that passed the placebo test.
      SCA data -> Ensemble   (lowest MAPE, small placebo gap)
      NSA data -> SARIMA     (lowest MAPE, small placebo gap)
 2. Uncertainty band = the real pre-COVID forecast errors (2.5% and 97.5% quantiles) for the same
    category and the same forecast horizon. This is better than the ARIMA band because
    the residuals were not normal (Jarque-Bera test).
 3. Bands are only given for the first 24 months after Feb 2020, because the backtest only proves
    skill up to 24 months.
 4. Noise floor = the biggest fake "COVID gap" we get when we pretend COVID started at each of
    the 7 backtest origins (2015-2018). A real gap is called "clear" only if it is bigger than this.
 5. "Recovered" = actual back inside the band for 3 months in a row.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--models", default="results/models/", help="folder with the step 05 results")
ap.add_argument("--out", default="results/final/", help="folder for the final results")
_a = ap.parse_args()
M, OUT = _a.models.rstrip("/") + "/", _a.out.rstrip("/") + "/"
os.makedirs(OUT, exist_ok=True)
HEAD = {"SCA": "Ensemble", "NSA": "SARIMA", "CA": "SARIMA", "CAR": "SARIMA"}
START, END = pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-01")    # headline COVID window = 22 months
TESTED_H, WIN_H = 24, 22

bt = pd.read_csv(M + "backtest_raw.csv", parse_dates=["origin"])
fin = pd.read_csv(M + "final_forecasts_all_models.csv", parse_dates=["month"])
bt = bt[bt.model == bt.adj.map(HEAD)].copy()
fin = fin[(fin.model == fin.adj.map(HEAD))].dropna(subset=["forecast"]).copy()

# ------------------------------------------------------------------ 1. empirical bands from backtest errors
bt["e"] = np.log(bt.actual / bt.forecast)                  # log(actual / forecast)
rows = []
for (adj, nace), g in bt.groupby(["adj", "nace"]):
    lo_prev, hi_prev = 0.0, 0.0
    for h in range(1, TESTED_H + 1):
        w = g.loc[(g.h >= h - 2) & (g.h <= h + 2), "e"]    # small window around h gives a stable estimate
        lo_prev = min(lo_prev, w.quantile(0.025))
        hi_prev = max(hi_prev, w.quantile(0.975))          # band never gets narrower as h grows
        rows.append((adj, nace, h, lo_prev, hi_prev, len(w)))
bands = pd.DataFrame(rows, columns=["adj", "nace", "h", "q_lo", "q_hi", "n_obs"])
bands.to_csv(OUT + "empirical_bands.csv", index=False)

fin["h"] = (fin.month.dt.year - 2020) * 12 + fin.month.dt.month - 2      # months after Feb 2020
fin = fin.drop(columns=["lo95", "hi95"]).merge(bands, on=["adj", "nace", "h"], how="left")
fin["in_tested"] = fin.h <= TESTED_H
fin["lo95"] = fin.forecast * np.exp(fin.q_lo)
fin["hi95"] = fin.forecast * np.exp(fin.q_hi)
fin["gap"] = fin.actual - fin.forecast
fin["gap_pct"] = (fin.actual / fin.forecast - 1) * 100
known = fin.in_tested & fin.actual.notna() & fin.lo95.notna()
fin["outside_band"] = np.where(known, ((fin.actual < fin.lo95) | (fin.actual > fin.hi95)).astype(float), np.nan)
fin = fin.drop(columns=["q_lo", "q_hi", "n_obs"]).sort_values(["adj", "nace", "country", "month"])
fin.to_csv(OUT + "headline_forecasts.csv", index=False)

# ------------------------------------------------------------------ 2. placebo noise floor (7 fake COVID starts)
bt["gap_pct"] = (bt.actual / bt.forecast - 1) * 100
pl = bt[bt.h <= WIN_H].groupby(["adj", "nace", "origin", "country"]).gap_pct.mean()
pl = pl.groupby(["adj", "nace", "origin"]).agg(["median", "count"]).reset_index()
pl.columns = ["adj", "nace", "origin", "fake_gap_median_pct", "n_countries"]
pl.to_csv(OUT + "placebo_by_origin.csv", index=False)
floor = (pl.assign(a=pl.fake_gap_median_pct.abs()).groupby(["adj", "nace"]).a.max()
         .rename("noise_floor_pct").reset_index())


# ------------------------------------------------------------------ 3. impact per series
def months_to_recover(g, peak):
    """Months from the peak gap until the actual is inside the band for 3 months in a row."""
    gg = g.loc[peak:]
    ob = gg.outside_band.to_numpy()
    for i in range(len(ob) - 2):
        if ob[i] == 0 and ob[i + 1] == 0 and ob[i + 2] == 0:
            return (gg.index[i].year - peak.year) * 12 + gg.index[i].month - peak.month
    return np.nan


rows = []
for (adj, nace, ctry), g in fin.groupby(["adj", "nace", "country"]):
    g = g.set_index("month")
    win = g.loc[START:END].dropna(subset=["actual"])
    if win.empty:
        continue
    peak = win.gap_pct.abs().idxmax()                      # month with the biggest gap (down or up)
    rows.append((adj, nace, ctry, peak.strftime("%Y-%m"), win.gap_pct[peak], win.gap_pct.mean(),
                 win.gap.sum() / win.forecast.mean(), win.outside_band.mean() * 100,
                 months_to_recover(g, peak), len(win)))
imp = pd.DataFrame(rows, columns=["adj", "nace", "country", "peak_month", "peak_gap_pct", "avg_gap_pct_2020_21",
                                  "cum_gap_months_of_sales", "pct_months_outside_band", "months_to_recover",
                                  "months_observed"]).round(2)
imp.to_csv(OUT + "impact_by_series.csv", index=False)

# ------------------------------------------------------------------ 4. impact per category and verdict
# Only countries with the FULL 22-month window count here (a country whose data stops in 2020 would look milder)
cat = imp[imp.months_observed >= WIN_H].groupby(["adj", "nace"]).agg(
    n_countries=("country", "count"),
    median_peak_gap_pct=("peak_gap_pct", "median"),
    median_avg_gap_pct=("avg_gap_pct_2020_21", "median"),
    q25_avg_gap_pct=("avg_gap_pct_2020_21", lambda x: x.quantile(.25)),
    q75_avg_gap_pct=("avg_gap_pct_2020_21", lambda x: x.quantile(.75)),
    median_cum_gap_months_of_sales=("cum_gap_months_of_sales", "median"),
    median_pct_months_outside_band=("pct_months_outside_band", "median"),
    median_months_to_recover=("months_to_recover", "median"),
    share_countries_same_direction=("avg_gap_pct_2020_21", lambda x: (np.sign(x) == np.sign(x.median())).mean() * 100),
).reset_index().merge(floor, on=["adj", "nace"], how="left").round(2)
cat["verdict"] = np.where(cat.median_avg_gap_pct.abs() > cat.noise_floor_pct, "clear effect", "within noise")
cat.to_csv(OUT + "impact_by_category.csv", index=False)

rng = cat.pivot(index="nace", columns="adj", values="median_avg_gap_pct")
rng.columns = [f"{c}_median_avg_gap_pct" for c in rng.columns]
rng["range_low"], rng["range_high"] = rng.min(axis=1), rng.max(axis=1)
rng.round(2).to_csv(OUT + "headline_range_by_category.csv")

# ------------------------------------------------------------------ 5. monthly median gap (for charts)
mm = (fin.dropna(subset=["actual"]).groupby(["adj", "nace", "month"])
      .agg(median_gap_pct=("gap_pct", "median"), q25=("gap_pct", lambda x: x.quantile(.25)),
           q75=("gap_pct", lambda x: x.quantile(.75)), n=("gap_pct", "count")).reset_index().round(2))
mm.to_csv(OUT + "monthly_median_gap.csv", index=False)

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print("Headline impact by category (window 2020-03 to 2021-12):")
print(cat[["adj", "nace", "n_countries", "median_peak_gap_pct", "median_avg_gap_pct", "noise_floor_pct", "verdict",
           "median_months_to_recover", "share_countries_same_direction"]].to_string(index=False))
print("\nRange across the two data types:\n", rng.round(1))
print("\nFiles saved in", OUT)