"""
Step 4: Statistical study of the data (Phase 1 = stationarity, Phase 2 = seasonality).
Uses ONLY pre-COVID data (2010-01 .. 2020-02) for all tests, so COVID cannot leak in.
Run:  pip install statsmodels
      python 04_stationarity_seasonality.py
Outputs go to results/stats/
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import STL
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN_END = pd.Timestamp("2020-02-01")
OUT = "results/stats"; os.makedirs(OUT, exist_ok=True)
NAMES = {"G47": "Total retail", "G47_FOOD": "Food, drink, tobacco",
         "G47_NFOOD": "Non-food (incl. fuel)", "G4791": "Online / mail order"}

# ------------------------------------------------------------------ load (VOL_SLS, index 2021=100)
df = pd.read_csv("sts_retail_clean_long.csv", parse_dates=["month"])
df = df[(df.indicator == "VOL_SLS") & (df.unit == "I21")]
wide = {a: df[df.s_adj == a].pivot_table(index="month", columns=["nace", "country"], values="value").sort_index()
        for a in ["NSA", "CA", "SCA"]}


def pre_series(a):
    """Series with all 122 months from 2010-01 to 2020-02."""
    w = wide[a].loc[:TRAIN_END]
    return w[[c for c in w.columns if w[c].notna().all()]]


def adf_p(x): return adfuller(x, autolag="AIC")[1]          # H0: unit root (non-stationary)
def kpss_p(x): return kpss(x, regression="c", nlags="auto")[1]   # H0: stationary


def is_stationary(x):
    a, k = adf_p(x), kpss_p(x)
    return a, k, (a < 0.05 and k >= 0.05)                    # both tests must agree


def strengths(y):
    """STL-based strength of seasonality (Fs) and trend (Ft), between 0 and 1 (Wang et al. 2006)."""
    r = STL(y, period=12, robust=True).fit()
    fs = max(0.0, 1 - np.var(r.resid) / np.var(r.seasonal + r.resid))
    ft = max(0.0, 1 - np.var(r.resid) / np.var(r.trend + r.resid))
    return fs, ft


# ------------------------------------------------------------------ PHASE 1 + 2a: stationarity and strengths
rows = []
for adj in ["SCA", "CA", "NSA"]:
    P = pre_series(adj)
    for (nace, ctry) in P.columns:
        y = np.log(P[(nace, ctry)].values)
        fs, ft = strengths(y)
        D = int(fs > 0.64)                                   # rule of thumb: seasonal differencing if Fs > 0.64
        z = y[12:] - y[:-12] if D else y
        a0, k0, st = is_stationary(z); a1 = k1 = np.nan; d = 0
        if not st:
            d = 1; z1 = np.diff(z); a1, k1, st = is_stationary(z1)
            if not st: d = 2
        rows.append((adj, nace, ctry, len(y), a0, k0, a1, k1, d, D, round(fs, 3), round(ft, 3)))
st_df = pd.DataFrame(rows, columns=["s_adj", "nace", "country", "n_months", "adf_p_level", "kpss_p_level",
                                    "adf_p_diff1", "kpss_p_diff1", "d_needed", "D_needed",
                                    "seasonal_strength", "trend_strength"])
st_df.to_csv(f"{OUT}/stationarity_and_strength_by_series.csv", index=False)
summ = st_df.groupby(["s_adj", "nace"]).agg(series=("country", "count"), mean_d=("d_needed", "mean"),
        share_D1=("D_needed", "mean"), mean_seasonal_strength=("seasonal_strength", "mean"),
        mean_trend_strength=("trend_strength", "mean")).round(2)
summ.to_csv(f"{OUT}/stationarity_summary.csv")
print("Stationarity / strength summary:\n", summ)

# ------------------------------------------------------------------ PHASE 2b: seasonal profile (NSA) + manual adjustment vs Eurostat SCA
prof_rows, cmp_rows = [], []
for (nace, ctry) in pre_series("NSA").columns:
    full = np.log(wide["NSA"][(nace, ctry)].dropna())
    pre = full.loc[:TRAIN_END]
    r = STL(pre.values, period=12, robust=True).fit()
    seas = pd.Series(r.seasonal, index=pre.index).iloc[-60:]            # last 5 pre-COVID years
    prof = seas.groupby(seas.index.month).mean()                        # 12 monthly effects (log points)
    for m, v in prof.items():
        prof_rows.append((nace, ctry, m, v * 100))
    adj_ser = full - prof.reindex(full.index.month).values              # seasonal factors FIXED from pre-COVID
    if (nace, ctry) in wide["SCA"].columns:
        sca = np.log(wide["SCA"][(nace, ctry)].dropna())
        dd = pd.concat([adj_ser.diff(), sca.diff()], axis=1, keys=["manual", "eurostat"]).dropna() * 100
        for name, a, b in [("pre_covid", "2010-01-01", "2020-02-01"), ("covid_2020_21", "2020-03-01", "2021-12-01"),
                           ("after_2022", "2022-01-01", "2026-12-01")]:
            w = dd.loc[a:b]
            if len(w) > 12:
                cmp_rows.append((nace, ctry, name, w.manual.corr(w.eurostat), (w.manual - w.eurostat).abs().mean()))
pd.DataFrame(prof_rows, columns=["nace", "country", "month", "effect_pct"]).to_csv(f"{OUT}/seasonal_profile_nsa.csv", index=False)
cmp_df = pd.DataFrame(cmp_rows, columns=["nace", "country", "period", "corr_monthly_change", "mean_abs_diff_pp"])
cmp_df.to_csv(f"{OUT}/manual_adjustment_vs_eurostat_sca.csv", index=False)
print("\nManual adjustment vs Eurostat SCA (median over countries):\n",
      cmp_df.groupby(["nace", "period"])[["corr_monthly_change", "mean_abs_diff_pp"]].median().round(2))

# ------------------------------------------------------------------ charts
prof = pd.read_csv(f"{OUT}/seasonal_profile_nsa.csv")
fig, ax = plt.subplots(figsize=(10, 5))
for n, g in prof.groupby("nace"):
    ax.plot(range(1, 13), g.groupby("month").effect_pct.median(), marker="o", label=NAMES[n])
ax.axhline(0, c="k", lw=.8); ax.set_xticks(range(1, 13)); ax.set_xlabel("Month"); ax.set_ylabel("Seasonal effect (%)")
ax.set_title("Typical monthly pattern (median across countries, pre-COVID NSA data)"); ax.legend(); fig.tight_layout()
fig.savefig(f"{OUT}/chart_seasonal_profile.png", dpi=130); plt.close(fig)

show = [c for c in ["Germany", "France", "Italy", "Spain", "Poland", "Netherlands"] if ("G47", c) in pre_series("SCA").columns]
fig, axs = plt.subplots(2, len(show), figsize=(3.2 * len(show), 6), sharey=True)
for i, c in enumerate(show):
    x = np.diff(np.log(pre_series("SCA")[("G47", c)].values))
    axs[0, i].bar(range(25), acf(x, nlags=24)); axs[0, i].set_title(f"ACF {c}")
    axs[1, i].bar(range(25), pacf(x, nlags=24)); axs[1, i].set_title(f"PACF {c}")
fig.suptitle("Total retail, SCA, first difference of log (pre-COVID)"); fig.tight_layout()
fig.savefig(f"{OUT}/chart_acf_pacf.png", dpi=130); plt.close(fig)
print("\nDone. Files in", OUT)