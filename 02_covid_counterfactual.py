"""
Step 2: COVID counterfactual for EU retail (Eurostat sts_trtu_m).
Idea: train ONLY on data up to 2020-02, forecast Mar-2020 onward ("world without COVID"),
      then Impact = Actual - Forecast.

Main data : VOL_SLS (volume of sales), SCA (seasonally+calendar adjusted), I21 (index 2021=100)
Models    : Naive12, LocalTrend60, DampedHolt, RidgeAR12, Combo (average of 3)
Backtest  : 3 pre-COVID origins (2016-02, 2017-02, 2018-02), each forecasting 24 months ahead
Outputs   : results/*.csv and results/*.png
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ settings
INDICATOR, S_ADJ, UNIT = "VOL_SLS", "SCA", "I21"
TRAIN_END   = pd.Timestamp("2020-02-01")       # last month before COVID
COVID_END   = pd.Timestamp("2021-12-01")       # headline impact window: 2020-03 .. 2021-12
BACKTEST_ORIGINS = ["2016-02-01", "2017-02-01", "2018-02-01"]
BT_H = 24
MIN_PRE = 122                                  # months 2010-01..2020-02 must all exist
OUT = "results"; os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ models (work on log index)
def f_naive12(y, h):
    return np.repeat(y[-12:].mean(), h)

def f_local_trend(y, h, win=60):
    yy = y[-win:]; t = np.arange(len(yy))
    b, a = np.polyfit(t, yy, 1)
    return a + b * (len(yy) - 1 + np.arange(1, h + 1))

def f_damped_holt(y, h):
    def run(p, ret=False):
        al, be, ph = p
        l, b = y[0], np.mean(np.diff(y[:12]))
        sse = 0.0
        for v in y[1:]:
            pred = l + ph * b
            sse += (v - pred) ** 2
            ln = pred + al * (v - pred)
            b = ph * b + be * al * (v - pred)
            l = ln
        return (l, b, ph) if ret else sse
    r = minimize(run, [0.5, 0.1, 0.9], bounds=[(0.01, 0.99), (0.001, 0.5), (0.8, 0.98)], method="L-BFGS-B")
    l, b, ph = run(r.x, ret=True)
    return np.array([l + sum(ph ** i for i in range(1, k + 1)) * b for k in range(1, h + 1)])

def f_ridge_ar(y, h, p=12):
    d = np.diff(y)
    X = np.array([d[i - p:i][::-1] for i in range(p, len(d))]); T = d[p:]
    m = Ridge(alpha=1.0).fit(X, T)
    hist = list(d[-p:]); out = []; lvl = y[-1]
    for _ in range(h):
        nd = m.predict(np.array(hist[-p:][::-1]).reshape(1, -1))[0]
        lvl += nd; out.append(lvl); hist.append(nd)
    return np.array(out)

def f_combo(y, h):
    return (f_damped_holt(y, h) + f_local_trend(y, h) + f_naive12(y, h)) / 3

MODELS = {"Naive12": f_naive12, "LocalTrend60": f_local_trend, "DampedHolt": f_damped_holt,
          "RidgeAR12": f_ridge_ar, "Combo": f_combo}

# ------------------------------------------------------------------ load + pick series
df = pd.read_csv("sts_retail_clean_long.csv", parse_dates=["month"])
df = df[(df.indicator == INDICATOR) & (df.s_adj == S_ADJ) & (df.unit == UNIT)]
wide = df.pivot_table(index="month", columns=["nace", "country"], values="value").sort_index()
pre = wide.loc[:TRAIN_END]
ok = [c for c in wide.columns if pre[c].notna().sum() >= MIN_PRE and pre[c].isna().sum() == 0]
wide = wide[ok]
print(f"Series used: {len(ok)}  (by category: {pd.Series([c[0] for c in ok]).value_counts().to_dict()})")

# ------------------------------------------------------------------ backtest
rows = []
for (nace, ctry) in ok:
    s = np.log(wide[(nace, ctry)].dropna())
    for o in BACKTEST_ORIGINS:
        o = pd.Timestamp(o)
        train, test = s.loc[:o], s.loc[o + pd.offsets.MonthBegin(1):][:BT_H]
        if len(test) < BT_H: continue
        for mn, fn in MODELS.items():
            f = fn(train.values, BT_H)
            for k in range(BT_H):
                rows.append((nace, ctry, mn, o, k + 1, f[k] - test.values[k]))
bt = pd.DataFrame(rows, columns=["nace", "country", "model", "origin", "h", "log_err"])
bt["ape"] = (np.exp(bt.log_err) - 1).abs() * 100
score = bt.groupby(["nace", "model"]).agg(MAPE=("ape", "mean"), MedianAPE=("ape", "median"),
                                          RMSE_log=("log_err", lambda x: np.sqrt((x ** 2).mean()))).reset_index()
score.to_csv(f"{OUT}/backtest_metrics_by_category.csv", index=False)
overall = bt.groupby("model").agg(MAPE=("ape", "mean"), MedianAPE=("ape", "median"),
                                  RMSE_log=("log_err", lambda x: np.sqrt((x ** 2).mean()))).sort_values("MAPE")
overall.to_csv(f"{OUT}/backtest_metrics_overall.csv")
print("\nBacktest (pre-COVID, 24-month ahead), lower is better:\n", overall.round(3))
BEST = overall.index[0]; print("Best model:", BEST)

# residual bias check (should be ~0 if the model has no pre-COVID bias)
bias = bt[bt.model == BEST].groupby("nace").log_err.mean().mul(100).round(2)
bias.to_csv(f"{OUT}/backtest_bias_pct.csv"); print("\nMean pre-COVID bias of best model (%):\n", bias)

# ------------------------------------------------------------------ final counterfactual
future = wide.index[wide.index > TRAIN_END]
sig = bt[bt.model == BEST].groupby(["nace", "country"]).log_err.std()
recs = []
for (nace, ctry) in ok:
    s = np.log(wide[(nace, ctry)].loc[:TRAIN_END].dropna())
    f = MODELS[BEST](s.values, len(future))
    sd = sig[(nace, ctry)] * np.sqrt(np.arange(1, len(future) + 1) / 12.0).clip(min=0.5)
    act = wide.loc[future, (nace, ctry)].values
    for i, m in enumerate(future):
        recs.append((nace, ctry, m, act[i], np.exp(f[i]), np.exp(f[i] - 1.96 * sd[i]), np.exp(f[i] + 1.96 * sd[i])))
fc = pd.DataFrame(recs, columns=["nace", "country", "month", "actual", "forecast", "lo95", "hi95"])
fc["horizon"] = (fc.month.dt.year - 2020) * 12 + fc.month.dt.month - 2     # months after 2020-02
fc["reliable"] = fc.horizon <= BT_H                                      # backtest only proves skill up to 24 months
fc["gap"] = fc.actual - fc.forecast
fc["gap_pct"] = fc.gap / fc.forecast * 100
fc["outside_band"] = (fc.actual < fc.lo95) | (fc.actual > fc.hi95)
fc.to_csv(f"{OUT}/counterfactual_forecasts.csv", index=False)

# ------------------------------------------------------------------ impact summary per series
summ = []
for (nace, ctry), g in fc.dropna(subset=["actual"]).groupby(["nace", "country"]):
    g = g.set_index("month")
    w = g.loc["2020-03-01":COVID_END]
    tr = w.gap_pct.idxmin()
    after = g.loc[tr:]
    rec = np.nan
    ge = (after.gap_pct >= 0).astype(int)
    for i in range(len(ge) - 2):
        if ge.iloc[i:i + 3].sum() == 3: rec = (after.index[i].year - tr.year) * 12 + after.index[i].month - tr.month; break
    summ.append((nace, ctry, tr.strftime("%Y-%m"), w.gap_pct.min(), w.gap_pct.mean(),
                 w.gap.sum(), w.gap.sum() / w.forecast.mean() , w.outside_band.mean() * 100,
                 rec, g.loc["2022-01-01":].gap_pct.mean(), g.loc["2025-01-01":].gap_pct.mean()))
imp = pd.DataFrame(summ, columns=["nace", "country", "trough_month", "trough_gap_pct", "avg_gap_pct_2020_21",
                                  "cum_gap_index_pt_months_2020_21", "cum_gap_in_months_of_normal_sales",
                                  "pct_months_outside_band", "months_trough_to_recovery",
                                  "avg_gap_pct_2022_26", "avg_gap_pct_2025_26"]).round(2)
imp.to_csv(f"{OUT}/impact_summary_by_series.csv", index=False)

by_nace = imp.groupby("nace").agg(n_countries=("country", "count"), median_trough_gap_pct=("trough_gap_pct", "median"),
        median_avg_gap_2020_21=("avg_gap_pct_2020_21", "median"), median_months_to_recover=("months_trough_to_recovery", "median"),
        median_gap_2022_26=("avg_gap_pct_2022_26", "median")).round(1)
by_nace.to_csv(f"{OUT}/impact_summary_by_category.csv"); print("\nImpact by category:\n", by_nace)

month_nace = fc.groupby(["nace", "month"]).gap_pct.median().unstack(0)
month_nace.to_csv(f"{OUT}/median_gap_pct_by_month_category.csv")


# ------------------------------------------------------------------ robustness: does the answer depend on the model?
rob_rows = []
for mn, fn in MODELS.items():
    for (nace, ctry) in ok:
        s_ = np.log(wide[(nace, ctry)].loc[:TRAIN_END].dropna())
        f_ = np.exp(fn(s_.values, len(future)))
        a_ = wide.loc[future, (nace, ctry)].values
        mask = (future >= "2020-03-01") & (future <= COVID_END)
        rob_rows.append((mn, nace, ctry, np.nanmean((a_[mask] / f_[mask] - 1) * 100)))
rob = pd.DataFrame(rob_rows, columns=["model", "nace", "country", "avg_gap_pct_2020_21"])
rob_tab = rob.groupby(["nace", "model"]).avg_gap_pct_2020_21.median().unstack().round(1)
rob_tab.to_csv(f"{OUT}/robustness_median_gap_by_model.csv"); print("\nRobustness - median avg gap 2020-21 (%) by model:\n", rob_tab)

# ------------------------------------------------------------------ charts
names = {"G47": "Total retail", "G47_FOOD": "Food, drink, tobacco", "G47_NFOOD": "Non-food (incl. fuel)", "G4791": "Online / mail order"}
fig, ax = plt.subplots(figsize=(10, 5))
for n in month_nace.columns:
    ax.plot(month_nace.index, month_nace[n], label=names[n])
ax.axhline(0, c="k", lw=.8); ax.axvspan(pd.Timestamp("2020-03-01"), COVID_END, color="grey", alpha=.15)
ax.set_title("Actual vs 'no COVID' forecast: median gap across countries (%)"); ax.set_ylabel("gap %"); ax.legend(); fig.tight_layout()
fig.savefig(f"{OUT}/chart_median_gap_by_category.png", dpi=130); plt.close(fig)

show = [c for c in ["Germany", "France", "Italy", "Spain", "Poland", "Netherlands"] if ("G47", c) in ok]
fig, axs = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
for a, c in zip(axs.ravel(), show):
    a.plot(wide.index, wide[("G47", c)], c="k", lw=1.2, label="Actual")
    g = fc[(fc.nace == "G47") & (fc.country == c)]
    a.plot(g.month, g.forecast, c="tab:red", ls="--", label="No-COVID forecast")
    a.fill_between(g.month, g.lo95, g.hi95, color="tab:red", alpha=.15); a.set_title(c)
axs[0, 0].legend(); fig.suptitle("Total retail volume (index 2021=100): actual vs counterfactual"); fig.tight_layout()
fig.savefig(f"{OUT}/chart_actual_vs_forecast_G47.png", dpi=130); plt.close(fig)
print("\nAll files saved in ./results")