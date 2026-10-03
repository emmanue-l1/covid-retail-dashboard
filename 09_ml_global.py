"""
Step 9: A real machine-learning model, trained on ALL countries of a category together ("global model").
Run : python 09_ml_global.py                (needs steps 01, 05, 06, and 07 for CAR; only scikit-learn, takes about a minute)
Out : results/ml/*.csv

Why a global model, and why it is different from step 05
 - Step 05 fits one small model per country. Here ONE gradient-boosting model learns from all countries of a category
   at once (about 3,000 training rows per category instead of 120), so it can learn patterns shared across countries.
 - Trees cannot go beyond what they saw, and an index trends. So the model does NOT predict the index. It predicts the
   MONTHLY GROWTH (change of the log index), which has no trend, and we add the predicted growth up to rebuild the level.
 - Features for the next month's growth: the last 12 monthly growths, the 12-month growth, the average growth over the last
   12 months, the month of the year (sin and cos) and the average growth of that country in the training window.
 - Forecasts are made recursively (each forecast is fed back as input), 24 months or more ahead.
 - Same rules as step 05: training data only up to the forecast origin, same 7 origins, same 24-month horizon, same series.
 - The ML model is a CHALLENGER. The headline models stay as they are; this step shows whether ML agrees or does better.
"""
import os, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import wilcoxon, spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--adj", nargs="+", default=["SCA", "NSA", "CAR"])
args = ap.parse_args()
OUT = "results/ml/"; os.makedirs(OUT, exist_ok=True)
TRAIN_END = pd.Timestamp("2020-02-01")
ORIGINS = pd.date_range("2015-02-01", "2018-02-01", freq="6MS")
H, WIN_END = 24, pd.Timestamp("2021-12-01")
FUTURE = pd.date_range("2020-03-01", "2026-08-01", freq="MS")
NAMES = {"G47": "Total retail", "G47_FOOD": "Food, drink, tobacco", "G47_NFOOD": "Non-food (incl. fuel)", "G4791": "Online / mail order"}
HEAD = {"SCA": "Ensemble", "NSA": "SARIMA", "CAR": "SARIMA"}
P = 12


def build_matrix(y, months, with_target=True):
    d = np.diff(y, axis=1)
    n, Td = d.shape
    X, Y = [], []
    gmean = d.mean(axis=1)
    for t in range(P - 1, Td - (1 if with_target else 0)):
        lags = np.stack([d[:, t - k] for k in range(P)], axis=1)          # 12 lags
        mt = months[t + 2]                                                  # month of the value being predicted: d[t+1] = y[t+2] - y[t+1]
        yoy = (y[:, t + 1] - y[:, t + 1 - 12])[:, None] if t + 1 - 12 >= 0 else np.zeros((n, 1))
        f = np.hstack([lags, yoy, lags.mean(axis=1, keepdims=True), gmean[:, None],
                       np.full((n, 1), np.sin(2 * np.pi * mt / 12)), np.full((n, 1), np.cos(2 * np.pi * mt / 12))])
        X.append(f)
        if with_target:
            Y.append(d[:, t + 1])
    return np.vstack(X), (np.concatenate(Y) if with_target else None)


FEAT = [f"growth_lag{k}" for k in range(1, P + 1)] + ["growth_12m", "growth_mean_12m", "country_mean_growth", "month_sin", "month_cos"]


def fit_forecast(Y, dates, steps, seed=0):
    """Y (n_series, T) log index with dates (DatetimeIndex). Returns forecast (n_series, steps) in log and the fitted model."""
    months = np.array([m.month for m in dates])
    X, T = build_matrix(Y, months)
    m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04, max_depth=3, min_samples_leaf=25,
                                      l2_regularization=1.0, random_state=seed).fit(X, T)
    n = Y.shape[0]
    y = Y.copy(); d = np.diff(y, axis=1)
    gmean = d.mean(axis=1)
    out = np.zeros((n, steps))
    cur = list(y.T)                                               # list of arrays per time
    last = dates[-1]
    for s in range(steps):
        tm = (last + pd.offsets.MonthBegin(s + 1)).month
        yy = np.stack(cur, axis=1)
        dd = np.diff(yy, axis=1)
        lags = np.stack([dd[:, -1 - k] for k in range(P)], axis=1)
        yoy = (yy[:, -1] - yy[:, -13])[:, None]
        f = np.hstack([lags, yoy, lags.mean(axis=1, keepdims=True), gmean[:, None],
                       np.full((n, 1), np.sin(2 * np.pi * tm / 12)), np.full((n, 1), np.cos(2 * np.pi * tm / 12))])
        nxt = yy[:, -1] + m.predict(f)
        out[:, s] = nxt; cur.append(nxt)
    return out, m, (X, T)


# ------------------------------------------------------------------ load
df = pd.read_csv("sts_retail_clean_long.csv", parse_dates=["month"])
df = df[(df.indicator == "VOL_SLS") & (df.unit == "I21")]
ext = "results/nonfood_ext/rebuilt_nonfood_CAR.csv"
if "CAR" in args.adj and os.path.exists(ext):
    df = pd.concat([df, pd.read_csv(ext, parse_dates=["month"])], ignore_index=True)
bt_rows, fin_rows, imp_rows = [], [], []
for adj in args.adj:
    w = df[df.s_adj == adj].pivot_table(index="month", columns=["nace", "country"], values="value").sort_index()
    if w.empty:
        continue
    for nace in sorted(set(c[0] for c in w.columns)):
        sub = w[nace]
        pre = sub.loc[:TRAIN_END]
        ctry = [c for c in sub.columns if pre[c].notna().all()]
        if len(ctry) < 5:
            continue
        S = sub[ctry]
        L = np.log(S.loc[:S.index[-1]].reindex(pd.date_range("2010-01-01", S.index.max(), freq="MS")))
        for o in ORIGINS:                                          # ---- backtest
            tr = L.loc[:o]
            te = L.loc[o + pd.offsets.MonthBegin(1):].iloc[:H]
            if len(te) < H:
                continue
            fc, _, _ = fit_forecast(tr.values.T, tr.index, H)
            for i, c in enumerate(ctry):
                for k in range(H):
                    bt_rows.append((adj, nace, c, "GlobalGBM", o, k + 1, np.exp(te.values[k, i]), np.exp(fc[i, k])))
        tr = L.loc[:TRAIN_END]                                      # ---- final counterfactual
        fc, model, (X, T) = fit_forecast(tr.values.T, tr.index, len(FUTURE))
        for i, c in enumerate(ctry):
            act = S[c].reindex(FUTURE).values
            for k, m_ in enumerate(FUTURE):
                fin_rows.append((adj, nace, c, "GlobalGBM", m_, act[k], np.exp(fc[i, k])))
        # simple importance: how much does shuffling each feature hurt the in-sample fit (indicative only)
        base = np.mean((model.predict(X) - T) ** 2); rng = np.random.default_rng(0)
        for j, fn in enumerate(FEAT):
            Xs = X.copy(); Xs[:, j] = rng.permutation(Xs[:, j])
            imp_rows.append((adj, nace, fn, np.mean((model.predict(Xs) - T) ** 2) / base - 1))
        print(f"  {adj} {NAMES[nace]}: {len(ctry)} countries, {len(T)} training rows")

bt = pd.DataFrame(bt_rows, columns=["adj", "nace", "country", "model", "origin", "h", "actual", "forecast"])
fin = pd.DataFrame(fin_rows, columns=["adj", "nace", "country", "model", "month", "actual", "forecast"])
bt.to_csv(OUT + "ml_backtest_raw.csv", index=False); fin.to_csv(OUT + "ml_final_forecasts.csv", index=False)
pd.DataFrame(imp_rows, columns=["adj", "nace", "feature", "importance_in_sample"]).round(3).to_csv(OUT + "ml_feature_importance.csv", index=False)

# ------------------------------------------------------------------ 1. accuracy against the headline statistical model (same series, same origins)
bs = pd.read_csv("results/models/backtest_raw.csv", parse_dates=["origin"])
bs = bs[bs.model == bs.adj.map(HEAD)]
key = ["adj", "nace", "country", "origin", "h"]
both = bt.merge(bs, on=key, suffixes=("_ml", "_st")).dropna()
both["ape_ml"] = (both.forecast_ml - both.actual_ml).abs() / both.actual_ml * 100
both["ape_st"] = (both.forecast_st - both.actual_st).abs() / both.actual_st * 100
acc = []
for (adj, nace), g in both.groupby(["adj", "nace"]):
    pc = g.groupby("country")[["ape_ml", "ape_st"]].mean()
    p = wilcoxon(pc.ape_ml, pc.ape_st).pvalue if len(pc) > 5 else np.nan
    acc.append((adj, nace, NAMES[nace], HEAD[adj], len(pc), g.ape_ml.mean(), g.ape_st.mean(), (pc.ape_ml < pc.ape_st).mean() * 100, p))
acc = pd.DataFrame(acc, columns=["adj", "nace", "category", "stat_model", "countries", "MAPE_GlobalGBM", "MAPE_stat_model",
                                 "pct_countries_ML_better", "wilcoxon_p"]).round(3)
acc.to_csv(OUT + "ml_accuracy_vs_stat.csv", index=False)
tot = both.groupby("adj")[["ape_ml", "ape_st"]].mean().round(3)
tot.to_csv(OUT + "ml_accuracy_overall.csv")
print("\nBacktest error (MAPE %), same series and origins:\n", tot, "\n\nBy category:\n", acc.drop(columns=["nace"]).to_string(index=False))

# ------------------------------------------------------------------ 2. COVID gap with ML, noise floor, verdict
bt["gap_pct"] = (bt.actual / bt.forecast - 1) * 100
pl = bt[bt.h <= 22].groupby(["adj", "nace", "origin", "country"]).gap_pct.mean().groupby(["adj", "nace", "origin"]).median().abs()
floor = pl.groupby(["adj", "nace"]).max().rename("ml_noise_floor_pct").reset_index()
fin["gap_pct"] = (fin.actual / fin.forecast - 1) * 100
win = fin[(fin.month <= WIN_END)].dropna(subset=["gap_pct"])
cnt = win.groupby(["adj", "nace", "country"]).gap_pct.agg(["mean", "count"]).reset_index()
cnt = cnt[cnt["count"] >= 22]
cnt.rename(columns={"mean": "ml_avg_gap_pct"}).drop(columns="count").round(2).to_csv(OUT + "ml_country_gaps.csv", index=False)
mlc = cnt.groupby(["adj", "nace"]).agg(ml_countries=("country", "count"), ml_median_avg_gap_pct=("mean", "median")).reset_index().merge(floor, on=["adj", "nace"])
imp = pd.read_csv("results/final/impact_by_category.csv")[["adj", "nace", "n_countries", "median_avg_gap_pct", "noise_floor_pct", "verdict"]]
imp = imp.rename(columns={"median_avg_gap_pct": "stat_median_avg_gap_pct", "noise_floor_pct": "stat_noise_floor_pct", "verdict": "stat_verdict"})
cmp = mlc.merge(imp, on=["adj", "nace"], how="left")
cmp["ml_verdict"] = np.where(cmp.ml_median_avg_gap_pct.abs() > cmp.ml_noise_floor_pct, "clear effect", "within noise")
cmp["category"] = cmp.nace.map(NAMES)
si = pd.read_csv("results/final/impact_by_series.csv")
cor = []
for (adj, nace), g in cnt.groupby(["adj", "nace"]):
    s_ = si[(si.adj == adj) & (si.nace == nace)].set_index("country").avg_gap_pct_2020_21
    j = g.set_index("country")["mean"].to_frame("ml").join(s_.rename("st"), how="inner")
    cor.append((adj, nace, spearmanr(j.ml, j.st)[0] if len(j) > 4 else np.nan))
cmp = cmp.merge(pd.DataFrame(cor, columns=["adj", "nace", "spearman_country_gaps_ml_vs_stat"]), on=["adj", "nace"])
cmp = cmp[["adj", "category", "nace", "ml_countries", "ml_median_avg_gap_pct", "ml_noise_floor_pct", "ml_verdict", "n_countries",
           "stat_median_avg_gap_pct", "stat_noise_floor_pct", "stat_verdict", "spearman_country_gaps_ml_vs_stat"]].round(2)
cmp.to_csv(OUT + "ml_vs_stat_gap.csv", index=False)
print("\nCOVID gap, ML versus statistical model (median country, Mar 2020 - Dec 2021):\n", cmp.drop(columns=["nace"]).to_string(index=False))
fi = pd.read_csv(OUT + "ml_feature_importance.csv").groupby("feature").importance_in_sample.mean().sort_values(ascending=False).round(2)
print("\nWhich inputs matter most (mean in-sample importance):\n", fi.head(8))
print("\nDone. Files in", OUT)