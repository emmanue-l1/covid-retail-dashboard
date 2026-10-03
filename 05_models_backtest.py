"""
Step 5: Time-series models, rolling backtest, diagnostics, placebo test, counterfactual.
Data  : VOL_SLS, index 2021=100, ALL series with full pre-COVID history.
        SCA  -> non-seasonal models   (Naive12, SES, Holt, DampedHolt, ARIMA, RidgeAR12, Ensemble)
        NSA  -> seasonal models       (SeasonalNaive, HW_noTrend, HW_trend, HW_damped, SARIMA, RidgeAR12, Ensemble)
All models work on log(index). Everything is trained on data up to the forecast origin only.
Run  : python 05_models_backtest.py          (uses all CPU cores; set QUICK=1 for a 6-series test run)
Extra: python 05_models_backtest.py --data results/nonfood_ext/rebuilt_nonfood_CAR.csv --adj CAR --append
       (step 07 does this for you; CAR = non-food index rebuilt from yearly changes, modelled with the seasonal models)
Out  : results/models/*.csv
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from sklearn.linear_model import Ridge
from scipy.stats import wilcoxon

TRAIN_END = pd.Timestamp("2020-02-01")
ORIGINS = pd.date_range("2015-02-01", "2018-02-01", freq="6MS")     # 7 backtest origins, each forecasts 24 months
H = 24
FUTURE = pd.date_range("2020-03-01", "2026-08-01", freq="MS")        # counterfactual horizon (78 months)
OUT = "results/models"
SEASONAL = {"NSA", "CA", "CAR"}          # data types that keep a seasonal pattern (SCA does not)
ARIMA_GRID = [(1, 1, 1), (0, 1, 1), (2, 1, 1), (1, 1, 0), (2, 1, 2), (0, 1, 2)]
SARIMA_GRID = [((1, 1, 1), (0, 1)), ((0, 1, 1), (0, 1)), ((1, 1, 1), (1, 1)),
               ((0, 1, 1), (1, 1)), ((2, 1, 1), (0, 1)), ((1, 1, 0), (0, 1))]   # (order, (P,Q)) with D=1, m=12


# ------------------------------------------------------------------ model building blocks (log scale)
def hw(y, h, trend, damped, seasonal):
    m = ExponentialSmoothing(y, trend=trend, damped_trend=damped if trend else False,
                             seasonal="add" if seasonal else None, seasonal_periods=12 if seasonal else None,
                             initialization_method="estimated").fit()
    return np.asarray(m.forecast(h))


def build_arima(y, spec, seasonal):
    if seasonal:
        order, (P, Q) = spec
        return ARIMA(y, order=order, seasonal_order=(P, 1, Q, 12), trend="n").fit()
    try:
        return ARIMA(y, order=spec, trend="t").fit()                  # ARIMA with drift
    except Exception:
        return ARIMA(y, order=spec, trend="n").fit()


def select_spec(y, seasonal):
    """Pick the order with the lowest AIC on the given training data."""
    best = None
    for spec in (SARIMA_GRID if seasonal else ARIMA_GRID):
        try:
            aic = build_arima(y, spec, seasonal).aic
            if best is None or aic < best[0]:
                best = (aic, spec)
        except Exception:
            pass
    return best[1] if best else (SARIMA_GRID[1] if seasonal else ARIMA_GRID[1])


def f_ridge_ar(y, h, p=12):
    d = np.diff(y)
    X = np.array([d[i - p:i][::-1] for i in range(p, len(d))]); T = d[p:]
    m = Ridge(alpha=1.0).fit(X, T)
    hist, out, lvl = list(d[-p:]), [], y[-1]
    for _ in range(h):
        nd = m.predict(np.array(hist[-p:][::-1]).reshape(1, -1))[0]
        lvl += nd; out.append(lvl); hist.append(nd)
    return np.array(out)


def forecast_all(y, h, seasonal, spec):
    out = {}

    def safe(name, fn):
        try:
            out[name] = np.asarray(fn(), dtype=float)
        except Exception:
            out[name] = np.full(h, np.nan)
    if seasonal:
        safe("SeasonalNaive", lambda: np.tile(y[-12:], h // 12 + 1)[:h])
        safe("HW_noTrend", lambda: hw(y, h, None, False, True))
        safe("HW_trend", lambda: hw(y, h, "add", False, True))
        safe("HW_damped", lambda: hw(y, h, "add", True, True))
        safe("SARIMA", lambda: np.asarray(build_arima(y, spec, True).forecast(h)))
        ens = ["HW_trend", "HW_damped", "SARIMA"]          # Ridge dropped here: weak on seasonal data in the test run
    else:
        safe("Naive12", lambda: np.repeat(y[-12:].mean(), h))
        safe("SES", lambda: hw(y, h, None, False, False))
        safe("Holt", lambda: hw(y, h, "add", False, False))
        safe("DampedHolt", lambda: hw(y, h, "add", True, False))
        safe("ARIMA", lambda: np.asarray(build_arima(y, spec, False).forecast(h)))
        ens = ["DampedHolt", "ARIMA", "RidgeAR12"]
    safe("RidgeAR12", lambda: f_ridge_ar(y, h))
    out["Ensemble"] = np.nanmean([out[k] for k in ens], axis=0)
    return out


# ------------------------------------------------------------------ one series: backtest + final forecast + diagnostics
def run_series(args):
    adj, nace, ctry, s = args
    seasonal = adj in SEASONAL
    ls = np.log(s.dropna())
    bt, fin = [], []
    spec0 = select_spec(ls.loc[:ORIGINS[0]].values, seasonal)          # order chosen BEFORE the first test origin (no leakage)
    for o in ORIGINS:
        tr = ls.loc[:o]; te = ls.loc[o + pd.offsets.MonthBegin(1):].iloc[:H]
        if len(te) < H:
            continue
        for mn, f in forecast_all(tr.values, H, seasonal, spec0).items():
            for k in range(H):
                bt.append((adj, nace, ctry, mn, o, k + 1, np.exp(te.values[k]), np.exp(f[k])))
    pre = ls.loc[:TRAIN_END]
    spec = select_spec(pre.values, seasonal)
    fcs = forecast_all(pre.values, len(FUTURE), seasonal, spec)
    act = s.reindex(FUTURE).values
    lo = hi = None; diag = (adj, nace, ctry, str(spec), np.nan, np.nan, np.nan, np.nan)
    try:
        r = build_arima(pre.values, spec, seasonal)
        ci = np.exp(np.asarray(r.get_forecast(len(FUTURE)).conf_int(alpha=0.05)))
        lo, hi = ci[:, 0], ci[:, 1]
        res = np.asarray(r.resid)[13:]
        diag = (adj, nace, ctry, str(spec), r.aic, r.bic,
                float(acorr_ljungbox(res, lags=[12], return_df=True).lb_pvalue.iloc[0]), float(jarque_bera(res)[1]))
    except Exception:
        pass
    key = "SARIMA" if seasonal else "ARIMA"
    for mn, f in fcs.items():
        for i, m in enumerate(FUTURE):
            fin.append((adj, nace, ctry, mn, m, act[i], np.exp(f[i]),
                        lo[i] if (mn == key and lo is not None) else np.nan,
                        hi[i] if (mn == key and hi is not None) else np.nan))
    return bt, fin, diag


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sts_retail_clean_long.csv",
                    help="long table from step 01 (or the rebuilt non-food file from step 07)")
    ap.add_argument("--adj", nargs="+", default=["SCA", "NSA"], help="data types to model")
    ap.add_argument("--out", default="results/models")
    ap.add_argument("--indicator", default="VOL_SLS", help="VOL_SLS = volume of sales (default), NETTUR = net turnover (value, in current prices)")
    ap.add_argument("--append", action="store_true", help="keep the rows of data types we do not run now")
    args = ap.parse_args()
    OUT, ADJS = args.out, args.adj
    os.makedirs(OUT, exist_ok=True)

    def save(d, name):
        """Write a result table. With --append, rows of the data types NOT run now are kept."""
        path = f"{OUT}/{name}"
        if args.append and os.path.exists(path):
            old = pd.read_csv(path)
            for c in ("month", "origin"):                 # parse dates first, so old and new rows are written the same way
                if c in old.columns:
                    old[c] = pd.to_datetime(old[c])
            d = pd.concat([old[~old.adj.isin(ADJS)], d], ignore_index=True)
        d.to_csv(path, index=False)

    df = pd.read_csv(args.data, parse_dates=["month"])
    df = df[(df.indicator == args.indicator) & (df.unit == "I21")]
    tasks = []
    for adj in ADJS:
        w = df[df.s_adj == adj].pivot_table(index="month", columns=["nace", "country"], values="value").sort_index()
        pre = w.loc[:TRAIN_END]
        for c in [c for c in w.columns if pre[c].notna().all()]:
            tasks.append((adj, c[0], c[1], w[c]))
    if os.environ.get("QUICK"):
        tasks = tasks[:3] + [t for t in tasks if t[0] == "NSA"][:3]
    print(f"{len(tasks)} series to model. Workers: {max(1, (os.cpu_count() or 2) - 1)}")
    BT, FIN, DIAG = [], [], []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        futs = [ex.submit(run_series, t) for t in tasks]
        for i, fu in enumerate(as_completed(futs), 1):
            b, f, d = fu.result(); BT += b; FIN += f; DIAG.append(d)
            if i % 10 == 0 or i == len(futs):
                print(f"  done {i}/{len(futs)}")
    bt = pd.DataFrame(BT, columns=["adj", "nace", "country", "model", "origin", "h", "actual", "forecast"]).dropna()
    fin = pd.DataFrame(FIN, columns=["adj", "nace", "country", "model", "month", "actual", "forecast", "lo95", "hi95"])
    diag = pd.DataFrame(DIAG, columns=["adj", "nace", "country", "chosen_order", "aic", "bic", "ljungbox_p_lag12", "jarque_bera_p"])
    save(bt, "backtest_raw.csv")
    save(fin, "final_forecasts_all_models.csv")
    save(diag, "arima_diagnostics.csv")

    # ---------------- accuracy tables
    bt["err"] = bt.forecast - bt.actual; bt["ape"] = bt.err.abs() / bt.actual * 100
    agg = dict(MAE=("err", lambda x: x.abs().mean()), RMSE=("err", lambda x: np.sqrt((x ** 2).mean())),
               MAPE=("ape", "mean"), MedianAPE=("ape", "median"))
    met = bt.groupby(["adj", "model"]).agg(**agg).round(3).sort_values(["adj", "MAPE"])
    save(met.reset_index(), "accuracy_overall.csv"); print("\nAccuracy (24-month-ahead, rolling origins):\n", met)
    save(bt.groupby(["adj", "nace", "model"]).agg(**agg).round(3).reset_index(), "accuracy_by_category.csv")
    bt["h_bucket"] = pd.cut(bt.h, [0, 6, 12, 18, 24], labels=["1-6", "7-12", "13-18", "19-24"])
    hb = bt.groupby(["adj", "model", "h_bucket"], observed=True).ape.mean().unstack().round(2)
    hb.columns = [str(c) for c in hb.columns]
    save(hb.reset_index(), "accuracy_by_horizon.csv")

    # relative MAE vs the naive benchmark (below 1 = better than naive) and paired significance test
    base = {"SCA": "Naive12", "NSA": "SeasonalNaive", "CA": "SeasonalNaive", "CAR": "SeasonalNaive"}
    pm = bt.groupby(["adj", "nace", "country", "origin", "model"]).err.apply(lambda x: x.abs().mean()).unstack("model")
    rel, tests = [], []
    for adj in ADJS:
        p = pm.loc[adj]; b = p[base[adj]]
        for m in p.columns:
            ok = p[m].notna() & b.notna()
            rel.append((adj, m, float(np.exp(np.log(p[m][ok] / b[ok]).mean()))))
        best = met.loc[adj].index[0]
        for m in p.columns:
            if m != best:
                ok = p[m].notna() & p[best].notna()
                try:
                    tests.append((adj, best, m, wilcoxon(p[best][ok], p[m][ok]).pvalue))
                except Exception:
                    pass
    save(pd.DataFrame(rel, columns=["adj", "model", "relative_MAE_vs_naive"]).round(3), "relative_mae.csv")
    save(pd.DataFrame(tests, columns=["adj", "best_model", "compared_with", "wilcoxon_p"]).round(4), "paired_tests_vs_best.csv")

    # ---------------- placebo test: pretend COVID started in 2018-03 (origin 2018-02, first 22 months)
    pl = bt[(bt.origin == "2018-02-01") & (bt.h <= 22)].assign(gap=lambda x: (x.actual / x.forecast - 1) * 100)
    placebo = pl.groupby(["adj", "nace", "model", "country"]).gap.mean().groupby(["adj", "nace", "model"]).median()
    f2 = fin[(fin.month <= "2021-12-01") & fin.actual.notna()].assign(gap=lambda x: (x.actual / x.forecast - 1) * 100)
    real = f2.groupby(["adj", "nace", "model", "country"]).gap.mean().groupby(["adj", "nace", "model"]).median()
    pr = pd.concat([placebo.rename("placebo_gap_pct_2018_19"), real.rename("real_gap_pct_2020_21")], axis=1).round(2)
    save(pr.reset_index(), "placebo_vs_real_gap.csv"); print("\nPlacebo (should be near 0) vs real COVID gap:\n", pr)
    print("\nDone. Files in", OUT)