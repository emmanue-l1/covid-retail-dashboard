"""
Analysis page (all controls and charts). Opened from the left menu of Story.py:   streamlit run Story.py
Needs, in the same folder:  sts_retail_clean_long.csv  and the results/ folder made by steps 04, 05 and 06.
Install once:  pip install streamlit plotly
"""
import os
import numpy as np, pandas as pd, streamlit as st, plotly.express as px, plotly.graph_objects as go
import ui

st.set_page_config(page_title="EU Retail and COVID-19", page_icon="📊", layout="wide")
ui.setup()
NAMES = {"G47": "Total retail", "G47_FOOD": "Food, drink, tobacco",
         "G47_NFOOD": "Non-food (incl. fuel)", "G4791": "Online / mail order"}
INV = {v: k for k, v in NAMES.items()}
BASIS = {"SCA": "SCA: Eurostat-adjusted data, Ensemble model",
         "NSA": "NSA: unadjusted data, SARIMA model",
         "CAR": "CAR: calendar-adjusted index rebuilt from yearly changes (non-food only), SARIMA model"}
HEAD = {"SCA": "Ensemble", "NSA": "SARIMA", "CAR": "SARIMA"}
M, S, F = "results/models/", "results/stats/", "results/final/"
T = pd.Timestamp
TESTED_END = T("2022-02-01")          # last month inside the 24-month tested range

FILES = ["sts_retail_clean_long.csv",
         S + "stationarity_and_strength_by_series.csv", S + "stationarity_summary.csv",
         S + "seasonal_profile_nsa.csv", S + "manual_adjustment_vs_eurostat_sca.csv",
         M + "accuracy_overall.csv", M + "accuracy_by_category.csv", M + "accuracy_by_horizon.csv",
         M + "relative_mae.csv", M + "paired_tests_vs_best.csv", M + "arima_diagnostics.csv",
         M + "placebo_vs_real_gap.csv", M + "final_forecasts_all_models.csv",
         F + "headline_forecasts.csv", F + "impact_by_series.csv", F + "impact_by_category.csv",
         F + "placebo_by_origin.csv", F + "monthly_median_gap.csv"]
missing = [p for p in FILES if not os.path.exists(p)]
if missing:
    st.error("Some result files are missing: " + ", ".join(missing) +
             ". Run steps 01, 04, 05 and 06 first, and start the dashboard from the project folder.")
    st.stop()


@st.cache_data
def load():
    D = {"raw": pd.read_csv("sts_retail_clean_long.csv", parse_dates=["month"]),
         "hf": pd.read_csv(F + "headline_forecasts.csv", parse_dates=["month"]),
         "imp_s": pd.read_csv(F + "impact_by_series.csv"),
         "imp_c": pd.read_csv(F + "impact_by_category.csv"),
         "pl_o": pd.read_csv(F + "placebo_by_origin.csv", parse_dates=["origin"]),
         "mm": pd.read_csv(F + "monthly_median_gap.csv", parse_dates=["month"]),
         "stat": pd.read_csv(S + "stationarity_and_strength_by_series.csv"),
         "stat_sum": pd.read_csv(S + "stationarity_summary.csv"),
         "prof": pd.read_csv(S + "seasonal_profile_nsa.csv"),
         "cmp": pd.read_csv(S + "manual_adjustment_vs_eurostat_sca.csv"),
         "acc": pd.read_csv(M + "accuracy_overall.csv"),
         "acc_c": pd.read_csv(M + "accuracy_by_category.csv"),
         "acc_h": pd.read_csv(M + "accuracy_by_horizon.csv"),
         "rel": pd.read_csv(M + "relative_mae.csv"),
         "tests": pd.read_csv(M + "paired_tests_vs_best.csv"),
         "diag": pd.read_csv(M + "arima_diagnostics.csv"),
         "plr": pd.read_csv(M + "placebo_vs_real_gap.csv"),
         "fin_all": pd.read_csv(M + "final_forecasts_all_models.csv", parse_dates=["month"])}
    ext = "results/nonfood_ext/rebuilt_nonfood_CAR.csv"
    if os.path.exists(ext):      # rebuilt non-food history from step 07, so the Actual vs forecast chart can draw it
        D["raw"] = pd.concat([D["raw"], pd.read_csv(ext, parse_dates=["month"])], ignore_index=True)
    for v in D.values():
        if "nace" in v.columns:
            v["category"] = v.nace.map(NAMES)
    return D


D = load()
raw, hf, imp_s, imp_c, mm = D["raw"], D["hf"], D["imp_s"], D["imp_c"], D["mm"]


def shade(fig, zero=False):
    """Grey band = the headline COVID window (Mar 2020 to Dec 2021)."""
    fig.add_vrect(x0="2020-03-01", x1="2021-12-01", fillcolor="grey", opacity=0.15, line_width=0)
    if zero:
        fig.add_hline(y=0, line_color="black", line_width=0.8)
    return fig


ui.page_header("Analysis", "Source: Eurostat sts_trtu_m, volume of sales. Counterfactual = a model trained only on data up to Feb 2020 that forecasts Mar 2020 onward. Gap = actual compared with that forecast.", eyebrow="All controls and charts")
# ================================================================ controls bar (top)
BASIS_SHORT = {"SCA": "SCA: Eurostat adjusted, Ensemble", "NSA": "NSA: unadjusted, SARIMA", "CAR": "CAR: rebuilt non-food, SARIMA"}
with st.container(border=True):
    c1, c2, c3, c4, c5 = st.columns([2.1, 2.4, 3.2, 2.6, 1.7], gap="medium")
    cat = c1.selectbox("Retail category", list(NAMES.values()))
    opts = [k for k in BASIS if k in set(hf[hf.category == cat].adj)]       # CAR exists for non-food only
    adj = c2.selectbox("Data basis", opts, index=opts.index("CAR") if "CAR" in opts else 0, format_func=lambda k: BASIS_SHORT[k],
                       help="SCA = Eurostat seasonally adjusted data (Ensemble model). NSA = unadjusted data (SARIMA). "
                            "CAR = non-food index we rebuilt from yearly % changes for more countries (Eurostat gives it for only 6; "
                            "checked against Eurostat's own index, typical error 0.2%).")
    avail = sorted(hf[(hf.category == cat) & (hf.adj == adj)].country.unique())
    default = [c for c in ["Germany", "France", "Italy", "Spain"] if c in avail] or avail[:3]
    ctry = c3.multiselect("Countries to compare", avail, default)
    d0, d1 = hf.month.min().to_pydatetime(), hf.month.max().to_pydatetime()
    dr = c4.slider("Forecast period", d0, d1, (d0, d1), format="MMM YYYY")
    only_tested = c5.checkbox("Tested months only", True,
                              help="Tested = the first 24 months, where the model was proven in the pre-COVID backtest. "
                                   "After Feb 2022 the forecast is only indicative and has no band.")

f = hf[(hf.category == cat) & (hf.adj == adj) & (hf.month >= T(dr[0])) & (hf.month <= T(dr[1]))]
if only_tested:
    f = f[f.in_tested]
s = imp_s[(imp_s.category == cat) & (imp_s.adj == adj) & (imp_s.months_observed >= 22)]   # full 22-month window only
cs_df = imp_c[(imp_c.category == cat) & (imp_c.adj == adj)]
cs = cs_df.iloc[0] if len(cs_df) else None

LABELS = ["Overview", "EDA", "Stationarity and seasonality", "Compare countries", "Actual vs forecast",
                "Models", "Diagnostics and placebo", "Data", "ML challenger", "Value vs volume", "Country drivers"]
with st.sidebar:
    st.markdown('<div style="font-size:12px;font-weight:800;letter-spacing:.14em;color:#64748B;text-transform:uppercase;margin:6px 0 4px">Sections</div>',
                unsafe_allow_html=True)
    page = st.radio("Section", LABELS, label_visibility="collapsed", key="nav")
main_col = st.container()
with main_col:

    # ================================================================ 0. overview
    if page == LABELS[0]:
        st.subheader(f"{cat}: what COVID did to sales volume")
        with st.expander("How to read this page"):
            st.write("The gap is how far real sales were from the no-COVID forecast (in %). "
                     "The noise floor is the biggest fake gap we got when we pretended COVID started at 7 earlier dates "
                     "(2015 to 2018). A real gap counts as a clear effect only if it is bigger than the noise floor.")
        if adj == "CAR":
            st.caption("Not in the results: Ireland (history only from 2016) and Hungary, Poland and the United Kingdom "
                       "(data stops in late 2020, so the 22-month COVID window is incomplete).")
        if cs is None:
            st.warning("No results for this selection.")
        else:
            a, b, c, d, e = st.columns(5)
            a.metric("Countries", int(cs.n_countries))
            b.metric("Median peak gap", f"{cs.median_peak_gap_pct:.1f}%", help="The month with the biggest gap, per country, then the median.")
            c.metric("Median average gap 2020-21", f"{cs.median_avg_gap_pct:.1f}%")
            d.metric("Noise floor (placebo)", f"±{cs.noise_floor_pct:.1f}%")
            e.metric("Median months to recover",
                     "n/a" if pd.isna(cs.median_months_to_recover) else f"{cs.median_months_to_recover:.0f}",
                     help="Months from the peak gap until actual sales are back inside the 95% band for 3 months in a row.")
            msg = (f"The average gap ({cs.median_avg_gap_pct:.1f}%) is bigger than the noise floor "
                   f"(±{cs.noise_floor_pct:.1f}%), so it is a clear effect." if cs.verdict == "clear effect" else
                   f"The average gap ({cs.median_avg_gap_pct:.1f}%) is inside the noise floor "
                   f"(±{cs.noise_floor_pct:.1f}%), so it is not clearly different from normal forecast error.")
            (st.success if cs.verdict == "clear effect" else st.info)(msg + f" {cs.share_countries_same_direction:.0f}% of countries move in the same direction.")

        m = mm[(mm.adj == adj) & (mm.month >= T(dr[0])) & (mm.month <= T(dr[1]))]
        if only_tested:
            m = m[m.month <= TESTED_END]
        l, r = st.columns(2)
        l.plotly_chart(shade(px.line(m, x="month", y="median_gap_pct", color="category",
                                     title="Median gap vs no-COVID forecast (%), all categories"), True),
                       width="stretch")
        mc = m[m.category == cat]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mc.month, y=mc.q75, line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=mc.month, y=mc.q25, fill="tonexty", fillcolor="rgba(31,119,180,0.2)",
                                 line=dict(width=0), name="Middle 50% of countries"))
        fig.add_trace(go.Scatter(x=mc.month, y=mc.median_gap_pct, name="Median", line=dict(color="#1f77b4")))
        r.plotly_chart(shade(fig.update_layout(title=f"{cat}: median gap and spread across countries (%)"), True),
                       width="stretch")

        bar = imp_c[imp_c.adj == adj]
        fig = go.Figure()
        fig.add_bar(x=bar.category, y=bar.median_avg_gap_pct, name="Median average gap 2020-21", marker_color="#1f77b4")
        for sign in (1, -1):
            fig.add_trace(go.Scatter(x=bar.category, y=sign * bar.noise_floor_pct, mode="markers",
                                     name="Noise floor (±)", showlegend=(sign == 1),
                                     marker=dict(symbol="line-ew", size=40, line=dict(width=3, color="red"))))
        st.plotly_chart(fig.update_layout(title="Average gap 2020-21 (%) vs noise floor. The bar must reach past the red marks."),
                        width="stretch")
        st.markdown("**Headline table (both data types)**")
        tbl = imp_c.rename(columns={"adj": "basis"})[["category", "basis", "n_countries", "median_peak_gap_pct",
                                                      "median_avg_gap_pct", "noise_floor_pct", "verdict",
                                                      "median_months_to_recover", "share_countries_same_direction"]]
        st.dataframe(tbl.round(1), width="stretch", hide_index=True)

    # ================================================================ 1. EDA
    if page == LABELS[1]:
        c1, c2, c3 = st.columns(3)
        ind = c1.radio("Indicator", ["VOL_SLS", "NETTUR"], horizontal=True, key="eda_ind")
        ea = c2.radio("Adjustment", ["SCA", "CA", "NSA"], horizontal=True, key="eda_adj")
        yr = c3.slider("From year", 2010, 2024, 2018, key="eda_yr")
        q = raw[(raw.indicator == ind) & (raw.s_adj == ea) & (raw.nace == INV[cat]) & (raw.month.dt.year >= yr)]
        st.plotly_chart(shade(px.line(q[(q.unit == "I21") & q.country.isin(ctry)], x="month", y="value",
                                      color="country", title=f"{cat}: index (2021=100)")), width="stretch")

        left, right = st.columns(2)
        h = raw[(raw.indicator == "VOL_SLS") & (raw.s_adj == "SCA") & (raw.unit == "I21")]
        rows = {}
        for k, v in NAMES.items():
            x = h[h.nace == k].pivot(index="country", columns="month", values="value")
            rows[v] = (x[T("2020-04-01")] / x[T("2020-02-01")] - 1) * 100
        left.plotly_chart(px.imshow(pd.DataFrame(rows), color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                                    aspect="auto", text_auto=".0f", height=700,
                                    title="Heatmap: change Feb to Apr 2020 (%), all countries"), width="stretch")
        rk = s.sort_values("avg_gap_pct_2020_21").assign(selected=lambda x: x.country.isin(ctry))
        right.plotly_chart(px.bar(rk, x="avg_gap_pct_2020_21", y="country", orientation="h", color="selected", height=700,
                                  title="Ranking: average gap vs forecast 2020-21 (%)"), width="stretch")

        left, right = st.columns(2)
        if ctry:
            se = raw[(raw.indicator == ind) & (raw.nace == INV[cat]) & (raw.unit == "I21") & (raw.country == ctry[0])
                     & (raw.month.dt.year >= max(yr, 2017)) & raw.s_adj.isin(["NSA", "SCA"])]
            left.plotly_chart(px.line(se, x="month", y="value", color="s_adj",
                                      title=f"Seasonality: unadjusted (NSA) vs adjusted (SCA), {ctry[0]}"),
                              width="stretch")
        # Eurostat gives the yearly % change (PCH_SM) only for the calendar-adjusted (CA) series
        yy = raw[(raw.indicator == ind) & (raw.nace == INV[cat]) & (raw.unit == "PCH_SM") & (raw.s_adj == "CA")
                 & (raw.month.dt.year >= yr) & (raw.month.dt.year <= 2024)].assign(year=lambda x: x.month.dt.year.astype(str))
        right.plotly_chart(px.box(yy, x="year", y="value", title="Spread of yearly % change across countries (CA data)"),
                           width="stretch")

    # ================================================================ 2. stationarity and seasonality
    if page == LABELS[2]:
        st.subheader("Stationarity and seasonality (pre-COVID data only: 2010-01 to 2020-02)")
        st.caption("A series is called stationary only if the ADF test (p below 0.05) and the KPSS test (p at or above 0.05) "
                   "agree. Tests are run after a seasonal difference where seasonality is strong (strength above 0.64).")
        a2 = st.radio("Adjustment type", ["SCA", "CA", "NSA"], horizontal=True, key="stat_adj")
        stt = D["stat"]
        sa = stt[stt.s_adj == a2]
        l, r = st.columns(2)
        fig = px.scatter(sa, x="adf_p_level", y="kpss_p_level", color="category", hover_name="country",
                         title="ADF p-value (x) vs KPSS p-value (y). Bottom right = not stationary")
        fig.add_vline(x=0.05, line_dash="dash")
        fig.add_hline(y=0.05, line_dash="dash")
        l.plotly_chart(fig, width="stretch")
        fig = px.box(stt, x="s_adj", y="seasonal_strength", color="category", points="all", hover_name="country",
                     title="Seasonal strength (0 = none, 1 = very strong)")
        fig.add_hline(y=0.64, line_dash="dash", annotation_text="0.64: seasonal difference needed")
        r.plotly_chart(fig, width="stretch")
        l, r = st.columns(2)
        l.plotly_chart(px.histogram(stt, x="d_needed", color="s_adj", barmode="group",
                                    title="Number of ordinary differences the tests ask for (d)"), width="stretch")
        r.plotly_chart(px.box(stt, x="s_adj", y="trend_strength", color="category",
                              title="Trend strength (0 to 1)"), width="stretch")
        st.dataframe(D["stat_sum"].round(2), width="stretch", hide_index=True)
        st.caption("d = 2 is likely over-differencing (the tests have low power with only 122 months). The models use d = 1.")

        st.markdown("**Monthly pattern (NSA, pre-COVID)**")
        p = D["prof"][D["prof"].category == cat]
        fig = go.Figure()
        for cn in ctry:
            pc = p[p.country == cn].sort_values("month")
            fig.add_trace(go.Scatter(x=pc.month, y=pc.effect_pct, name=cn, mode="lines+markers", opacity=0.6))
        med = p.groupby("month").effect_pct.median().reset_index()
        fig.add_trace(go.Scatter(x=med.month, y=med.effect_pct, name="Median of all countries",
                                 line=dict(color="black", width=4)))
        st.plotly_chart(fig.update_layout(title=f"{cat}: typical effect of each calendar month (%)",
                                          xaxis_title="Month", yaxis_title="Seasonal effect (%)"), width="stretch")
        if os.path.exists(S + "chart_acf_pacf.png"):
            st.image(S + "chart_acf_pacf.png", caption="ACF and PACF, total retail, SCA, first difference of log")

        st.markdown("**Manual seasonal adjustment vs Eurostat SCA**")
        st.caption("We removed a seasonal pattern learned only from pre-COVID years from the NSA data and compared the monthly "
                   "changes with Eurostat's SCA. A likely reason for the low match before COVID is that normal monthly changes are "
                   "small and noisy, while in 2020-21 the big COVID swings dominate both series.")
        cm = D["cmp"][D["cmp"].category == cat]
        l, r = st.columns(2)
        l.plotly_chart(px.box(cm, x="period", y="corr_monthly_change", points="all", hover_name="country",
                              title="Correlation of monthly changes (manual vs SCA)"), width="stretch")
        r.plotly_chart(px.box(cm, x="period", y="mean_abs_diff_pp", points="all", hover_name="country",
                              title="Average absolute difference (percentage points)"), width="stretch")

    # ================================================================ 3. compare countries
    if page == LABELS[3]:
        if not ctry:
            st.info("Pick at least one country in the bar above.")
        else:
            g = f[f.country.isin(ctry)]
            st.plotly_chart(shade(px.line(g, x="month", y="gap_pct", color="country",
                                          title="Gap vs no-COVID forecast (%)"), True), width="stretch")
            a, b = st.columns(2)
            sel = s[s.country.isin(ctry)]
            a.plotly_chart(px.bar(sel, x="country", y="avg_gap_pct_2020_21", color="country",
                                  title="Average gap 2020-21 (%)"), width="stretch")
            s2 = s.assign(selected=s.country.isin(ctry))
            b.plotly_chart(px.scatter(s2, x="peak_gap_pct", y="months_to_recover", color="selected",
                                      hover_name="country", title="Bigger peak gap = slower recovery? (all countries)"),
                           width="stretch")
            st.dataframe(sel.drop(columns=["nace", "adj"]), width="stretch", hide_index=True)

    # ================================================================ 4. actual vs forecast
    if page == LABELS[4]:
        if not (ctry or avail):
            st.info("No countries available for this selection.")
        else:
            one = st.selectbox("Country", ctry or avail)
            fa = D["fin_all"]
            others = sorted(m for m in fa[fa.adj == adj].model.unique() if m != HEAD[adj])
            extra = st.multiselect("Also show other models", others, [])
            y0 = st.slider("Show history from", 2010, 2024, 2016, key="avf_year")
            hist = raw[(raw.indicator == "VOL_SLS") & (raw.s_adj == adj) & (raw.unit == "I21") & (raw.nace == INV[cat])
                       & (raw.country == one) & (raw.month.dt.year >= y0)]
            gall = hf[(hf.category == cat) & (hf.adj == adj) & (hf.country == one)]
            g = gall[gall.in_tested] if only_tested else gall
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.month, y=hist.value, name="Actual", line=dict(color="black")))
            fig.add_trace(go.Scatter(x=g.month, y=g.hi95, line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=g.month, y=g.lo95, fill="tonexty", fillcolor="rgba(220,50,50,0.18)",
                                     line=dict(width=0), name="95% range (from pre-COVID errors)"))
            fig.add_trace(go.Scatter(x=g.month, y=g.forecast, name=f"No-COVID forecast ({HEAD[adj]})",
                                     line=dict(color="red", dash="dash")))
            for mname in extra:
                o = fa[(fa.adj == adj) & (fa.category == cat) & (fa.country == one) & (fa.model == mname)]
                if only_tested:
                    o = o[o.month <= TESTED_END]
                fig.add_trace(go.Scatter(x=o.month, y=o.forecast, name=mname, line=dict(dash="dot")))
            st.plotly_chart(shade(fig.update_layout(title=f"{one}: {cat} (volume index, {adj})")), width="stretch")
            w = gall[(gall.month <= "2021-12-01") & gall.actual.notna()]
            if len(w):
                a, b, c = st.columns(3)
                a.metric("Months outside the 95% range (2020-21)", f"{w.outside_band.mean() * 100:.0f}%")
                b.metric("Peak gap", f"{w.gap_pct.loc[w.gap_pct.abs().idxmax()]:.1f}%")
                c.metric("Total gap in months of normal sales", f"{w.gap.sum() / w.forecast.mean():.2f}",
                         help="Sum of monthly gaps divided by the average forecast level. -1 means one month of sales lost.")
                st.plotly_chart(px.bar(w, x="month", y="gap_pct", title="Monthly gap vs forecast (%)"),
                                width="stretch")

    # ================================================================ 5. models
    if page == LABELS[5]:
        adjs = sorted(D["acc"].adj.unique())
        ba = st.radio("Compare models for", adjs, index=adjs.index(adj) if adj in adjs else 0, horizontal=True, key="mod_adj")
        st.caption("Rolling-origin backtest on pre-COVID data: 7 start dates (2015-02 to 2018-02), each forecasting 24 months ahead, "
                   "all series. Lower is better.")
        a = D["acc"][D["acc"].adj == ba].sort_values("MAPE")
        l, r = st.columns(2)
        l.plotly_chart(px.bar(a, x="model", y="MAPE", title="MAPE % (lower is better)"), width="stretch")
        r.dataframe(a.drop(columns="adj").round(2), width="stretch", hide_index=True)
        st.caption("A big RMSE next to a normal MAE means a few series went badly wrong. Median APE is the safer number then.")
        l, r = st.columns(2)
        hh = D["acc_h"][D["acc_h"].adj == ba].melt(id_vars=["adj", "model"], var_name="horizon_months", value_name="MAPE")
        l.plotly_chart(px.line(hh, x="horizon_months", y="MAPE", color="model", markers=True,
                               title="Error grows with the forecast horizon (MAPE %)"), width="stretch")
        rl = D["rel"][D["rel"].adj == ba]
        fig = px.bar(rl, x="model", y="relative_MAE_vs_naive", title="MAE relative to the naive model (below 1 = better)")
        fig.add_hline(y=1, line_dash="dash")
        r.plotly_chart(fig, width="stretch")
        hm = D["acc_c"][D["acc_c"].adj == ba].pivot(index="category", columns="model", values="MAPE")
        st.plotly_chart(px.imshow(hm, text_auto=".1f", aspect="auto", color_continuous_scale="RdYlGn_r",
                                  title="MAPE % by category and model"), width="stretch")
        st.markdown("**Is the best model really better? (paired Wilcoxon test over series and origins)**")
        st.dataframe(D["tests"][D["tests"].adj == ba].round(4), width="stretch", hide_index=True)
        st.caption("A p-value below 0.05 means the best model beat that model more often than luck would explain.")

    # ================================================================ 6. diagnostics and placebo
    if page == LABELS[6]:
        da = D["diag"][D["diag"].adj == adj]
        st.subheader(f"Residual checks for the {HEAD[adj]} family ({adj} data)")
        a, b, c = st.columns(3)
        a.metric("Series modelled", len(da))
        b.metric("Leftover pattern (Ljung-Box p below 0.05)", f"{(da.ljungbox_p_lag12 < 0.05).mean() * 100:.0f}%",
                 help="About 5% would fail by pure chance. Lower is better.")
        c.metric("Non-normal residuals (Jarque-Bera p below 0.05)", f"{(da.jarque_bera_p < 0.05).mean() * 100:.0f}%",
                 help="Fat tails make normal-theory bands unreliable, so the dashboard bands use real backtest errors.")
        oc = da.chosen_order.value_counts().rename_axis("order").reset_index(name="series")
        l, r = st.columns(2)
        l.plotly_chart(px.bar(oc, x="order", y="series", title="Model order picked by AIC"), width="stretch")
        r.dataframe(da[da.category == cat].drop(columns=["adj", "nace"]).round(3), width="stretch", hide_index=True)

        st.subheader("Placebo test: pretend COVID started earlier")
        st.caption("If the model is unbiased, fake COVID gaps before 2020 should be about zero. The grey band is the noise floor "
                   "(the largest fake gap). The red line is the real 2020-21 gap.")
        po = D["pl_o"][(D["pl_o"].adj == adj) & (D["pl_o"].category == cat)].copy()
        po["fake_covid_start"] = (po.origin + pd.offsets.MonthBegin(1)).dt.strftime("%Y-%m")
        fig = px.bar(po, x="fake_covid_start", y="fake_gap_median_pct", title="Fake COVID gap by fake start month (median across countries, %)")
        if cs is not None:
            fig.add_hrect(y0=-cs.noise_floor_pct, y1=cs.noise_floor_pct, fillcolor="grey", opacity=0.2, line_width=0)
            fig.add_hline(y=cs.median_avg_gap_pct, line_dash="dash", line_color="red", annotation_text="real gap 2020-21")
        st.plotly_chart(fig, width="stretch")

        pr = D["plr"][(D["plr"].adj == adj) & (D["plr"].category == cat)].copy()
        pr["passes_placebo"] = pr.placebo_gap_pct_2018_19.abs() <= 3
        st.markdown("**Every model: fake gap (2018-19) vs real gap (2020-21), median across countries (%)**")
        st.dataframe(pr.drop(columns=["adj", "nace", "category"]).round(2), width="stretch", hide_index=True)
        st.caption("Passes = fake gap within 3 points before COVID (a threshold we chose). Models that fail, such as naive models "
                   "with no trend, give misleading COVID gaps and are kept only for comparison.")
        hm = D["plr"][D["plr"].adj == adj].pivot(index="category", columns="model", values="real_gap_pct_2020_21")
        st.plotly_chart(px.imshow(hm, text_auto=".1f", aspect="auto", color_continuous_scale="RdYlGn",
                                  color_continuous_midpoint=0, title="Robustness: real gap 2020-21 (%) under every model"),
                        width="stretch")

    # ================================================================ 7. data
    if page == LABELS[7]:
        out = f[f.country.isin(ctry)] if ctry else f
        st.dataframe(out, width="stretch", hide_index=True)
        st.download_button("Download filtered data (CSV)", out.to_csv(index=False), "filtered_forecasts.csv")

    # ================================================================ 8. ML challenger (step 09)
    if page == LABELS[8]:
        MLD = "results/ml/"
        if not all(os.path.exists(MLD + x) for x in ["ml_vs_stat_gap.csv", "ml_accuracy_vs_stat.csv", "ml_country_gaps.csv", "ml_feature_importance.csv"]):
            st.info("Run step 09 (09_ml_global.py) to see the machine-learning model here.")
        else:
            mg, ma = pd.read_csv(MLD + "ml_vs_stat_gap.csv"), pd.read_csv(MLD + "ml_accuracy_vs_stat.csv")
            mc, mf = pd.read_csv(MLD + "ml_country_gaps.csv"), pd.read_csv(MLD + "ml_feature_importance.csv")
            st.write("One gradient-boosting model learns from all countries of a category together. It predicts the monthly growth (not the index), "
                     "so the trend does not limit it, and it uses the same backtest and the same 24-month horizon as the statistical models. "
                     "It is a challenger: the headline results stay with the statistical models.")
            mb = st.radio("Data basis", list(mg.adj.unique()), horizontal=True, key="ml_adj", format_func=lambda k: BASIS[k])
            l, r = st.columns(2)
            a_ = ma[ma.adj == mb].melt(id_vars=["category"], value_vars=["MAPE_GlobalGBM", "MAPE_stat_model"], var_name="model", value_name="MAPE")
            a_["model"] = a_.model.map({"MAPE_GlobalGBM": "Global ML (GBM)", "MAPE_stat_model": f"Statistical ({HEAD[mb]})"})
            l.plotly_chart(px.bar(a_, x="category", y="MAPE", color="model", barmode="group",
                                  title="Backtest error (MAPE %, lower is better)"), width="stretch")
            g_ = mg[mg.adj == mb].melt(id_vars=["category"], value_vars=["ml_median_avg_gap_pct", "stat_median_avg_gap_pct"], var_name="model", value_name="gap")
            g_["model"] = g_.model.map({"ml_median_avg_gap_pct": "Global ML (GBM)", "stat_median_avg_gap_pct": f"Statistical ({HEAD[mb]})"})
            fig = px.bar(g_, x="category", y="gap", color="model", barmode="group", title="Average COVID gap 2020-21 (%), median country")
            fig.add_hline(y=0, line_color="grey", line_width=0.8)
            r.plotly_chart(fig, width="stretch")
            st.dataframe(mg[mg.adj == mb].drop(columns=["adj", "nace"]).rename(columns={
                "ml_countries": "ML countries", "ml_median_avg_gap_pct": "ML gap %", "ml_noise_floor_pct": "ML noise floor %", "ml_verdict": "ML verdict",
                "n_countries": "Stat countries", "stat_median_avg_gap_pct": "Stat gap %", "stat_noise_floor_pct": "Stat noise floor %",
                "stat_verdict": "Stat verdict", "spearman_country_gaps_ml_vs_stat": "Rank agreement across countries (Spearman)"}),
                width="stretch", hide_index=True)
            cats = sorted(mc[mc.adj == mb].nace.map(NAMES).unique())
            cx = st.selectbox("Category for the country comparison", cats, key="ml_cat")
            si = imp_s[(imp_s.adj == mb) & (imp_s.category == cx)][["country", "avg_gap_pct_2020_21"]]
            sc = mc[(mc.adj == mb) & (mc.nace == INV[cx])].merge(si, on="country")
            if len(sc):
                fig = px.scatter(sc, x="avg_gap_pct_2020_21", y="ml_avg_gap_pct", hover_name="country",
                                 labels={"avg_gap_pct_2020_21": "Statistical model gap (%)", "ml_avg_gap_pct": "ML model gap (%)"},
                                 title=f"{cx}: each dot is a country. On the diagonal = both models agree")
                lo_, hi_ = float(min(sc.avg_gap_pct_2020_21.min(), sc.ml_avg_gap_pct.min())), float(max(sc.avg_gap_pct_2020_21.max(), sc.ml_avg_gap_pct.max()))
                fig.add_shape(type="line", x0=lo_, y0=lo_, x1=hi_, y1=hi_, line=dict(dash="dash", color="grey"))
                st.plotly_chart(fig, width="stretch")
            fi = mf[mf.adj == mb].groupby("feature").importance_in_sample.mean().sort_values(ascending=False).head(8).reset_index()
            st.plotly_chart(px.bar(fi, x="importance_in_sample", y="feature", orientation="h",
                                   title="Which inputs the ML model relies on (shuffle test, in-sample, indicative)"), width="stretch")
            st.caption("Growth one year earlier (lag 12) dominates: the model mainly learned the yearly rhythm. This is a check that it is learning something sensible, "
                       "not a causal explanation.")

    # ================================================================ 9. value vs volume (step 10)
    if page == LABELS[9]:
        NT = "results/nettur/"
        if not all(os.path.exists(NT + x) for x in ["value_vs_volume_category.csv", "monthly_value_vs_volume.csv", "value_vs_volume_country.csv"]):
            st.info("Run step 10 (10_net_turnover.py) to compare sales value with sales volume.")
        else:
            vc_, vm_, vy_ = (pd.read_csv(NT + "value_vs_volume_category.csv"), pd.read_csv(NT + "monthly_value_vs_volume.csv", parse_dates=["month"]),
                             pd.read_csv(NT + "value_vs_volume_country.csv"))
            st.write("Volume = how much people bought. Value (net turnover) = how much money shops took, in current prices. "
                     "The same method is applied to both. **Value gap minus volume gap = the price effect.** If both measures point the same way, "
                     "the finding does not depend on the measure.")
            vb = st.radio("Data basis", list(vc_.adj.unique()), horizontal=True, key="vv_adj", format_func=lambda k: BASIS[k].split(",")[0])
            g_ = vc_[vc_.adj == vb].melt(id_vars=["category"], value_vars=["volume_gap_pct", "value_gap_pct"], var_name="measure", value_name="gap")
            g_["measure"] = g_.measure.map({"volume_gap_pct": "Volume of sales", "value_gap_pct": "Net turnover (value)"})
            fig = px.bar(g_, x="category", y="gap", color="measure", barmode="group", title="Average gap 2020-21 (%), median country")
            fig.add_hline(y=0, line_color="grey", line_width=0.8)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(vc_[vc_.adj == vb].drop(columns=["adj", "nace"]), width="stretch", hide_index=True)
            vcx = st.selectbox("Category over time", sorted(vm_.category.unique()), key="vv_cat")
            mq = vm_[(vm_.adj == vb) & (vm_.category == vcx) & (vm_.month <= TESTED_END)]
            mq = mq.melt(id_vars=["month"], value_vars=["median_gap_pct_volume", "median_gap_pct_value", "price_effect_pp"], var_name="line", value_name="pct")
            mq["line"] = mq.line.map({"median_gap_pct_volume": "Volume gap", "median_gap_pct_value": "Value gap", "price_effect_pp": "Price effect (value - volume)"})
            st.plotly_chart(shade(px.line(mq, x="month", y="pct", color="line", title=f"{vcx}: value, volume and the price effect (median country, tested months)"), True),
                            width="stretch")
            st.caption("Before 2021 prices moved little, so the two gaps are close. From the second half of 2021 inflation lifts the value gap above the volume gap.")
            st.markdown("**By country**")
            st.dataframe(vy_[(vy_.adj == vb) & (vy_.category == vcx)][["country", "avg_gap_pct_2020_21_volume", "avg_gap_pct_2020_21_value", "price_effect_pp"]]
                         .sort_values("price_effect_pp"), width="stretch", hide_index=True)

    # ================================================================ 10. country drivers (step 11)
    if page == LABELS[10]:
        DR = "results/drivers/"
        if not all(os.path.exists(DR + f) for f in ["correlations.csv", "regression.csv", "regression_stability.csv", "predictor_correlations.csv", "wave_correlations.csv"]):
            st.info("Run step 11 (11_country_drivers.py) to see what explains the differences between countries.")
        else:
            cr, rg, sb, pcr, wcr = (pd.read_csv(DR + "correlations.csv"), pd.read_csv(DR + "regression.csv"), pd.read_csv(DR + "regression_stability.csv"),
                                    pd.read_csv(DR + "predictor_correlations.csv"), pd.read_csv(DR + "wave_correlations.csv"))
            st.write("Which country facts go together with a bigger or smaller COVID gap? Facts used: excess deaths (health burden), online shopping before COVID, income and, if available, how strict the rules were. "
                     "This shows association, not cause, and only 19 to 33 countries are available per category.")
            hm = cr.pivot(index="category", columns="factor", values="spearman_rho")
            st.plotly_chart(px.imshow(hm, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu", zmin=-1, zmax=1,
                                      title="Rank correlation with the average gap (blue = more of the factor, smaller fall)"), width="stretch")
            st.markdown("**Regression: each factor with the others held fixed** (change in the gap, in percentage points, for +1 standard deviation)")
            st.dataframe(rg.rename(columns={"coef_pp_per_SD": "Change in gap (pp per SD)", "robust_se": "Robust SE", "p_value": "p-value", "r2": "R2", "adj_r2": "Adj. R2"}),
                         width="stretch", hide_index=True)
            st.markdown("**Does the result survive dropping one country at a time?**")
            st.dataframe(sb, width="stretch", hide_index=True)
            st.markdown("**How much do the facts overlap?** (correlation between predictors; high numbers make the regression hard to read)")
            st.dataframe(pcr, width="stretch", hide_index=True)
            st.markdown("**By phase:** gap in the phase against the same factor in the same phase (rank correlation)")
            wf = st.radio("Factor", sorted(wcr.factor.unique()), horizontal=True, key="wave_fac")
            st.plotly_chart(px.imshow(wcr[wcr.factor == wf].pivot(index="category", columns="wave", values="spearman_rho")[["First lockdown", "Summer 2020", "Winter wave", "Reopening 2021"]],
                                      text_auto=".2f", aspect="auto", color_continuous_scale="RdBu", zmin=-1, zmax=1), width="stretch")