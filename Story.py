"""
Main page: the STORY of what happened to European retail during COVID-19.
Run (from the project folder):   streamlit run Story.py
The detailed dashboard is in pages/1_Analysis.py and shows up in the left menu.
Needs: sts_retail_clean_long.csv and the results/ folder made by steps 04, 05, 06 (and 07 for non-food).
"""
import os
import numpy as np, pandas as pd, streamlit as st, plotly.express as px, plotly.graph_objects as go
import ui

st.set_page_config(page_title="The COVID-19 retail story", page_icon="🛍️", layout="wide")
ui.setup()
NAMES = {"G47": "Total retail", "G47_FOOD": "Food, drink, tobacco",
         "G47_NFOOD": "Non-food (incl. fuel)", "G4791": "Online / mail order"}
INV = {v: k for k, v in NAMES.items()}
# one fixed colour per category (colour-blind safe, same colour everywhere on this page)
COL = {"Total retail": "#0072B2", "Food, drink, tobacco": "#E69F00",
       "Non-food (incl. fuel)": "#009E73", "Online / mail order": "#CC79A7"}
M, S, F = "results/models/", "results/stats/", "results/final/"
T = pd.Timestamp
TESTED_END = T("2022-02-01")
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

FILES = ["sts_retail_clean_long.csv", S + "seasonal_profile_nsa.csv", M + "placebo_vs_real_gap.csv",
         F + "headline_forecasts.csv", F + "impact_by_series.csv", F + "impact_by_category.csv",
         F + "placebo_by_origin.csv", F + "monthly_median_gap.csv"]
missing = [p for p in FILES if not os.path.exists(p)]
if missing:
    st.error("Some result files are missing: " + ", ".join(missing) +
             ". Run steps 01, 04, 05 and 06 first, and start the page from the project folder.")
    st.stop()


@st.cache_data
def load():
    D = {"raw": pd.read_csv("sts_retail_clean_long.csv", parse_dates=["month"]),
         "hf": pd.read_csv(F + "headline_forecasts.csv", parse_dates=["month"]),
         "imp_s": pd.read_csv(F + "impact_by_series.csv"),
         "imp_c": pd.read_csv(F + "impact_by_category.csv"),
         "pl_o": pd.read_csv(F + "placebo_by_origin.csv", parse_dates=["origin"]),
         "mm": pd.read_csv(F + "monthly_median_gap.csv", parse_dates=["month"]),
         "prof": pd.read_csv(S + "seasonal_profile_nsa.csv"),
         "plr": pd.read_csv(M + "placebo_vs_real_gap.csv")}
    ext = "results/nonfood_ext/rebuilt_nonfood_CAR.csv"
    if os.path.exists(ext):
        D["raw"] = pd.concat([D["raw"], pd.read_csv(ext, parse_dates=["month"])], ignore_index=True)
    for v in D.values():
        if "nace" in v.columns:
            v["category"] = v.nace.map(NAMES)
    return D


D = load()
raw, hf, imp_s, imp_c, mm, pl_o, prof, plr = (D[k] for k in ["raw", "hf", "imp_s", "imp_c", "mm", "pl_o", "prof", "plr"])

# main basis per category: Eurostat seasonally adjusted data (SCA); for non-food the rebuilt index (CAR) covers many more countries
PRIMARY = {n: ("CAR" if n == "G47_NFOOD" and ((imp_c.adj == "CAR") & (imp_c.nace == n)).any() else "SCA") for n in NAMES}
CHECK = {n: "NSA" for n in NAMES}
BASIS_TXT = {"SCA": "Eurostat seasonally adjusted data, Ensemble model", "NSA": "unadjusted data, SARIMA model",
             "CAR": "calendar-adjusted index rebuilt from yearly changes, SARIMA model"}


def pick(df, mapping):
    """Keep for every category only the rows of its chosen basis."""
    return df[df.adj == df.nace.map(mapping)]


cat_p = pick(imp_c, PRIMARY).set_index("nace")
cat_c = pick(imp_c, CHECK).set_index("nace")
ser_p = pick(imp_s, PRIMARY)
ser_p = ser_p[ser_p.months_observed >= 22]
mm_p = pick(mm, PRIMARY)
plo_p = pick(pl_o, PRIMARY)
order = [n for n in NAMES if n in cat_p.index]


def sgn(x):
    return f"{x:+.1f}%"


def shade(fig, zero=True):
    fig.add_vrect(x0="2020-03-01", x1="2021-12-01", fillcolor="grey", opacity=0.15, line_width=0)
    if zero:
        fig.add_hline(y=0, line_color="grey", line_width=0.8)
    return fig


def style(fig, h=400):
    return fig.update_layout(height=h, legend_title_text="", margin=dict(l=10, r=10, t=64, b=10),
                             hovermode="x unified")


def verdict_text(n):
    r = cat_p.loc[n]
    c = cat_c.loc[n] if n in cat_c.index else None
    base = (f"{NAMES[n]}: sales were on average {sgn(r.median_avg_gap_pct)} versus the no-COVID forecast in "
            f"Mar 2020 to Dec 2021. ")
    if r.verdict == "clear effect":
        base += f"That is bigger than the noise floor (±{r.noise_floor_pct:.1f}%), so it is a clear effect. "
    else:
        base += f"That is inside the noise floor (±{r.noise_floor_pct:.1f}%), so it is not clearly different from normal forecast error. "
    if c is not None:
        base += f"The unadjusted-data check gives {sgn(c.median_avg_gap_pct)}."
    return base


# ================================================================ title + headline numbers
nctry = int(cat_p.n_countries.max())
ui.hero("What COVID-19 did to European retail",
        "We forecast what retail sales would have been without the pandemic, then measured the gap between that forecast and what really happened.",
        chips=[f"<b>{nctry}</b> countries", "<b>4</b> retail categories", "<b>2010 to 2026</b> monthly data", "<b>7</b> backtests, <b>24</b>-month horizon"],
        eyebrow="Eurostat sts_trtu_m  |  volume of retail sales")
ui.nav([(1, "Before COVID"), (2, "The shock"), (3, "Winners and losers"), (4, "Countries"), (5, "The forecast"),
        (6, "How sure"), (7, "Recovery"), (8, "After 2022"), (9, "Take-aways")])
ui.kpi_cards([dict(label=NAMES[n], value=sgn(cat_p.loc[n].median_avg_gap_pct), color=COL[NAMES[n]],
                   sub=f"worst month {sgn(cat_p.loc[n].median_peak_gap_pct)}  |  {int(cat_p.loc[n].n_countries)} countries",
                   badge="Clear effect" if cat_p.loc[n].verdict == "clear effect" else "Within noise",
                   clear=cat_p.loc[n].verdict == "clear effect",
                   help=f"Median over {int(cat_p.loc[n].n_countries)} countries of the average gap, Mar 2020 to Dec 2021. Basis: {BASIS_TXT[PRIMARY[n]]}.")
              for n in order])
st.caption("Big number = average gap between real sales and the no-COVID forecast, Mar 2020 to Dec 2021 (median country). "
           "Small line = the median country's worst single month. 'Clear effect' = bigger than the noise floor. For all controls and extra charts, open **Analysis** in the left menu.")
with st.expander("Read the verdict for each category"):
    for n in order:
        st.write(verdict_text(n))

# ================================================================ 1. before covid
ui.section(1, 'Before COVID', 'Retail has a strong yearly rhythm, so we compare each month with what was expected for that month.')
ui.key('Sales swing through the year, so simple month-to-month or year-to-year comparisons would mislead.')
p = prof[prof.nace.isin(order)].groupby(["category", "month"]).effect_pct.median().reset_index()
p["Month"] = p.month.map(lambda m: MONTHS[int(m) - 1])
fig = px.line(p, x="Month", y="effect_pct", color="category", markers=True, color_discrete_map=COL,
              category_orders={"Month": MONTHS, "category": [NAMES[n] for n in order]},
              labels={"effect_pct": "Seasonal effect (%)"},
              title="Typical monthly pattern before COVID (median across countries)")
fig.add_hline(y=0, line_color="grey", line_width=0.8)
st.plotly_chart(style(fig), width="stretch")
spans = p.groupby("category").effect_pct.agg(lambda x: x.max() - x.min())
big = spans.idxmax()
peak_m = p[p.category == big].sort_values("effect_pct").iloc[-1].Month
st.write(f"Sales are not flat through the year. **{big}** swings the most (about {spans.max():.0f} points from its lowest to its "
         f"highest month, highest in {peak_m}). Because of this, we cannot just compare April 2020 with March 2020. "
         "We compare each month with what a model expected for that month, which already includes the usual pattern.")

# ================================================================ 2. the shock
ui.section(2, 'The shock', 'March 2020 onwards: the lines split by category.')
ui.key('Total retail and non-food fell sharply in spring 2020 while online jumped. The first lockdown was the hardest phase; by summer most categories had recovered.')
m = mm_p[(mm_p.month >= T("2019-01-01")) & (mm_p.month <= TESTED_END)]
fig = px.line(m, x="month", y="median_gap_pct", color="category", color_discrete_map=COL,
              category_orders={"category": [NAMES[n] for n in order]},
              labels={"median_gap_pct": "Gap vs no-COVID forecast (%)", "month": ""},
              title="Real sales compared with the no-COVID forecast (median country, %)")
st.plotly_chart(style(shade(fig)), width="stretch")
worst = ser_p.groupby("category").peak_month.agg(lambda x: x.mode().iloc[0])
lines = [f"**{c}**: most countries hit their biggest gap in {pd.Timestamp(v + '-01'):%B %Y}" for c, v in worst.items()]
st.write("The grey band is Mar 2020 to Dec 2021. Before it, the lines sit near zero, which means the model worked well on normal times. "
         "Then they split. " + "; ".join(lines) + ".")

# ---- extra (step 08): waves, groups and map
EX = "results/extra/"
HAS_EX = all(os.path.exists(EX + f) for f in ["gap_by_wave_category.csv", "country_clusters.csv", "cluster_summary.csv", "map_data.csv"])
if HAS_EX:
    wv = pd.read_csv(EX + "gap_by_wave_category.csv")
    WAVE_ORDER = ["First lockdown", "Summer 2020", "Winter wave", "Reopening 2021"]
    st.markdown("#### Wave by wave")
    fig = px.bar(wv, x="wave", y="median", color="category", barmode="group", color_discrete_map=COL,
                 category_orders={"wave": WAVE_ORDER, "category": [NAMES[n] for n in order]},
                 labels={"median": "Median gap (%)", "wave": ""}, title="Median gap in each phase of the pandemic")
    fig.add_hline(y=0, line_color="grey", line_width=0.8)
    st.plotly_chart(style(fig, 380).update_layout(hovermode="closest"), width="stretch")
    st.caption("Phases: first lockdown Mar-Jun 2020, summer Jul-Oct 2020, winter wave Nov 2020-Apr 2021, reopening May-Dec 2021. "
               "They were fixed from the calendar before looking at the results.")

# ================================================================ 3. winners and losers
ui.section(3, 'Winners and losers', 'Which kinds of shops gained or lost, and is it bigger than normal forecast error?')
ui.key('; '.join(f"{NAMES[n].split(' (')[0]} {sgn(cat_p.loc[n].median_avg_gap_pct)}" + ('' if cat_p.loc[n].verdict == 'clear effect' else ' (not clear)') for n in order) + '.')
w = cat_p.reset_index()
fig = px.bar(w, x="category", y="median_avg_gap_pct", color="category", color_discrete_map=COL,
             category_orders={"category": [NAMES[n] for n in order]},
             labels={"median_avg_gap_pct": "Average gap 2020-21 (%)", "category": ""},
             title="Average gap by category. The black dashes show the noise floor (±)")
fig.add_trace(go.Scatter(x=w.category, y=w.noise_floor_pct, mode="markers", name="Noise floor (+)", showlegend=False,
                         marker=dict(symbol="line-ew", color="black", size=34, line=dict(width=2, color="black"))))
fig.add_trace(go.Scatter(x=w.category, y=-w.noise_floor_pct, mode="markers", name="Noise floor (-)", showlegend=False,
                         marker=dict(symbol="line-ew", color="black", size=34, line=dict(width=2, color="black"))))
fig.add_hline(y=0, line_color="grey", line_width=0.8)
st.plotly_chart(style(fig, 380).update_layout(hovermode="closest", showlegend=False), width="stretch")
st.write("A bar that reaches beyond the black dashes is a real effect. A bar inside them could be ordinary forecast error. "
         "In short, shops selling things people could not buy in person lost the most, while online selling gained.")
tab = w[["category", "n_countries", "median_avg_gap_pct", "noise_floor_pct", "verdict", "share_countries_same_direction"]].rename(
    columns={"category": "Category", "n_countries": "Countries", "median_avg_gap_pct": "Average gap (%)",
             "noise_floor_pct": "Noise floor (±%)", "verdict": "Verdict", "share_countries_same_direction": "% countries same direction"})
st.dataframe(tab, width="stretch", hide_index=True)

# ================================================================ 4. countries
ui.section(4, 'Not every country had the same story', 'The same shock hit countries very differently.')
ui.key('Southern and South-Eastern Europe were hit hardest; Belgium, Denmark, Finland, Norway and Switzerland barely moved. Countries that already shopped online did better (an association, not proof of cause).')
cat4 = st.selectbox("Category", [NAMES[n] for n in order], key="c4")
n4 = INV[cat4]
c4 = ser_p[ser_p.nace == n4].sort_values("avg_gap_pct_2020_21")
fig = px.bar(c4, x="avg_gap_pct_2020_21", y="country", orientation="h", color_discrete_sequence=[COL[cat4]],
             labels={"avg_gap_pct_2020_21": "Average gap 2020-21 (%)", "country": ""},
             title=f"{cat4}: average gap by country")
fig.add_vline(x=0, line_color="grey", line_width=0.8)
st.plotly_chart(fig.update_layout(height=max(350, 22 * len(c4) + 100), margin=dict(l=10, r=10, t=64, b=10)),
                width="stretch")
if len(c4):
    lo, hi = c4.iloc[0], c4.iloc[-1]
    st.write(f"Biggest fall: **{lo.country}** ({sgn(lo.avg_gap_pct_2020_21)}). Biggest rise or smallest fall: **{hi.country}** "
             f"({sgn(hi.avg_gap_pct_2020_21)}). Only countries with the full 22-month window are shown.")

st.markdown('')
if HAS_EX:
    mpd = pd.read_csv(EX + "map_data.csv")
    mp = mpd[(mpd.category == cat4) & mpd.iso3.notna()]
    fig = px.choropleth(mp, locations="iso3", color="avg_gap_pct_2020_21", hover_name="country", scope="europe",
                        color_continuous_scale="RdBu", color_continuous_midpoint=0,
                        labels={"avg_gap_pct_2020_21": "Average gap (%)"}, title=f"{cat4}: average gap by country (map)")
    st.plotly_chart(fig.update_layout(height=480, margin=dict(l=0, r=0, t=64, b=0)), width="stretch")
    grp = pd.read_csv(EX + "cluster_summary.csv")
    grp = grp[grp.category == cat4].sort_values("avg_gap_pct")
    st.markdown("**Countries grouped by how their sales moved through the four phases**")
    st.dataframe(grp[["group", "countries", "avg_gap_pct", "members"]].rename(
        columns={"group": "Group", "countries": "Countries", "avg_gap_pct": "Average gap (%)", "members": "Members"}),
        width="stretch", hide_index=True)
    if len(grp) >= 2:
        st.write(f"The groups are made by a clustering method (Ward) on the gaps in the four phases. In {cat4.lower()}, the group with the lowest gap "
                 f"averaged {grp.iloc[0].avg_gap_pct:+.1f}% and the group with the highest gap {grp.iloc[-1].avg_gap_pct:+.1f}%.")
else:
    st.caption("Run step 08 (08_waves_clusters.py) to add the map, the phases and the country groups.")

# ---- why countries differ (step 11)
DR = "results/drivers/"
st.markdown("#### Why did countries differ?")
if all(os.path.exists(DR + f) for f in ["country_features.csv", "correlations.csv", "regression.csv"]):
    cf, cr, rg = (pd.read_csv(DR + "country_features.csv"), pd.read_csv(DR + "correlations.csv"), pd.read_csv(DR + "regression.csv"))
    FACT = {"Online shopping 2019": "online_shopping_2019_pct", "Income (GDP per capita)": "gdp_per_capita_pps_2019",
            "Excess deaths": "excess_deaths_per_100k_per_day", "Confirmed COVID deaths": "confirmed_deaths_per_100k_per_day",
            "Rules strictness (stringency index)": "stringency_index"}
    UNIT = {"Online shopping 2019": "% of people who bought online in the last 12 months (2019)", "Income (GDP per capita)": "GDP per capita in PPS (EU = 100), 2019",
            "Excess deaths": "average estimated excess deaths per 100,000 per day, Mar 2020 - Dec 2021",
            "Confirmed COVID deaths": "average confirmed COVID-19 deaths per 100,000 per day, Mar 2020 - Dec 2021",
            "Rules strictness (stringency index)": "average stringency index (0-100), Mar 2020 - Dec 2021"}
    FACT = {k: v for k, v in FACT.items() if v in cf.columns}
    fx = st.selectbox("Compare the gap in the category above (section 4) with", list(FACT), key="fx")
    pts = cf[(cf.category == cat4)].dropna(subset=[FACT[fx], "avg_gap_pct_2020_21"])
    if len(pts) >= 8:
        fig = px.scatter(pts, x=FACT[fx], y="avg_gap_pct_2020_21", hover_name="country", color_discrete_sequence=[COL[cat4]],
                         labels={FACT[fx]: UNIT[fx], "avg_gap_pct_2020_21": "Average gap 2020-21 (%)"}, title=f"{cat4}: each dot is a country")
        k_, b_ = np.polyfit(pts[FACT[fx]], pts.avg_gap_pct_2020_21, 1)
        xs = np.array([pts[FACT[fx]].min(), pts[FACT[fx]].max()])
        fig.add_trace(go.Scatter(x=xs, y=k_ * xs + b_, mode="lines", line=dict(color="grey", dash="dash"), name="Trend", showlegend=False))
        fig.add_hline(y=0, line_color="grey", line_width=0.6)
        st.plotly_chart(fig.update_layout(height=420, margin=dict(l=10, r=10, t=64, b=10)), width="stretch")
        rr = cr[(cr.category == cat4) & (cr.factor == fx)]
        if len(rr):
            r0 = rr.iloc[0]
            strength = "strong" if abs(r0.spearman_rho) >= 0.5 else "moderate" if abs(r0.spearman_rho) >= 0.3 else "weak"
            st.write(f"Rank correlation: **{r0.spearman_rho:+.2f}** ({strength}), {int(r0.countries)} countries, p = {r0.p_value:.3f}. "
                     "A positive number means that countries with more of this factor lost less (or gained more).")
    tot = rg[(rg.category == cat4)]
    if len(tot):
        st.caption(f"Holding the other factors fixed ({int(tot.countries.iloc[0])} countries, R-squared {tot.r2.iloc[0]:.2f}): " +
                   "; ".join(f"{r.factor} {r.coef_pp_per_SD:+.1f} points per standard deviation (p = {r.p_value:.2f})" for r in tot.itertuples()) +
                   ". With so few countries and factors that move together, treat these as clues, not proof of cause.")
else:
    st.caption("Run step 11 (11_country_drivers.py, needs the three downloaded files) to see what explains the differences between countries.")

# ================================================================ 5. counterfactual
ui.section(5, 'How we know what would have happened', 'The counterfactual: a model that only knows the world before March 2020.')
ui.key('The model is trained only on data up to February 2020, so it cannot know about COVID. The gap between it and reality is the estimated effect.')
st.write("**Step 1.** Take only data up to February 2020. **Step 2.** Fit the model and forecast March 2020 onwards. "
         "**Step 3.** The difference between real sales and this forecast is the COVID effect. "
         "The shaded ribbon is a 95% range built from real forecast errors in normal years (2015 to 2018).")
a, b = st.columns(2)
cat5 = a.selectbox("Category", [NAMES[n] for n in order], key="c5")
n5 = INV[cat5]
adj5 = PRIMARY[n5]
avail = sorted(hf[(hf.nace == n5) & (hf.adj == adj5)].country.unique())
ctry5 = b.selectbox("Country", avail, index=avail.index("Germany") if "Germany" in avail else 0, key="k5")
f5 = hf[(hf.nace == n5) & (hf.adj == adj5) & (hf.country == ctry5)].sort_values("month")
f5t = f5[f5.in_tested]
h5 = raw[(raw.indicator == "VOL_SLS") & (raw.unit == "I21") & (raw.s_adj == adj5) & (raw.nace == n5) &
         (raw.country == ctry5) & (raw.month >= T("2018-01-01")) & (raw.month <= TESTED_END)].sort_values("month")
fig = go.Figure()
fig.add_trace(go.Scatter(x=f5t.month, y=f5t.hi95, line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=f5t.month, y=f5t.lo95, fill="tonexty", fillcolor="rgba(120,120,120,0.25)", line=dict(width=0),
                         name="95% range of normal error"))
fig.add_trace(go.Scatter(x=f5t.month, y=f5t.forecast, name="Forecast without COVID", line=dict(color="#555555", dash="dash", width=2)))
fig.add_trace(go.Scatter(x=h5.month, y=h5.value, name="Real sales", line=dict(color=COL[cat5], width=2)))
fig.update_layout(title=f"{ctry5}, {cat5}: real sales vs no-COVID forecast (index, 2021 = 100)")
fig.add_vline(x="2020-03-01", line_dash="dot", line_color="grey")
st.plotly_chart(style(fig, 420), width="stretch")
r5 = imp_s[(imp_s.nace == n5) & (imp_s.adj == adj5) & (imp_s.country == ctry5)]
if len(r5):
    r5 = r5.iloc[0]
    rec = "not back inside the range within the tested months" if pd.isna(r5.months_to_recover) else f"back inside the range {r5.months_to_recover:.0f} months after the worst month"
    st.write(f"In {ctry5}, the worst month was {pd.Timestamp(r5.peak_month + '-01'):%B %Y} ({sgn(r5.peak_gap_pct)}). "
             f"On average sales were {sgn(r5.avg_gap_pct_2020_21)} versus the forecast, and they were {rec}. "
             f"Basis: {BASIS_TXT[adj5]}.")
else:
    st.info("No result for this country and category.")

# ================================================================ 6. how sure
ui.section(6, 'How sure are we?', 'Four checks: fake COVID dates, two data versions, a machine-learning model, and value against volume.')
ui.key('We tested ourselves four ways. In each, the direction of the results held: online up, non-food down, total retail slightly down, food not clearly different.')
st.write("A forecast is never perfect, so we tested ourselves. We pretended COVID started at 7 earlier dates (2015 to 2018) and measured the "
         "fake 'gap' the model gives when nothing special happened. That is the **noise floor**. Below, the bars are those fake gaps and the "
         "dashed line is the real COVID gap.")
cat6 = st.selectbox("Category", [NAMES[n] for n in order], key="c6")
n6 = INV[cat6]
pp = plo_p[plo_p.nace == n6].copy()
pp["Fake start"] = pp.origin.dt.strftime("%b %Y")
fig = px.bar(pp, x="Fake start", y="fake_gap_median_pct", color_discrete_sequence=["#999999"],
             labels={"fake_gap_median_pct": "Median fake gap (%)"}, title=f"{cat6}: fake COVID starts (grey) vs the real gap (dashed)")
fig.add_hline(y=float(cat_p.loc[n6].median_avg_gap_pct), line_dash="dash", line_color=COL[cat6], line_width=3)
fig.add_hline(y=0, line_color="grey", line_width=0.8)
st.plotly_chart(style(fig, 360).update_layout(hovermode="closest"), width="stretch")
rng = []
for n in order:
    a_, c_ = cat_p.loc[n].median_avg_gap_pct, (cat_c.loc[n].median_avg_gap_pct if n in cat_c.index else np.nan)
    rng.append({"Category": NAMES[n], f"Main basis (%)": a_, "Main basis": PRIMARY[n], "Unadjusted check (%)": c_,
                "Same sign?": "yes" if pd.notna(c_) and np.sign(a_) == np.sign(c_) else "no"})
st.write("Second test: we ran the whole analysis on two different versions of the data. If they agree, we trust the result more.")
st.dataframe(pd.DataFrame(rng), width="stretch", hide_index=True)

st.markdown("#### A machine-learning model as a third opinion")
MLF = "results/ml/ml_vs_stat_gap.csv"
if os.path.exists(MLF):
    mlg = pd.read_csv(MLF)
    mlg = mlg[mlg.adj == mlg.nace.map(PRIMARY)]
    rows_ = []
    for _, r_ in mlg.iterrows():
        rows_.append({"category": r_.category, "model": "Statistical model", "gap": r_.stat_median_avg_gap_pct})
        rows_.append({"category": r_.category, "model": "Machine learning (all countries together)", "gap": r_.ml_median_avg_gap_pct})
    fig = px.bar(pd.DataFrame(rows_), x="category", y="gap", color="model", barmode="group",
                 color_discrete_sequence=["#5B6770", "#D55E00"], labels={"gap": "Average gap 2020-21 (%)", "category": ""},
                 title="Two different kinds of model, the same story")
    fig.add_hline(y=0, line_color="grey", line_width=0.8)
    st.plotly_chart(style(fig, 360).update_layout(hovermode="closest"), width="stretch")
    agree = mlg.spearman_country_gaps_ml_vs_stat.median()
    st.write(f"A gradient-boosting model trained on all countries of a category together gives almost the same size of effect, and the same "
             f"ranking of countries (rank agreement about {agree:.2f} on a scale where 1 is identical). When two quite different methods agree, "
             "the result is less likely to be an accident of one method.")
else:
    st.caption("Run step 09 (09_ml_global.py) to add the machine-learning comparison.")

st.markdown("#### Value or volume?")
NTF = "results/nettur/value_vs_volume_category.csv"
if os.path.exists(NTF):
    nt = pd.read_csv(NTF)
    nt = nt[nt.adj == "SCA"]
    rows_ = []
    for _, r_ in nt.iterrows():
        rows_.append({"category": r_.category, "measure": "Volume of sales (how much was bought)", "gap": r_.volume_gap_pct})
        rows_.append({"category": r_.category, "measure": "Net turnover (money taken in)", "gap": r_.value_gap_pct})
    fig = px.bar(pd.DataFrame(rows_), x="category", y="gap", color="measure", barmode="group",
                 color_discrete_sequence=["#0072B2", "#999999"], labels={"gap": "Average gap 2020-21 (%)", "category": ""},
                 title="The same method on sales volume and on sales value (seasonally adjusted data)")
    fig.add_hline(y=0, line_color="grey", line_width=0.8)
    st.plotly_chart(style(fig, 360).update_layout(hovermode="closest"), width="stretch")
    st.write("Net turnover is measured in current prices, so it also contains price changes. Value minus volume is therefore the price effect. "
             "In 2020 the two are close because prices barely moved; from the second half of 2021 inflation pushes value above volume. "
             "Volume is the better measure of how much people bought, which is why it is the main one here. Details are on the Analysis page.")
else:
    st.caption("Run step 10 (10_net_turnover.py) to compare sales value with sales volume.")

# ================================================================ 7. recovery
ui.section(7, 'Did retail recover?', 'How long the drop lasted and how much was lost in total.')
ui.key('The drop was sharp but short: the median country was back inside its normal range within a few months.')
rc = cat_p.reset_index()
l, r = st.columns(2)
fig = px.bar(rc, x="category", y="median_months_to_recover", color="category", color_discrete_map=COL,
             category_orders={"category": [NAMES[n] for n in order]},
             labels={"median_months_to_recover": "Months", "category": ""},
             title="Months from the worst month until back in the normal range")
l.plotly_chart(style(fig, 360).update_layout(showlegend=False, hovermode="closest"), width="stretch")
fig = px.bar(rc, x="category", y="median_cum_gap_months_of_sales", color="category", color_discrete_map=COL,
             category_orders={"category": [NAMES[n] for n in order]},
             labels={"median_cum_gap_months_of_sales": "Months of normal sales", "category": ""},
             title="Total sales lost (-) or gained (+), in months of normal sales")
fig.add_hline(y=0, line_color="grey", line_width=0.8)
r.plotly_chart(style(fig, 360).update_layout(showlegend=False, hovermode="closest"), width="stretch")
st.write("The drop was sharp but short for most categories: the median country was back inside the normal range within a few months of its "
         "worst month. 'Total sales lost' adds up all gaps from Mar 2020 to Dec 2021 and expresses them as months of normal sales. "
         "Be careful with online: its band is wide, so 'recovery' there is a weak measure.")

# ================================================================ 8. after 2022
ui.section(8, 'After February 2022', 'Beyond the period we tested.')
ui.key('After early 2022 other shocks (energy prices, inflation) mix in, so gaps there are only indicative.')
a22 = pick(mm, PRIMARY)
a22 = a22[(a22.month >= T("2021-01-01"))]
fig = px.line(a22, x="month", y="median_gap_pct", color="category", color_discrete_map=COL,
              category_orders={"category": [NAMES[n] for n in order]},
              labels={"median_gap_pct": "Gap vs no-COVID forecast (%)", "month": ""},
              title="Median gap, whole period (the right side is only indicative)")
fig.add_vrect(x0="2022-03-01", x1=a22.month.max(), fillcolor="orange", opacity=0.12, line_width=0)
st.plotly_chart(style(shade(fig)), width="stretch")
st.warning("The orange area is more than 24 months after the forecast start. We only proved the models up to 24 months, and from 2022 "
           "there were other big shocks (energy prices, high inflation). So gaps there should not be read as the COVID effect.")

# ================================================================ 9. takeaways
ui.section(9, 'Take-aways and limits', '')
t1, t2 = st.columns(2)
with t1:
    st.markdown("**What we found**")
    for n in order:
        r = cat_p.loc[n]
        st.markdown(f"- **{NAMES[n]}**: {sgn(r.median_avg_gap_pct)} on average, worst month {sgn(r.median_peak_gap_pct)} "
                    f"({'clear effect' if r.verdict == 'clear effect' else 'within noise'}).")
with t2:
    st.markdown("**What to keep in mind**")
    st.markdown("- The no-COVID world is a model estimate, not a fact.\n"
                "- Gaps smaller than the noise floor are not proven.\n"
                "- Non-food uses an index we rebuilt from yearly % changes (checked against Eurostat's own index).\n"
                "- Online data has fewer countries and wide bands.\n"
                "- Results are medians across countries, so single countries can differ a lot (see section 4).")
with st.expander("Method in four lines"):
    st.write("Models are fitted on the log of the index and trained only on data up to Feb 2020. The backtest (7 start dates, 24-month horizon) "
             "chooses the headline model: Ensemble for seasonally adjusted data, SARIMA for unadjusted and rebuilt data. The 95% range comes "
             "from the real backtest errors. The noise floor is the largest median fake gap among the 7 fake COVID starts. "
             "See the Analysis page for all models, tests and diagnostics.")