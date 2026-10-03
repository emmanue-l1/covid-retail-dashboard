"""
Look and feel for the dashboard (colours, fonts, cards, chart style).
Used by Story.py and pages/1_Analysis.py:  import ui; ui.setup()
Nothing here changes any number. Delete ui.py and the calls to it to get the plain look back.
"""
import html
import streamlit as st

INK, MUTED, LINE, BG, ACCENT = "#0F172A", "#64748B", "#E2E8F0", "#F5F7FB", "#2563EB"
FONT = "'Inter', -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
COLORWAY = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#5B6770"]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
.stApp button, .stApp input, .stApp textarea, .stApp [data-baseweb], .stApp [data-testid="stMarkdownContainer"] {{ font-family: {FONT}; }}
[data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded, .material-symbols-outlined {{
    font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important; }}
.stApp {{ background: {BG}; color: {INK}; }}
[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer, [data-testid="stDeployButton"], .stAppDeployButton {{ visibility: hidden; }}
[data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"], [data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarHeader"] button {{ visibility: visible !important; opacity: 1 !important; z-index: 999; }}
.block-container {{ max-width: 1180px; padding-top: 1.6rem; padding-bottom: 5rem; }}
h1, h2, h3, h4, h5 {{ color: {INK}; letter-spacing: -0.02em; font-weight: 700; }}
.stApp h2 {{ font-size: 1.6rem; padding: .4rem 0 .3rem; }}
.stApp h3 {{ font-size: 1.3rem; padding: .3rem 0 .2rem; }}
.stApp h4 {{ margin-top: 1.4rem; font-size: 1.12rem; padding: .2rem 0; }}
[data-testid="stHeaderActionElements"] {{ display: none; }}
p, li, label, .stMarkdown {{ color: #1E293B; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: {MUTED}; }}
a {{ color: {ACCENT}; }}

/* hero */
.hero {{ background: radial-gradient(1200px 400px at 85% -20%, rgba(37,99,235,.45), transparent 60%),
         linear-gradient(135deg, #0B1220 0%, #16233B 100%);
         border-radius: 24px; padding: 44px 48px 38px; color: #fff; margin-bottom: 18px;
         box-shadow: 0 18px 40px -18px rgba(11,18,32,.55); }}
.hero .eyebrow {{ font-size: 12px; letter-spacing: .16em; text-transform: uppercase; color: #93C5FD; font-weight: 600; }}
.hero h1 {{ color: #fff !important; font-size: 3rem; line-height: 1.05; margin: 10px 0 12px; font-weight: 800; letter-spacing: -0.035em; }}
.hero p {{ color: #CBD5E1 !important; font-size: 1.1rem; max-width: 720px; margin: 0 0 20px; line-height: 1.5; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.chip {{ background: rgba(255,255,255,.09); border: 1px solid rgba(255,255,255,.16); color: #E2E8F0;
         padding: 6px 12px; border-radius: 999px; font-size: 13px; font-weight: 500; }}
.chip b {{ color: #fff; }}

/* nav pills */
.nav {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 22px; }}
.nav a {{ text-decoration: none; background: #fff; border: 1px solid {LINE}; color: #334155 !important; font-size: 13px;
          font-weight: 600; padding: 6px 12px; border-radius: 999px; transition: all .15s; }}
.nav a:hover {{ border-color: {ACCENT}; color: {ACCENT} !important; transform: translateY(-1px); }}

/* KPI cards */
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin: 6px 0 8px; }}
.kpi {{ background: #fff; border: 1px solid {LINE}; border-radius: 18px; padding: 18px 20px 16px; position: relative; overflow: hidden;
        box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px -16px rgba(15,23,42,.18); }}
.kpi:before {{ content: ""; position: absolute; left: 0; top: 0; right: 0; height: 4px; background: var(--c); }}
.kpi .lab {{ font-size: 13px; font-weight: 600; color: {MUTED}; display: flex; align-items: center; gap: 8px; }}
.kpi .dot {{ width: 9px; height: 9px; border-radius: 50%; background: var(--c); display: inline-block; }}
.kpi .val {{ font-size: 2.5rem; font-weight: 800; letter-spacing: -0.04em; line-height: 1.1; margin: 8px 0 4px; color: {INK}; }}
.kpi .sub {{ font-size: 13px; color: {MUTED}; }}
.badge {{ display: inline-block; margin-top: 10px; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; }}
.badge.clear {{ background: #DCFCE7; color: #166534; }}
.badge.noise {{ background: #F1F5F9; color: #475569; }}

/* section header */
.sec {{ margin: 56px 0 14px; scroll-margin-top: 20px; }}
.sec .row {{ display: flex; align-items: center; gap: 14px; }}
.sec .num {{ flex: none; width: 40px; height: 40px; border-radius: 12px; background: {INK}; color: #fff; font-weight: 700;
             display: flex; align-items: center; justify-content: center; font-size: 16px; line-height: 1; }}
.stApp .sec h2 {{ margin: 0 !important; padding: 0 !important; font-size: 1.75rem; line-height: 1.15; }}
.sec p {{ margin: 8px 0 0 54px; color: {MUTED} !important; font-size: 1rem; max-width: 820px; }}
.sec.plain p {{ margin-left: 0; }}

/* key point */
.key {{ background: linear-gradient(90deg, rgba(37,99,235,.09), rgba(37,99,235,.02)); border-left: 4px solid {ACCENT};
        border-radius: 0 14px 14px 0; padding: 12px 18px; margin: 4px 0 16px; font-size: 1.02rem; color: {INK}; }}
.key .k {{ font-size: 11px; font-weight: 800; letter-spacing: .14em; color: {ACCENT}; text-transform: uppercase; display: block; margin-bottom: 2px; }}

/* charts, tables, widgets */
.js-plotly-plot .legendtitletext {{ display: none; }}
[data-testid="stPlotlyChart"] {{ background: #fff; border: 1px solid {LINE}; border-radius: 18px; padding: 10px 10px 4px;
        box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px -18px rgba(15,23,42,.2); margin: 6px 0 10px; }}
[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 14px; overflow: hidden; background: #fff; }}
[data-testid="stAlert"] {{ border-radius: 14px; border: 1px solid {LINE}; }}
[data-testid="stExpander"] {{ background: #fff; border: 1px solid {LINE}; border-radius: 14px; }}
[data-baseweb="select"] > div {{ border-radius: 12px; background: #fff; border-color: {LINE}; }}
[data-testid="stMetric"] {{ background: #fff; border: 1px solid {LINE}; border-radius: 16px; padding: 14px 16px; }}
[data-testid="stSidebar"] {{ background: #fff; border-right: 1px solid {LINE}; }}
hr {{ border-color: {LINE}; }}

/* tabs (analysis page): pills that wrap onto two lines, no scroll arrows */
.stTabs [data-baseweb="tab-list"] {{ gap: 8px; flex-wrap: wrap !important; overflow: visible !important; height: auto !important;
        border-bottom: 0; padding: 4px 0 14px; }}
.stTabs [data-baseweb="tab-list"] button[kind], .stTabs [data-baseweb="tab-list"] > button {{ display: none !important; }}
.stTabs [data-baseweb="tab"] {{ height: 38px; padding: 0 16px; border-radius: 999px; background: #fff; border: 1px solid {LINE};
        color: #475569; font-weight: 600; font-size: 14px; outline: none !important; box-shadow: none !important; }}
.stTabs [data-baseweb="tab"]:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
.stTabs [aria-selected="true"] {{ background: {INK}; border-color: {INK}; color: #fff !important; }}
.stTabs [aria-selected="true"] p {{ color: #fff !important; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none !important; }}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: 8px; }}

/* sidebar */
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ font-size: 1.15rem; }}
[data-testid="stSidebarNav"] a {{ border-radius: 10px; }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background: #EEF2FF; font-weight: 700; }}

/* controls bar (analysis page): short and compact */
[data-testid="stVerticalBlockBorderWrapper"] {{ background: #fff; border: 1px solid {LINE} !important; border-radius: 16px !important;
        box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px -18px rgba(15,23,42,.2); }}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stWidgetLabel"] p {{ font-size: 12px; font-weight: 700; color: {MUTED};
        letter-spacing: .02em; margin-bottom: 0; }}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stWidgetLabel"] {{ min-height: 0; margin-bottom: 2px; }}
[data-testid="stVerticalBlockBorderWrapper"] [data-baseweb="select"] > div {{ min-height: 38px; max-height: 76px; overflow-y: auto; }}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSlider"] {{ padding-top: 0; }}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stCheckbox"] {{ padding-top: 1.45rem; }}

/* left menu (analysis page) */
.st-key-nav {{ background: #fff; border: 1px solid {LINE}; border-radius: 16px; padding: 8px;
        box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px -18px rgba(15,23,42,.2); }}
.st-key-nav [role="radiogroup"] {{ gap: 2px; width: 100%; }}
.st-key-nav label {{ width: 100%; margin: 0; padding: 9px 12px; border-radius: 11px; cursor: pointer; transition: background .12s; }}
.st-key-nav label > div:first-child {{ display: none; }}
.st-key-nav label p {{ font-size: 14px; font-weight: 600; color: #475569; margin: 0; }}
.st-key-nav label:hover {{ background: #EEF3FF; }}
.st-key-nav label:has(input:checked) {{ background: {INK}; }}
.st-key-nav label:has(input:checked) p {{ color: #fff; }}

@media (max-width: 800px) {{ .hero {{ padding: 28px 22px; }} .hero h1 {{ font-size: 2.1rem; }} }}
</style>
"""


def setup():
    """Inject the CSS and make every Plotly chart use the same clean style."""
    st.markdown(CSS, unsafe_allow_html=True)
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        pio.templates["covid"] = go.layout.Template(layout=dict(
            font=dict(family=FONT, size=12, color="#334155"),
            title=dict(font=dict(size=15, color=INK), x=0.01, xanchor="left", y=0.96, yanchor="top", yref="container"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=COLORWAY,
            xaxis=dict(gridcolor="#EEF2F7", linecolor="#CBD5E1", zeroline=False, ticks="outside", tickcolor="#CBD5E1"),
            yaxis=dict(gridcolor="#EEF2F7", linecolor="rgba(0,0,0,0)", zeroline=False, ticks=""),
            legend=dict(orientation="h", x=0, xanchor="left", y=-0.24, yanchor="top", font=dict(size=12), title_text=""),
            hoverlabel=dict(bgcolor="white", bordercolor=LINE, font=dict(family=FONT, size=12, color=INK)),
            margin=dict(l=10, r=10, t=56, b=10),
            coloraxis=dict(colorbar=dict(outlinewidth=0, thickness=12)),
        ))
        pio.templates.default = "covid"
    except Exception:
        pass


def _e(t):
    return html.escape(str(t))


def hero(title, subtitle, chips=(), eyebrow=""):
    ch = "".join(f'<span class="chip">{c}</span>' for c in chips)
    st.markdown(f'<div class="hero"><div class="eyebrow">{_e(eyebrow)}</div><h1>{_e(title)}</h1><p>{_e(subtitle)}</p>'
                f'<div class="chips">{ch}</div></div>', unsafe_allow_html=True)


def nav(items):
    """items = [(section number, label)] -> pill links to the sections"""
    st.markdown('<div class="nav">' + "".join(f'<a href="#s{n}">{_e(n)}. {_e(l)}</a>' for n, l in items) + "</div>",
                unsafe_allow_html=True)


def kpi_cards(items):
    """items = list of dict(label, value, sub, color, badge, clear(bool), help)"""
    out = []
    for it in items:
        cls = "clear" if it.get("clear") else "noise"
        out.append(f'<div class="kpi" style="--c:{it["color"]}" title="{_e(it.get("help", ""))}">'
                   f'<div class="lab"><span class="dot"></span>{_e(it["label"])}</div>'
                   f'<div class="val">{_e(it["value"])}</div><div class="sub">{_e(it["sub"])}</div>'
                   f'<span class="badge {cls}">{_e(it["badge"])}</span></div>')
    st.markdown('<div class="kpis">' + "".join(out) + "</div>", unsafe_allow_html=True)


def section(num, title, lead=""):
    p = f"<p>{_e(lead)}</p>" if lead else ""
    st.markdown(f'<div class="sec" id="s{num}"><div class="row"><div class="num">{num}</div><h2>{_e(title)}</h2></div>{p}</div>',
                unsafe_allow_html=True)


def key(text):
    st.markdown(f'<div class="key"><span class="k">Key point</span>{_e(text)}</div>', unsafe_allow_html=True)


def page_header(title, subtitle, eyebrow=""):
    st.markdown(f'<div class="sec plain" style="margin-top:2px"><div style="font-size:12px;letter-spacing:.16em;text-transform:uppercase;'
                f'color:{ACCENT};font-weight:700;margin-bottom:6px">{_e(eyebrow)}</div>'
                f'<div class="row"><h2 style="font-size:2.3rem !important">{_e(title)}</h2></div><p>{_e(subtitle)}</p></div>',
                unsafe_allow_html=True)