"""
TrackPulse AI — Railway Operations Intelligence
Train delay forecasting, corridor analytics and network-level insights.
"""

import sys
import json
import textwrap
import html as html_lib
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import duckdb
import polars as pl

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.predictor import DelayPredictor
from src.graph_builder import get_zone_centrality_metrics
from src.trajectory_profiler import TrajectoryProfiler


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TrackPulse AI | Railway Operations Intelligence",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# DESIGN SYSTEM
# ============================================================

PRIMARY = "#092F35"
PRIMARY_LIGHT = "#10545A"
ACCENT = "#E59A45"
SUCCESS = "#18856F"
DANGER = "#D65353"
WARNING = "#D99A36"

BACKGROUND = "#F2F6F5"
CARD = "#FFFFFF"
TEXT = "#1B3032"
MUTED = "#718184"
BORDER = "#DCE6E4"

ZONE_FULL_NAMES = {
    "NCR": "North Central Railway",
    "ECR": "East Central Railway",
    "NR": "Northern Railway",
    "NER": "North Eastern Railway",
    "NE": "North Eastern Railway",
    "NFR": "Northeast Frontier Railway",
    "NWR": "North Western Railway",
    "ER": "Eastern Railway",
    "ECOR": "East Coast Railway",
    "SR": "Southern Railway",
    "SCR": "South Central Railway",
    "SER": "South Eastern Railway",
    "SECR": "South East Central Railway",
    "SWR": "South Western Railway",
    "WR": "Western Railway",
    "WCR": "West Central Railway",
    "CR": "Central Railway",
    "KRCL": "Konkan Railway"
}

RISK_SCALE = ["#83BDB0", "#E5A04B", "#D65353"]


# ============================================================
# RENDER HELPERS
# ============================================================

def render_html(markup: str):
    """
    Render raw HTML safely in Streamlit.

    Streamlit's Markdown parser treats lines indented by 4+ spaces as code
    blocks and ends an HTML block at a blank line. Stripping indentation and
    blank lines avoids blank/broken cards.
    """
    cleaned = "\n".join(
        line.strip() for line in markup.splitlines() if line.strip()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


def render_md(markup: str):
    """Render normal Markdown with indentation removed."""
    st.markdown(textwrap.dedent(markup).strip())


def esc(value) -> str:
    """Escape dynamic text before putting it inside HTML."""
    return html_lib.escape(str(value))


def style_figure(fig):
    """Force readable dark text colours on every part of a Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        title=dict(font=dict(color=PRIMARY, size=16)),
        legend=dict(
            font=dict(color=TEXT),
            title=dict(font=dict(color=TEXT))
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            font=dict(color=TEXT)
        ),
    )
    fig.update_xaxes(
        tickfont=dict(color=TEXT),
        title_font=dict(color=TEXT),
        gridcolor=BORDER,
        linecolor=BORDER,
    )
    fig.update_yaxes(
        tickfont=dict(color=TEXT),
        title_font=dict(color=TEXT),
        gridcolor=BORDER,
        linecolor=BORDER,
    )
    fig.update_annotations(font=dict(color=TEXT))
    return fig


def show_chart(fig, **kwargs):
    """Plotly chart with fixed text colours (works in light and dark themes)."""
    style_figure(fig)

    # theme=None stops Streamlit from overriding our colours.
    attempts = [
        dict(width="stretch", theme=None),
        dict(use_container_width=True, theme=None),
        dict(use_container_width=True),
    ]

    last_error = None
    for opts in attempts:
        try:
            st.plotly_chart(fig, **opts, **kwargs)
            return
        except TypeError as exc:
            last_error = exc

    raise last_error


def show_df(df, **kwargs):
    """Dataframe, compatible with old and new Streamlit versions."""
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def section_header(title: str, subtitle: str):
    render_html(f"""
    <div class="section-heading">{title}</div>
    <div class="section-subtitle">{subtitle}</div>
    """)


# ============================================================
# CUSTOM CSS
# ============================================================

render_html("""
<style>

/* Main application */
.stApp {
    background-color: #F2F6F5;
    color: #1B3032;
}

/* Remove large empty space at the top of the page */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* Hide the empty container that holds this <style> block (removes a blank gap) */
div[data-testid="stElementContainer"]:has(style),
.element-container:has(style) {
    display: none !important;
}

/* Main-area headings (force dark colour) */
section.main h1, section.main h2, section.main h3, section.main h4,
[data-testid="stMain"] h1, [data-testid="stMain"] h2,
[data-testid="stMain"] h3, [data-testid="stMain"] h4,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h4 {
    color: #092F35 !important;
}

/* Main-area body text */
section.main [data-testid="stMarkdownContainer"] p,
section.main label,
section.main [data-testid="stWidgetLabel"] p,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMain"] label,
[data-testid="stMain"] [data-testid="stWidgetLabel"] p {
    color: #1B3032;
}

section.main [data-testid="stCaptionContainer"] p,
[data-testid="stMain"] [data-testid="stCaptionContainer"] p {
    color: #718184 !important;
}

/* Plotly text inside the main area */
[data-testid="stMain"] .js-plotly-plot .gtitle,
[data-testid="stMain"] .js-plotly-plot .xtitle,
[data-testid="stMain"] .js-plotly-plot .ytitle,
[data-testid="stMain"] .js-plotly-plot .xtick text,
[data-testid="stMain"] .js-plotly-plot .ytick text,
[data-testid="stMain"] .js-plotly-plot .legendtext {
    fill: #1B3032 !important;
}

[data-testid="stMain"] .js-plotly-plot .gtitle {
    fill: #092F35 !important;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #092F35 !important;
}

/* Sidebar normal text */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #FFFFFF !important;
}

/* Sidebar headings */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    color: #E59A45 !important;
    font-weight: 800 !important;
}

/* Sidebar captions */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #C1DCD8 !important;
}

/* Sidebar alerts */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    border-radius: 9px;
}

/* Sidebar form controls */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #12464B !important;
    color: #FFFFFF !important;
    border-color: #28646A !important;
}

/* Sidebar navigation items */
.nav-item {
    background: #12464B;
    padding: 12px;
    border-radius: 9px;
    margin-bottom: 8px;
    color: #FFFFFF;
    font-size: 0.92rem;
}

/* Main brand header */
.brand {
    background: linear-gradient(110deg, #092F35, #12636A);
    padding: 27px 30px;
    border-radius: 14px;
    margin-bottom: 25px;
    color: white;
}

.brand-title {
    font-size: 2rem;
    font-weight: 800;
    color: white;
    letter-spacing: -0.5px;
}

.brand-subtitle {
    font-size: 1rem;
    color: #E59A45;
    font-weight: 700;
    margin-top: 4px;
}

.brand-description {
    font-size: 0.84rem;
    color: #D6E8E4;
    margin-top: 12px;
    line-height: 1.6;
}

/* Section headings */
.section-heading {
    color: #092F35;
    font-size: 1.55rem;
    font-weight: 800;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #718184;
    font-size: 0.9rem;
    margin-bottom: 20px;
}

/* Information cards */
.info-card {
    background: #FFFFFF;
    border: 1px solid #DCE6E4;
    border-radius: 12px;
    padding: 19px;
    margin-bottom: 13px;
    color: #1B3032;
}

.info-card.equal {
    min-height: 130px;
}

.info-card h4 {
    color: #092F35 !important;
    margin-top: 0;
    margin-bottom: 8px;
}

.info-card p {
    color: #718184 !important;
    font-size: 0.88rem;
    line-height: 1.6;
    margin: 0;
}

/* Selected service hero card */
.hero-card {
    background: linear-gradient(110deg, #092F35, #12636A);
    color: white;
    padding: 23px;
    border-radius: 12px;
    margin-bottom: 15px;
}

.hero-kicker {
    color: #C1E0DA;
    font-size: 0.78rem;
}

.hero-title {
    font-size: 1.65rem;
    font-weight: 800;
    margin: 7px 0;
    color: #FFFFFF;
}

.hero-meta {
    color: #E0F0EC;
    font-size: 0.9rem;
}

.hero-zone {
    color: #C1E0DA;
    font-size: 0.8rem;
    margin-top: 8px;
}

/* Risk labels */
.risk-low {
    color: #18856F;
    font-weight: 800;
}

.risk-medium {
    color: #D99A36;
    font-weight: 800;
}

.risk-high {
    color: #D65353;
    font-weight: 800;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    color: #718184;
    font-weight: 650;
}

.stTabs [aria-selected="true"] {
    color: #092F35 !important;
    border-bottom-color: #E59A45 !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #DCE6E4;
    padding: 15px;
    border-radius: 11px;
}

[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] {
    color: #718184 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #092F35 !important;
    font-weight: 800;
}

[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] div {
    color: #718184 !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #DCE6E4;
    border-radius: 9px;
}

/* Dividers */
hr {
    border-color: #DCE6E4 !important;
}

/* Footer */
.footer-title {
    color: #092F35;
    font-weight: 800;
    text-align: center;
}

.footer-description {
    color: #718184;
    font-size: 0.78rem;
    text-align: center;
}

</style>
""")


# ============================================================
# DATA PATHS
# ============================================================

DATA_DIR = BASE_DIR / "data"
TEST_CSV = DATA_DIR / "ir_test.csv"
DB_PATH = BASE_DIR / "db" / "railway.duckdb"
BENCHMARK_PATH = BASE_DIR / "models" / "benchmark_report.json"


# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_resource
def load_services():
    return DelayPredictor(), TrajectoryProfiler()


@st.cache_data
def load_benchmark_report():

    if BENCHMARK_PATH.exists():
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    return None


@st.cache_data
def load_sample_trains():

    if DB_PATH.exists():

        df = TrajectoryProfiler().get_available_trains(limit=500)

        if not df.empty:

            if "display_label" not in df.columns:

                if "train_type" in df.columns:
                    train_type = df["train_type"].astype(str)
                else:
                    train_type = pd.Series("Express", index=df.index)

                df["display_label"] = (
                    "Train "
                    + df["train_number"].astype(str)
                    + " - "
                    + train_type
                )

            return df

    if TEST_CSV.exists():

        df = pl.read_csv(
            str(TEST_CSV),
            n_rows=400
        ).to_pandas()

        df["display_label"] = (
            "Train "
            + df["train_number"].astype(str)
            + " - "
            + df["train_type"].astype(str)
            + " ("
            + df["zone_abbr"].astype(str)
            + ")"
        )

        return df

    return pd.DataFrame()


@st.cache_data
def load_zone_analytics():

    if not DB_PATH.exists():
        return pd.DataFrame()

    con = duckdb.connect(str(DB_PATH), read_only=True)

    try:

        df = con.execute("""
            SELECT
                zone_abbr,
                COUNT(*) AS total_journeys,
                ROUND(AVG(delay_minutes), 1) AS avg_delay_minutes,
                ROUND(AVG(is_delayed) * 100, 1) AS delay_rate_pct,
                ROUND(AVG(zone_congestion_index), 2) AS avg_congestion
            FROM journeys
            GROUP BY zone_abbr
            ORDER BY avg_delay_minutes DESC
        """).fetchdf()

    finally:
        con.close()

    if not df.empty:

        df["zone_full_name"] = (
            df["zone_abbr"]
            .map(ZONE_FULL_NAMES)
            .fillna(df["zone_abbr"])
        )

    return df


# ============================================================
# INITIALISE APPLICATION
# ============================================================

try:

    predictor, profiler = load_services()
    benchmark_data = load_benchmark_report()
    sample_trains_df = load_sample_trains()
    zone_df = load_zone_analytics()
    zone_centrality = get_zone_centrality_metrics()

except Exception as exc:

    st.error(f"Unable to initialise the application: {exc}")
    st.stop()


# ============================================================
# MAIN HEADER
# ============================================================

render_html("""
<div class="brand">
    <div class="brand-title">TrackPulse AI</div>
    <div class="brand-subtitle">Railway Operations Intelligence</div>
    <div class="brand-description">
        Historical-data-based analytics and model-driven forecasts.
        <br>
        Not a live railway tracking service.
    </div>
</div>
""")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html("""
    <div style="padding:8px 0 15px 0;">
        <div style="font-size:1.55rem;font-weight:800;color:white;">
            🚆 TrackPulse AI
        </div>
        <div style="font-size:0.7rem;color:#C1DCD8;letter-spacing:1px;">
            OPERATIONS INTELLIGENCE
        </div>
    </div>
    """)

    st.divider()

    st.markdown("### Navigation")

    render_html("""
    <div class="nav-item">🌐 &nbsp; Network overview</div>
    <div class="nav-item">🎯 &nbsp; Train forecasting</div>
    <div class="nav-item">🛤️ &nbsp; Corridor intelligence</div>
    <div class="nav-item">📊 &nbsp; Model evaluation</div>
    """)

    st.divider()

    st.markdown("### DATA & MODEL INFORMATION")

    st.caption(
        "Analytical railway intelligence using available journey "
        "records, historical data and trained prediction models."
    )

    st.warning("Forecasts are not live train tracking.")


# ============================================================
# NAVIGATION TABS
# ============================================================

overview_tab, forecast_tab, corridor_tab, model_tab = st.tabs([
    "🌐 Network Overview",
    "🎯 Train Forecast",
    "🛤️ Corridor Intelligence",
    "📊 Model & Architecture"
])


# ============================================================
# TAB 1 — NETWORK OVERVIEW
# ============================================================

with overview_tab:

    section_header(
        "National Railway Network Overview",
        "Explore recorded railway journey patterns, zone-level delays "
        "and network bottlenecks."
    )

    if zone_df.empty:

        st.warning("Zone analytics are unavailable. Check the database.")

    else:

        available_zones = ["All Zones"] + sorted(
            zone_df["zone_abbr"].dropna().unique().tolist()
        )

        selected_zone = st.selectbox(
            "Filter railway zone",
            available_zones,
            key="overview_zone_filter"
        )

        if selected_zone == "All Zones":
            overview_df = zone_df.copy()
        else:
            overview_df = zone_df[
                zone_df["zone_abbr"] == selected_zone
            ].copy()

        total_journeys = int(overview_df["total_journeys"].sum())

        avg_delay = (
            overview_df["avg_delay_minutes"].mean()
            if not overview_df.empty else 0
        )

        avg_delay_rate = (
            overview_df["delay_rate_pct"].mean()
            if not overview_df.empty else 0
        )

        total_zones = overview_df["zone_abbr"].nunique()

        a, b, c, d = st.columns(4)

        a.metric("Railway Zones", f"{total_zones}")
        b.metric("Recorded Journeys", f"{total_journeys:,}")
        c.metric("Average Delay", f"{avg_delay:.1f} min")
        d.metric("Average Delay Rate", f"{avg_delay_rate:.1f}%")

        st.divider()

        chart1, chart2 = st.columns(2)

        with chart1:

            st.markdown("### Zone Delay Comparison")

            zone_chart = overview_df.sort_values(
                "avg_delay_minutes",
                ascending=True
            )

            fig = px.bar(
                zone_chart,
                x="avg_delay_minutes",
                y="zone_full_name",
                orientation="h",
                color="avg_delay_minutes",
                color_continuous_scale=RISK_SCALE,
                labels={
                    "avg_delay_minutes": "Average delay (min)",
                    "zone_full_name": "Railway zone"
                }
            )

            fig.update_layout(
                template="plotly_white",
                height=410,
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=20, b=10)
            )

            show_chart(fig)

        with chart2:

            st.markdown("### Recorded Journey Delay Split")

            delayed = (
                overview_df["total_journeys"]
                * overview_df["delay_rate_pct"] / 100
            ).sum()

            on_time = max(total_journeys - delayed, 0)

            split_df = pd.DataFrame({
                "Journey Status": ["Delayed", "Not delayed"],
                "Journeys": [delayed, on_time]
            })

            fig_split = px.pie(
                split_df,
                names="Journey Status",
                values="Journeys",
                hole=0.65,
                color="Journey Status",
                color_discrete_map={
                    "Delayed": DANGER,
                    "Not delayed": SUCCESS
                }
            )

            fig_split.update_layout(
                template="plotly_white",
                height=410,
                margin=dict(t=20, b=10, l=10, r=10),
                legend_title_text=""
            )

            show_chart(fig_split)

            st.caption(
                "Journey split is calculated from aggregated recorded delay rates."
            )

        st.divider()

        st.markdown("### Network Bottleneck Monitor")

        try:

            bottlenecks = profiler.get_national_chokepoints(limit=10)

            if not isinstance(bottlenecks, pd.DataFrame):
                bottlenecks = pd.DataFrame(bottlenecks)

        except Exception:

            bottlenecks = pd.DataFrame()

        if not bottlenecks.empty:

            if {"avg_minutes_lost", "section"}.issubset(bottlenecks.columns):

                bottleneck_fig = px.bar(
                    bottlenecks.sort_values("avg_minutes_lost"),
                    x="avg_minutes_lost",
                    y="section",
                    orientation="h",
                    color="avg_minutes_lost",
                    color_continuous_scale=RISK_SCALE,
                    labels={
                        "section": "Railway section",
                        "avg_minutes_lost": "Average minutes lost"
                    }
                )

                bottleneck_fig.update_layout(
                    template="plotly_white",
                    height=390,
                    coloraxis_showscale=False,
                    margin=dict(l=10, r=10, t=20, b=10)
                )

                show_chart(bottleneck_fig)

            show_df(bottlenecks)

        else:

            st.info(
                "No bottleneck records are available from the current data source."
            )

        st.markdown("### Zone Performance Directory")

        show_df(
            overview_df[[
                "zone_abbr",
                "zone_full_name",
                "total_journeys",
                "avg_delay_minutes",
                "delay_rate_pct",
                "avg_congestion"
            ]].rename(columns={
                "zone_abbr": "Zone",
                "zone_full_name": "Railway Zone",
                "total_journeys": "Recorded Journeys",
                "avg_delay_minutes": "Average Delay (min)",
                "delay_rate_pct": "Delay Rate (%)",
                "avg_congestion": "Average Congestion"
            })
        )


# ============================================================
# TAB 2 — TRAIN FORECAST
# ============================================================

with forecast_tab:

    section_header(
        "Train Delay Forecasting",
        "Search for a train, inspect its prediction and analyse its route history."
    )

    if sample_trains_df.empty:

        st.warning("No train records are available.")

    else:

        filter_col, search_col = st.columns([1, 2])

        with filter_col:

            forecast_zones = ["All Zones"] + sorted(
                sample_trains_df["zone_abbr"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            forecast_zone = st.selectbox(
                "Railway zone",
                forecast_zones,
                key="forecast_zone"
            )

        filtered_trains = sample_trains_df.copy()

        if forecast_zone != "All Zones":

            filtered_trains = filtered_trains[
                filtered_trains["zone_abbr"].astype(str) == forecast_zone
            ]

        with search_col:

            train_search = st.text_input(
                "Search by train number or name",
                placeholder="Enter train number or train name"
            )

        if train_search.strip():

            search = train_search.strip().lower()

            mask = (
                filtered_trains["display_label"]
                .astype(str)
                .str.lower()
                .str.contains(search, na=False, regex=False)
            )

            filtered_trains = filtered_trains[mask]

        train_options = filtered_trains["display_label"].tolist()

        if not train_options:

            st.warning("No matching trains found. Change your filters.")

        else:

            default_index = 0

            for idx, label in enumerate(train_options):

                if "12301" in label or "22503" in label:
                    default_index = idx
                    break

            selected_label = st.selectbox(
                "Select train service",
                train_options,
                index=default_index,
                key="selected_train"
            )

            row = filtered_trains[
                filtered_trains["display_label"] == selected_label
            ].iloc[0]

            train_num = str(row["train_number"])

            train_name = str(
                row.get("train_name", row.get("train_type", "Express"))
            )

            zone_abbr = str(row.get("zone_abbr", "NR"))

            zone_full = ZONE_FULL_NAMES.get(zone_abbr, zone_abbr)

            dist_km = int(row.get("distance_km", 0) or 0)

            stop_cnt = int(row.get("stop_count", 0) or 0)

            fog_score = float(row.get("fog_risk_score", 0.25) or 0.25)

            late_rake = bool(row.get("late_incoming_rake", 0) == 1)

            pressure = float(row.get("avg_delay_mins", 42.0) or 42.0)

            try:

                traj_df, metrics = profiler.get_route_trajectory(train_num)

            except Exception as exc:

                st.error(f"Could not load route trajectory: {exc}")
                traj_df = pd.DataFrame()
                metrics = {}

            if traj_df is None:
                traj_df = pd.DataFrame()

            if metrics is None:
                metrics = {}

            origin_stn = metrics.get("origin_station", "Origin")
            dest_stn = metrics.get("destination_station", "Destination")

            centrality = zone_centrality.get(
                zone_abbr,
                {
                    "corridor_betweenness_centrality": 0.05,
                    "corridor_degree_centrality": 0.05
                }
            )

            features = {
                "zone_abbr": zone_abbr,
                "train_type": str(
                    row.get("train_type", "Superfast Express")
                ),
                "season": "Winter/Fog",
                "distance_km": dist_km,
                "num_scheduled_stops": stop_cnt,
                "scheduled_travel_hours": float(dist_km / 65.0),
                "departure_hour": 8,
                "is_peak_hour": 1,
                "is_hdn_route": (
                    1 if zone_abbr in ["NR", "NCR", "ECR"] else 0
                ),
                "is_fog_risk": 1 if fog_score > 0.4 else 0,
                "fog_risk_score": fog_score,
                "zone_congestion_index": 0.82,
                "late_incoming_rake": 1 if late_rake else 0,
                "is_rake_shared": 1,
                "rake_cascade_chain_length": 2 if late_rake else 0,
                "zone_delay_pressure": pressure,
                "route_historical_ontime_pct": 74.0,
                "loco_age_years": 8.0,
                "coach_age_years": 6.0,
                "maintenance_score": 8.5,
                "seat_utilisation_pct": 92.0,
                "is_overloaded": 0,
                "is_monsoon_season": 0,
                "corridor_betweenness_centrality":
                    centrality["corridor_betweenness_centrality"],
                "corridor_degree_centrality":
                    centrality["corridor_degree_centrality"]
            }

            try:

                prediction = predictor.predict(features)

                pred_delay = float(prediction["predicted_delay_minutes"])

                pred_probability = float(prediction["delay_probability_pct"])

                risk_tier = prediction.get("risk_category", "Not available")

            except Exception as exc:

                st.error(f"Prediction failed: {exc}")
                st.stop()

            if pred_delay < 15:
                risk_class = "risk-low"
                risk_label = "Low"

            elif pred_delay < 45:
                risk_class = "risk-medium"
                risk_label = "Moderate"

            else:
                risk_class = "risk-high"
                risk_label = "High"

            render_html(f"""
            <div class="hero-card">
                <div class="hero-kicker">SELECTED SERVICE · {esc(zone_abbr)}</div>
                <div class="hero-title">{esc(train_num)} — {esc(train_name)}</div>
                <div class="hero-meta">
                    {esc(origin_stn)} → {esc(dest_stn)}
                    &nbsp; | &nbsp; {dist_km:,} km
                    &nbsp; | &nbsp; {stop_cnt} scheduled stops
                </div>
                <div class="hero-zone">{esc(zone_full)}</div>
            </div>
            """)

            k1, k2, k3, k4 = st.columns(4)

            k1.metric(
                "Predicted Delay",
                f"{pred_delay:.1f} min",
                risk_label + " risk",
                delta_color="off"
            )

            k2.metric("Delay Probability", f"{pred_probability:.1f}%")

            k3.metric("Route Distance", f"{dist_km:,} km")

            k4.metric("Scheduled Stops", f"{stop_cnt}")

            st.caption(
                "Model output based on the supplied feature values. "
                "This application does not provide live train tracking."
            )

            st.divider()

            left, right = st.columns(2)

            with left:

                st.markdown("### Prediction Assessment")

                render_html(f"""
                <div class="info-card">
                    <div style="color:{MUTED};font-size:0.8rem;">
                        MODEL RISK CATEGORY
                    </div>
                    <div class="{risk_class}" style="font-size:1.55rem;margin:8px 0;">
                        {esc(risk_tier)}
                    </div>
                    <div style="color:{MUTED};font-size:0.87rem;">
                        Predicted delay: <b>{pred_delay:.1f} minutes</b>
                    </div>
                    <div style="color:{MUTED};font-size:0.87rem;">
                        Delay probability: <b>{pred_probability:.1f}%</b>
                    </div>
                </div>
                """)

                st.markdown("### Operating Conditions")

                st.write(f"**Railway zone:** {zone_full}")
                st.write(f"**Fog risk score:** {fog_score:.2f}")
                st.write(f"**Historical delay pressure:** {pressure:.1f} min")
                st.write(
                    "**Incoming rake delay:** "
                    + ("Yes" if late_rake else "No")
                )

            with right:

                st.markdown("### Delay Attribution")

                track_loss = float(
                    metrics.get("total_track_loss_mins", 0.0) or 0.0
                )

                dwell_loss = float(
                    metrics.get("total_dwell_loss_mins", 0.0) or 0.0
                )

                loss_df = pd.DataFrame({
                    "Delay Component": ["Track / Running", "Station Dwell"],
                    "Minutes": [track_loss, dwell_loss]
                })

                if loss_df["Minutes"].sum() > 0:

                    fig_loss = px.pie(
                        loss_df,
                        names="Delay Component",
                        values="Minutes",
                        hole=0.62,
                        color_discrete_sequence=[ACCENT, SUCCESS]
                    )

                    fig_loss.update_layout(
                        template="plotly_white",
                        height=290,
                        margin=dict(t=15, b=15, l=10, r=10),
                        legend_title_text=""
                    )

                    show_chart(fig_loss)

                else:

                    st.info("Route delay attribution is unavailable.")

            st.divider()

            st.markdown("### Prediction Input Transparency")

            st.caption(
                "These are the feature values supplied to the model. "
                "They are not necessarily real-time measurements."
            )

            feature_df = pd.DataFrame(
                [(k, str(v)) for k, v in features.items()],
                columns=["Model Feature", "Supplied Value"]
            )

            show_df(feature_df)

            with st.expander("Understand the prediction inputs"):

                render_md("""
                The prediction is generated from a combination of:

                - **Route features:** distance, scheduled stops and railway zone.
                - **Operating indicators:** fog risk, rake delay and delay pressure.
                - **Network indicators:** corridor centrality metrics.
                - **Application assumptions:** certain operating values are currently
                  fixed in the application rather than dynamically retrieved.

                A feature list alone does not establish which variables caused
                a prediction. Feature attribution would require a validated
                explainability method such as SHAP.
                """)

            st.divider()

            st.markdown("### Station-by-Station Route Analysis")

            if not traj_df.empty:

                timeline_cols = [
                    col for col in [
                        "stop_sequence",
                        "station_name",
                        "station_code",
                        "cumulative_distance_km",
                        "sch_arr_time",
                        "act_arr_time",
                        "arr_delay_mins",
                        "running_delay_delta",
                        "dwell_delay_delta"
                    ]
                    if col in traj_df.columns
                ]

                if {"arr_delay_mins", "station_name"}.issubset(traj_df.columns):

                    chart_df = traj_df.copy()

                    chart_df["arr_delay_mins"] = pd.to_numeric(
                        chart_df["arr_delay_mins"],
                        errors="coerce"
                    )

                    fig_route = px.bar(
                        chart_df,
                        x="station_name",
                        y="arr_delay_mins",
                        color="arr_delay_mins",
                        color_continuous_scale=[SUCCESS, ACCENT, DANGER],
                        labels={
                            "station_name": "Station",
                            "arr_delay_mins": "Arrival delay (minutes)"
                        },
                        title="Recorded Arrival Delay Across Route"
                    )

                    fig_route.update_layout(
                        template="plotly_white",
                        height=350,
                        coloraxis_showscale=False,
                        xaxis_tickangle=-35,
                        margin=dict(l=10, r=10, t=50, b=10)
                    )

                    show_chart(fig_route)

                if timeline_cols:

                    show_df(
                        traj_df[timeline_cols].rename(columns={
                            "stop_sequence": "Stop",
                            "station_name": "Station",
                            "station_code": "Code",
                            "cumulative_distance_km": "Distance (km)",
                            "sch_arr_time": "Scheduled Arrival",
                            "act_arr_time": "Recorded Arrival",
                            "arr_delay_mins": "Arrival Delay (min)",
                            "running_delay_delta": "Running Delay Change",
                            "dwell_delay_delta": "Dwell Delay Change"
                        })
                    )

            else:

                st.info("Station sequence data is unavailable.")

            st.divider()

            st.markdown("### Downstream Cascade Impact")

            try:

                impacted = profiler.get_downstream_impacted_trains(
                    train_num,
                    date=metrics.get("journey_date"),
                    limit=5
                )

            except Exception:

                impacted = []

            if isinstance(impacted, pd.DataFrame):
                impacted_has_rows = not impacted.empty
                impacted_df = impacted
            else:
                impacted_has_rows = bool(impacted)
                impacted_df = pd.DataFrame(impacted) if impacted_has_rows else None

            if impacted_has_rows:

                show_df(impacted_df)

            else:

                st.info("No downstream impact records were returned.")

            if metrics.get("worst_sections"):

                st.markdown("### Identified Bottleneck Sections")

                show_df(pd.DataFrame(metrics["worst_sections"]))


# ============================================================
# TAB 3 — CORRIDOR INTELLIGENCE
# ============================================================

with corridor_tab:

    section_header(
        "Corridor Intelligence",
        "Compare railway zones, congestion indicators and recorded bottleneck sections."
    )

    if zone_df.empty:

        st.warning("Zone analytics are unavailable.")

    else:

        all_zone_codes = sorted(
            zone_df["zone_abbr"].dropna().unique().tolist()
        )

        selected_corridor_zones = st.multiselect(
            "Compare railway zones",
            options=all_zone_codes,
            default=all_zone_codes,
            format_func=lambda x: f"{x} — {ZONE_FULL_NAMES.get(x, x)}"
        )

        comparison_df = zone_df[
            zone_df["zone_abbr"].isin(selected_corridor_zones)
        ].copy()

        if comparison_df.empty:

            st.info("Select at least one zone to view comparisons.")

        else:

            c1, c2, c3 = st.columns(3)

            c1.metric("Zones Compared", comparison_df["zone_abbr"].nunique())

            c2.metric(
                "Journeys",
                f"{int(comparison_df['total_journeys'].sum()):,}"
            )

            c3.metric(
                "Mean Delay",
                f"{comparison_df['avg_delay_minutes'].mean():.1f} min"
            )

            st.divider()

            left, right = st.columns(2)

            with left:

                fig_delay = px.bar(
                    comparison_df.sort_values("avg_delay_minutes"),
                    x="avg_delay_minutes",
                    y="zone_full_name",
                    orientation="h",
                    color="avg_delay_minutes",
                    color_continuous_scale=[SUCCESS, ACCENT, DANGER],
                    title="Average Delay by Zone",
                    labels={
                        "avg_delay_minutes": "Average delay (min)",
                        "zone_full_name": "Railway zone"
                    }
                )

                fig_delay.update_layout(
                    template="plotly_white",
                    height=440,
                    coloraxis_showscale=False,
                    margin=dict(l=10, r=10, t=50, b=10)
                )

                show_chart(fig_delay)

            with right:

                fig_rate = px.bar(
                    comparison_df.sort_values("delay_rate_pct"),
                    x="delay_rate_pct",
                    y="zone_full_name",
                    orientation="h",
                    color="delay_rate_pct",
                    color_continuous_scale=[SUCCESS, ACCENT, DANGER],
                    title="Recorded Delay Rate by Zone",
                    labels={
                        "delay_rate_pct": "Delay rate (%)",
                        "zone_full_name": "Railway zone"
                    }
                )

                fig_rate.update_layout(
                    template="plotly_white",
                    height=440,
                    coloraxis_showscale=False,
                    margin=dict(l=10, r=10, t=50, b=10)
                )

                show_chart(fig_rate)

            st.divider()

            st.markdown("### Congestion Indicator Comparison")

            fig_congestion = px.scatter(
                comparison_df,
                x="avg_congestion",
                y="avg_delay_minutes",
                size="total_journeys",
                color="zone_abbr",
                hover_name="zone_full_name",
                labels={
                    "avg_congestion": "Average congestion index",
                    "avg_delay_minutes": "Average delay (minutes)",
                    "zone_abbr": "Zone"
                }
            )

            fig_congestion.update_layout(
                template="plotly_white",
                height=390,
                margin=dict(l=10, r=10, t=20, b=10)
            )

            show_chart(fig_congestion)

            st.caption(
                "This chart displays associations in the available data; "
                "it does not establish that congestion causes delay."
            )

            st.markdown("### Zone Performance Directory")

            show_df(
                comparison_df[[
                    "zone_abbr",
                    "zone_full_name",
                    "total_journeys",
                    "avg_delay_minutes",
                    "delay_rate_pct",
                    "avg_congestion"
                ]].rename(columns={
                    "zone_abbr": "Zone",
                    "zone_full_name": "Railway Zone",
                    "total_journeys": "Journeys",
                    "avg_delay_minutes": "Average Delay (min)",
                    "delay_rate_pct": "Delay Rate (%)",
                    "avg_congestion": "Congestion Index"
                })
            )

            st.divider()

            st.markdown("### National Bottleneck Sections")

            try:

                chokepoints_df = profiler.get_national_chokepoints(limit=10)

                if not isinstance(chokepoints_df, pd.DataFrame):
                    chokepoints_df = pd.DataFrame(chokepoints_df)

            except Exception:

                chokepoints_df = pd.DataFrame()

            if not chokepoints_df.empty:

                show_df(chokepoints_df)

            else:

                st.info("Bottleneck section records are unavailable.")


# ============================================================
# TAB 4 — MODEL & ARCHITECTURE
# ============================================================

with model_tab:

    section_header(
        "Model Evaluation & Architecture",
        "Inspect available model benchmark results and the application technology stack."
    )

    st.markdown("### Model Benchmark Report")

    if benchmark_data is not None:

        if isinstance(benchmark_data, dict):

            st.json(benchmark_data)

        elif isinstance(benchmark_data, list):

            show_df(pd.DataFrame(benchmark_data))

        else:

            st.write(benchmark_data)

    else:

        st.info("No benchmark_report.json found in the models directory.")

    st.divider()

    st.markdown("### Technology Architecture")

    architecture = pd.DataFrame([
        {
            "Layer": "Data Processing",
            "Technology": "Polars",
            "Responsibility": "ETL and data transformations"
        },
        {
            "Layer": "Analytical Storage",
            "Technology": "DuckDB",
            "Responsibility": "Analytical database queries"
        },
        {
            "Layer": "Machine Learning",
            "Technology": "LightGBM",
            "Responsibility": "Train delay prediction"
        },
        {
            "Layer": "Network Analytics",
            "Technology": "NetworkX",
            "Responsibility": "Railway network graph metrics"
        },
        {
            "Layer": "Visualisation",
            "Technology": "Plotly",
            "Responsibility": "Interactive charts"
        },
        {
            "Layer": "Application",
            "Technology": "Streamlit",
            "Responsibility": "Dashboard and user interaction"
        }
    ])

    show_df(architecture)

    st.divider()

    st.markdown("### Application Modules")

    left, right = st.columns(2)

    with left:

        render_html("""
        <div class="info-card equal">
            <h4>01 · Data Engineering</h4>
            <p>
                Data transformation, journey processing and analytical
                storage through Polars and DuckDB.
            </p>
        </div>
        <div class="info-card equal">
            <h4>02 · Predictive Modelling</h4>
            <p>
                Feature-based delay inference and risk categorisation
                using the trained prediction model.
            </p>
        </div>
        """)

    with right:

        render_html("""
        <div class="info-card equal">
            <h4>03 · Network Intelligence</h4>
            <p>
                Zone connectivity, corridor metrics and downstream
                cascade impact inspection.
            </p>
        </div>
        <div class="info-card equal">
            <h4>04 · Decision Dashboard</h4>
            <p>
                Train search, route analysis, network summaries
                and model benchmark inspection.
            </p>
        </div>
        """)

    st.divider()

    st.markdown("### Model Usage Notes")

    st.info(
        "The dashboard displays the output of the existing trained model. "
        "Model quality should be assessed using its actual benchmark report, "
        "held-out evaluation data and validation methodology."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

_, footer_col, _ = st.columns([1, 3, 1])

with footer_col:

    render_html("""
    <div class="footer-title">
        TrackPulse AI · Railway Operations Intelligence
    </div>
    <div class="footer-description">
        Historical-data-based analytics and model-driven forecasts.
        Not a live railway tracking service.
    </div>
    """)