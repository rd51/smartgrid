"""
Smart City Energy Grid — Control Room Dashboard
================================================
Run:  streamlit run dashboard/app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from env.grid_env import GridEnv, SOURCES, DEFAULT_CONFIG
from agents.solver import run_episode_solver

st.set_page_config(page_title="Smart City Grid", layout="wide", page_icon="⚡")

# ----------------------------- palette & style ----------------------------- #
PALETTE = {
    "solar": "#F6C445",   # gold
    "wind": "#2EC4B6",    # teal
    "gas": "#E8743B",     # orange
    "coal": "#9AA3B2",    # grey
    "battery": "#9B8CFF", # purple
    "city": "#5BD99C",    # green
}
ICONS = {
    "solar": "☀️", "wind": "💨", "gas": "🔥", "coal": "⚫", "battery": "🔋", "city": "🏙️",
}
GROUP = {
    "solar": "Renewable", "wind": "Renewable",
    "gas": "Conventional", "coal": "Conventional",
    "battery": "Storage",
}
SPEED_TO_SEC = {"0.5x": 1.6, "1x": 0.8, "2x": 0.4, "4x": 0.2}

# Status-color discipline: green = healthy, amber = strained, red = unmet/over cap.
STATUS_GOOD, STATUS_WARN, STATUS_BAD = "#5BD99C", "#F6C445", "#F0626B"
STATUS_LABEL = {STATUS_GOOD: "Healthy", STATUS_WARN: "Strained", STATUS_BAD: "Critical"}

st.markdown("""
<style>
  .stApp { background: #0E1117; }
  .metric-card { background:#171B26; border:1px solid #262C3A; border-radius:14px;
    padding:14px 16px; transition: border-color .15s; }
  .metric-card:hover { border-color:#3A4256; }
  .metric-icon { font-size:18px; margin-bottom:4px; }
  .metric-val { font-size:26px; font-weight:700; color:#F2F4F8; line-height:1.1; }
  .metric-lbl { font-size:11px; color:#8A93A6; text-transform:uppercase;
    letter-spacing:.08em; margin-top:6px; }
  .stSlider label, .stSelectbox label, .stRadio label { color:#C4CBDA !important; }
  .engine-banner { background:linear-gradient(90deg,#171B26,#1E2433);
    border:1px solid #2C3346; border-radius:14px; padding:14px 20px;
    font-size:15px; color:#F2F4F8; margin-bottom:14px; }
  .engine-banner b { color:#F6C445; }
  .insight-box { background:#15281F; border:1px solid #2C4A3A; border-radius:12px;
    padding:14px 18px; font-size:15px; color:#E3F4EC; margin:8px 0 16px 0; }
  .legend-pill { display:inline-block; padding:2px 10px; border-radius:999px;
    font-size:11px; font-weight:700; letter-spacing:.04em; margin-right:6px; }
  .header-band { background:linear-gradient(135deg,#12161F 0%,#1B2333 100%);
    border:1px solid #262C3A; border-radius:18px; padding:26px 32px 22px 32px;
    margin-bottom:18px; }
  .eyebrow { font-size:12px; letter-spacing:.3em; text-transform:uppercase;
    color:#7C8AA5; font-weight:700; }
  .header-title { font-size:38px; font-weight:800; color:#F2F4F8;
    letter-spacing:-.02em; margin:8px 0 6px 0; }
  .header-tagline { font-size:15px; color:#C4CBDA; }
  .donut-legend { text-align:center; font-size:12px; color:#8A93A6; margin-top:-8px; }
  .status-dot { display:inline-block; width:9px; height:9px; border-radius:50%;
    margin-right:5px; }
  .status-tag { display:inline-block; padding:2px 12px; border-radius:999px;
    font-size:12px; font-weight:700; }
  h1 { color:#F2F4F8 !important; font-weight:800 !important; letter-spacing:-.02em; }
</style>
""", unsafe_allow_html=True)

# ----------------------------- header band ----------------------------------- #
st.markdown(
    '<div class="header-band">'
    '<div class="eyebrow">Smart City Energy Grid · RDMU Topic 5</div>'
    '<div class="header-title">⚡ Energy Grid Optimizer</div>'
    '<div class="header-tagline">Balancing clean and conventional power to meet a city\'s '
    'demand every hour — without blackouts, at the lowest cost and emissions.</div>'
    '</div>', unsafe_allow_html=True)

# ----------------------------- sidebar: nav + controls ----------------------- #
with st.sidebar:
    st.markdown("### 🧭 Navigate")
    mode = st.radio("View", ["Overview", "Live Dispatch", "Engine Comparison",
                              "Trade-offs", "Decision Log"],
                     label_visibility="collapsed")
    st.divider()
    st.header("🌍 City scenario controls")
    ren = st.slider("Clean energy availability", 0.3, 1.8, 1.0, 0.1,
                     help="Scales how much solar & wind power the city can draw on. "
                          "1.0 = baseline; higher means more installed renewable capacity.")
    var = st.slider("Demand unpredictability", 0.0, 0.30, 0.10, 0.02,
                     help="How much hour-to-hour electricity demand can randomly swing — "
                          "the harder this is to forecast, the riskier the dispatch.")
    cap = st.slider("Daily carbon budget (tCO₂)", 100, 800, 400, 50,
                     help="Total emissions the city aims to stay under for the day. "
                          "Going over this budget triggers a steep carbon penalty.")
    pen = st.slider("Carbon price ($/tCO₂)", 0, 200, 40, 10,
                     help="The cost penalty applied per tonne of CO₂ emitted — higher "
                          "prices push dispatch toward cleaner sources.")
    seed = st.number_input("Scenario seed", 0, 9999, 7, 1,
                            help="Change this to generate a different random day "
                                 "(demand pattern & weather).")
    st.divider()
    st.subheader("🤖 Dispatch engine")
    engine = st.radio("Who controls the grid?",
                       ["Constraint solver (LP)", "Both — compare with SAC agent"],
                       index=1,
                       help="The LP solver computes a provably least-cost dispatch each "
                            "hour. The SAC agent is a trained reinforcement-learning "
                            "policy, kept feasible by a safety projection.")

cfg = {"renewable_penetration": ren, "demand_variability": var,
       "emission_penalty": pen, "emission_cap": cap, "seed": int(seed)}

CAPACITY = DEFAULT_CONFIG["capacity"]
BATTERY_CAPACITY = DEFAULT_CONFIG["battery_capacity"]
HORIZON = DEFAULT_CONFIG["horizon"]
GAS_COST = DEFAULT_CONFIG["cost"]["gas"]

# ----------------------------- cached episodes ------------------------------ #
@st.cache_data(show_spinner=False)
def run_solver_episode(cfg):
    env = GridEnv(cfg)
    hist = run_episode_solver(env)
    return pd.DataFrame(hist)

@st.cache_resource(show_spinner=False)
def load_sac_model(path):
    from stable_baselines3 import SAC
    return SAC.load(path)

@st.cache_data(show_spinner=False)
def run_agent_episode(_model, cfg):
    from agents.rl_agent import run_episode_agent
    hist = run_episode_agent(_model, cfg, project=True)
    return pd.DataFrame(hist)

df = run_solver_episode(cfg)

agent_df = None
agent_error = None
ckpt = os.path.join(os.path.dirname(__file__), "..", "assets", "sac_grid.zip")
if engine.startswith("Both"):
    if os.path.exists(ckpt):
        try:
            model = load_sac_model(ckpt[:-4])
            agent_df = run_agent_episode(model, cfg)
        except Exception as e:
            agent_error = str(e)
    else:
        agent_error = "no checkpoint found at assets/sac_grid.zip"

# ----------------------------- engine banner -------------------------------- #
if engine.startswith("Both") and agent_df is not None:
    st.markdown('<div class="engine-banner">🆚 Comparing the <b>Constraint Solver (LP)</b> '
                 'against the trained <b>SAC reinforcement-learning agent</b> '
                 '(with feasibility projection) on the same scenario.</div>',
                 unsafe_allow_html=True)
elif engine.startswith("Both"):
    st.markdown(f'<div class="engine-banner">📐 Running the <b>Constraint Solver (LP)</b> only — '
                 f'the SAC agent is unavailable ({agent_error}). Train one with '
                 f'<code>python -m agents.train_sac 8000</code>.</div>',
                 unsafe_allow_html=True)
else:
    st.markdown('<div class="engine-banner">📐 Running the <b>Constraint Solver (LP)</b> only.</div>',
                 unsafe_allow_html=True)

# ----------------------------- live source selection ------------------------ #
live_options = ["Constraint solver"]
if agent_df is not None:
    live_options.append("SAC agent")

if len(live_options) > 1:
    live_src = st.radio("**🔎 Viewing:**", live_options, horizontal=True, key="live_src")
else:
    live_src = live_options[0]
    st.markdown(f"**🔎 Viewing:** {live_src}")

live_df = df if live_src == "Constraint solver" else agent_df

# ----------------------------- helpers --------------------------------------- #
def kpis(d):
    supply = d["supply"].sum()
    renew = d[["d_solar", "d_wind"]].sum().sum() / supply if supply else 0
    return {
        "cost": d["cost"].sum(),
        "emit": d["emit"].sum(),
        "renew": renew,
        "unmet": d["unmet"].sum(),
        "stability": 1 - (d["unmet"].sum() / d["demand"].sum()),
    }

k = kpis(live_df)
cost_ref = max(1.0, df["demand"].sum() * GAS_COST)
cost_score = max(0.0, min(100.0, (1 - k["cost"] / cost_ref) * 100))

# ----------------------------- baseline & sensitivity ------------------------ #
# Baseline for "emission reduction %": an all-fossil grid (no renewables) facing
# zero carbon price — i.e. business-as-usual with no clean energy and no climate
# policy, on the same demand scenario.
BASELINE_LABEL = "all-fossil dispatch, zero carbon price (renewable penetration = 0, carbon price = $0)"
baseline_cfg = {**cfg, "renewable_penetration": 0.0, "emission_penalty": 0.0,
                "emission_cap": 1e9}
baseline_df = run_solver_episode(baseline_cfg)
baseline_emit = baseline_df["emit"].sum()
emit_reduction_pct = ((baseline_emit - k["emit"]) / baseline_emit * 100) if baseline_emit > 0 else 0.0

CAP_SWEEP_RANGE = [100, 150, 200, 300, 400, 550, 700, 850]

@st.cache_data(show_spinner=False)
def run_cap_sweep(base_cfg, caps):
    rows = []
    for c in caps:
        cc = {**base_cfg, "emission_cap": c}
        d = run_solver_episode(cc)
        rows.append({"cap": c, "cost": d["cost"].sum(), "unmet": d["unmet"].sum(),
                      "emit": d["emit"].sum()})
    return pd.DataFrame(rows)

def status_for(value, good, warn, higher_is_better=True):
    if higher_is_better:
        if value >= good:
            return STATUS_GOOD
        if value >= warn:
            return STATUS_WARN
        return STATUS_BAD
    else:
        if value <= good:
            return STATUS_GOOD
        if value <= warn:
            return STATUS_WARN
        return STATUS_BAD

def build_donut(frac, title, good, warn, higher_is_better=True):
    frac = max(0.0, min(1.0, frac))
    pct = frac * 100
    color = status_for(pct, good, warn, higher_is_better)
    fig = go.Figure(go.Pie(
        values=[frac, max(0.0001, 1 - frac)], hole=0.72,
        marker=dict(colors=[color, "#262C3A"], line=dict(color="#0E1117", width=2)),
        textinfo="none", sort=False, direction="clockwise",
        hoverinfo="skip", showlegend=False))
    fig.add_annotation(text=f"<b>{pct:.0f}%</b>", x=0.5, y=0.54, showarrow=False,
                        font=dict(size=30, color="#F2F4F8"))
    fig.add_annotation(text=title, x=0.5, y=0.54, yshift=-44, showarrow=False,
                        font=dict(size=12, color="#8A93A6"))
    fig.update_layout(height=210, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                       margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    return fig, color

def build_gauges(k, cost_score):
    sustain_color = status_for(k["renew"] * 100, 50, 25)
    reli_color = status_for(k["stability"] * 100, 98, 90)
    cost_color = status_for(cost_score, 60, 30)
    gfig = make_subplots(rows=1, cols=3, specs=[[{"type": "indicator"}] * 3],
                          subplot_titles=("Sustainability — clean energy share",
                                           "Reliability — demand met",
                                           "Cost efficiency"))
    gfig.add_trace(go.Indicator(
        mode="gauge+number", value=k["renew"] * 100, number={"suffix": "%"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": sustain_color}}), row=1, col=1)
    gfig.add_trace(go.Indicator(
        mode="gauge+number", value=k["stability"] * 100, number={"suffix": "%"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": reli_color}}), row=1, col=2)
    gfig.add_trace(go.Indicator(
        mode="gauge+number", value=cost_score,
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": cost_color}}), row=1, col=3)
    gfig.update_layout(height=240, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=30, r=30, t=50, b=10), font=dict(color="#F2F4F8"))
    return gfig

NODE_POS = {
    "solar": (0.05, 0.90),
    "wind": (0.05, 0.65),
    "gas": (0.05, 0.40),
    "coal": (0.05, 0.15),
    "battery": (0.55, 0.08),
    "city": (0.95, 0.50),
}
NODE_LABELS = {
    "solar": "Solar", "wind": "Wind", "gas": "Gas", "coal": "Coal",
    "battery": "Battery", "city": "City Load",
}

def build_schematic(row):
    fig = go.Figure()
    max_w = 16
    for s in ["solar", "wind", "gas", "coal"]:
        x0, y0 = NODE_POS[s]
        x1, y1 = NODE_POS["city"]
        val = max(0.0, row[f"d_{s}"])
        capv = CAPACITY[s]
        ratio = min(1.0, val / capv) if capv else 0.0
        width = 1.5 + max_w * ratio
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(width=width, color=PALETTE[s]), opacity=0.85,
            hovertemplate=f"{s.capitalize()}: {val:.1f} MW<extra></extra>",
            showlegend=False))

    bx, by = NODE_POS["battery"]
    cx, cy = NODE_POS["city"]
    bval = row["d_battery"]
    ratio = min(1.0, abs(bval) / CAPACITY["battery"]) if CAPACITY["battery"] else 0.0
    width = 1.5 + max_w * ratio
    direction = "discharging → city" if bval >= 0 else "charging ← from city"
    fig.add_trace(go.Scatter(
        x=[bx, cx], y=[by, cy], mode="lines",
        line=dict(width=width, color=PALETTE["battery"], dash="solid" if bval >= 0 else "dot"),
        opacity=0.85,
        hovertemplate=f"Battery {direction}: {abs(bval):.1f} MW<extra></extra>",
        showlegend=False))

    for n, (x, y) in NODE_POS.items():
        if n == "city":
            value_txt = f"{row['demand']:.0f} MW demand"
        elif n == "battery":
            value_txt = f"{row['soc']:.0f} / {BATTERY_CAPACITY:.0f} MWh stored"
        else:
            value_txt = f"{max(0.0, row[f'd_{n}']):.0f} MW"
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=52, color="#171B26", line=dict(width=3, color=PALETTE[n])),
            text=[ICONS[n]], textfont=dict(size=22), textposition="middle center",
            hoverinfo="skip", showlegend=False))
        fig.add_annotation(x=x, y=y - 0.14, text=f"<b>{NODE_LABELS[n]}</b><br>{value_txt}",
                            showarrow=False, font=dict(size=11, color="#C4CBDA"),
                            align="center")

    fig.update_xaxes(visible=False, range=[-0.1, 1.1])
    fig.update_yaxes(visible=False, range=[-0.28, 1.1])
    fig.update_layout(height=380, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=10, b=10))
    return fig

def build_alloc_bar(row):
    vals = [max(0.0, row[f"d_{s}"]) for s in SOURCES]
    colors = [PALETTE[s] for s in SOURCES]
    labels = [f"{ICONS[s]} {s.capitalize()}" for s in SOURCES]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h", marker_color=colors,
        text=[f"{v:.1f} MW" for v in vals], textposition="outside",
        cliponaxis=False))
    fig.add_vline(x=row["demand"], line=dict(color="#F2F4F8", dash="dot", width=2))
    fig.add_annotation(x=row["demand"], y=1.1, yref="paper",
                        text=f"Demand {row['demand']:.0f} MW",
                        showarrow=False, font=dict(size=11, color="#F2F4F8"))
    x_max = max(max(vals, default=1.0), row["demand"]) * 1.25
    fig.update_layout(height=380, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=80, t=40, b=10),
                       xaxis_title="MW", xaxis_range=[0, x_max])
    return fig

def build_dispatch_chart(d, h=None, compact=False):
    fig = go.Figure()
    seen = set()
    for s in SOURCES:
        grp = GROUP[s]
        kwargs = {}
        if grp not in seen:
            kwargs["legendgrouptitle_text"] = grp
            seen.add(grp)
        fig.add_trace(go.Scatter(
            x=d["t"], y=d[f"d_{s}"].clip(lower=0), name=s.capitalize(),
            stackgroup="one", mode="none", fillcolor=PALETTE[s],
            line=dict(width=0), legendgroup=grp, **kwargs))
    fig.add_trace(go.Scatter(x=d["t"], y=d["demand"], name="Demand",
                             mode="lines", line=dict(color="#F2F4F8", width=2.5, dash="dot"),
                             legendgroup="Demand", legendgrouptitle_text="Demand"))
    if h is not None:
        fig.add_vline(x=h, line=dict(color="#F2F4F8", width=2, dash="dash"))
        fig.add_annotation(x=h, y=1.05, yref="paper", text=f"Hour {h}",
                            showarrow=False, font=dict(color="#F2F4F8", size=11))
    fig.update_layout(
        height=230 if compact else 380, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=30, b=10),
        showlegend=not compact, legend=dict(orientation="h", y=-0.25),
        xaxis_title="Hour", yaxis_title=None if compact else "MW")
    return fig

def build_soc_chart(d, h):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["t"], y=d["soc"], mode="lines",
                              line=dict(color=PALETTE["battery"], width=3),
                              fill="tozeroy", fillcolor="rgba(155,140,255,.15)",
                              name="Battery level"))
    fig.add_vline(x=h, line=dict(color="#F2F4F8", width=2, dash="dash"))
    fig.update_layout(height=380, template="plotly_dark",
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       margin=dict(l=10, r=10, t=30, b=10),
                       xaxis_title="Hour", yaxis_title="MWh stored", showlegend=False)
    return fig

def classify_hour(row, cfg, engine_label):
    """Derive a bold lead-in tag, a plain-language detail sentence, an action
    summary, and any binding-constraint notes — all from recorded dispatch
    values (no claims about an agent's internal reasoning)."""
    demand = row["demand"]
    clean = row["d_solar"] + row["d_wind"]
    clean_share = clean / demand if demand else 0
    fossil = row["d_gas"] + row["d_coal"]
    batt = row["d_battery"]

    if row["unmet"] > 0.5:
        lead = "Shortfall"
        detail = (f"{row['unmet']:.1f} MWh of demand goes unmet — a near-blackout moment, even "
                  f"with gas at {row['d_gas']:.0f} MW and coal at {row['d_coal']:.0f} MW.")
    elif clean_share >= 0.6:
        lead = "Clean energy leading"
        detail = f"renewables cover {clean_share:.0%} of demand, so gas and coal stay backed off."
    elif clean_share <= 0.15:
        lead = "Fossil-heavy hour"
        detail = (f"renewables cover only {clean_share:.0%} of demand, so gas and coal ramp up "
                  f"to {fossil:.0f} MW combined to fill the gap.")
    else:
        lead = "Balanced mix"
        detail = f"clean ({clean_share:.0%}) and conventional power share the load."

    if batt > 1:
        detail += f" The battery discharges {batt:.0f} MW to help meet load."
    elif batt < -1:
        detail += f" Surplus power charges the battery at {abs(batt):.0f} MW for later."

    pace = cfg["emission_cap"] / HORIZON
    if row["emit"] > pace * 1.5:
        detail += " Emissions this hour are running above the daily carbon-budget pace."

    notes = []
    if row["d_gas"] >= CAPACITY["gas"] - 0.5:
        notes.append("gas at max output")
    if row["d_coal"] >= CAPACITY["coal"] - 0.5:
        notes.append("coal at max output")
    if abs(row["d_battery"]) >= CAPACITY["battery"] - 0.5:
        notes.append("battery at max rate")
    if row["soc"] <= 0.5:
        notes.append("battery empty")
    elif row["soc"] >= BATTERY_CAPACITY - 0.5:
        notes.append("battery full")
    if row["unmet"] > 0.5:
        notes.append("demand constraint binding")

    if engine_label == "SAC agent":
        action = (f"Clean sources covered {clean_share:.0%} of load "
                  f"(solar {row['d_solar']:.0f} + wind {row['d_wind']:.0f} MW); "
                  f"gas {row['d_gas']:.0f}, coal {row['d_coal']:.0f} MW, "
                  f"battery {row['d_battery']:+.0f} MW.")
    else:
        action = (f"LP dispatched solar {row['d_solar']:.0f}, wind {row['d_wind']:.0f}, "
                  f"gas {row['d_gas']:.0f}, coal {row['d_coal']:.0f}, "
                  f"battery {row['d_battery']:+.0f} MW to minimize cost + emissions.")

    return lead, detail, action, "; ".join(notes)

def hour_insight(row, cfg, engine_label):
    lead, detail, _, _ = classify_hour(row, cfg, engine_label)
    return f"**{lead}** — Hour {int(row['t']):02d}:00: {detail}"

@st.cache_data(show_spinner=False)
def build_decision_log(d, cfg, engine_label):
    rows = []
    emit_cum = d["emit"].cumsum()
    for i, r in d.iterrows():
        lead, detail, action, notes = classify_hour(r, cfg, engine_label)
        rows.append({
            "Hour": int(r["t"]),
            "Demand seen (MW)": round(r["demand"], 1),
            "Clean output (MW)": round(r["d_solar"] + r["d_wind"], 1),
            "Battery level (MWh)": round(r["soc"], 1),
            "Carbon budget left (tCO₂)": round(cfg["emission_cap"] - emit_cum.iloc[i], 1),
            "Lead": lead,
            "Detail": detail,
            "Action": action,
            "Constraints": notes,
        })
    return pd.DataFrame(rows)

# =============================================================================
# Mode: Overview
# =============================================================================
def render_overview():
    st.subheader("📊 Today at a glance")

    m1, m2, m3 = st.columns(3)
    m1.metric("Grid stability index (1 − unmet/demand)", f"{k['stability']:.2f}")
    m2.metric("Renewable utilization rate (clean share)", f"{k['renew']:.0%}")
    m3.metric("Emission reduction %", f"{emit_reduction_pct:.1f}%")
    st.caption(f"**Emission reduction %** is measured against a stated baseline — "
               f"{BASELINE_LABEL} — on the same demand scenario "
               f"({baseline_emit:.0f} tCO₂ baseline emissions vs. {k['emit']:.0f} tCO₂ today).")
    st.divider()

    donut_col, gauge_col = st.columns([2, 3])
    with donut_col:
        d1, d2 = st.columns(2)
        with d1:
            fig, color = build_donut(k["renew"], "Clean energy share", good=50, warn=25)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f'<div class="donut-legend">'
                f'<span class="status-dot" style="background:{color}"></span>Clean {k["renew"]:.0%}'
                f' &nbsp;·&nbsp; '
                f'<span class="status-dot" style="background:#3A4256"></span>Conventional {1 - k["renew"]:.0%}'
                f'</div>', unsafe_allow_html=True)
        with d2:
            fig, color = build_donut(k["stability"], "Grid reliability", good=98, warn=90)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f'<div class="donut-legend">'
                f'<span class="status-dot" style="background:{color}"></span>Served {k["stability"]:.0%}'
                f' &nbsp;·&nbsp; '
                f'<span class="status-dot" style="background:#3A4256"></span>Unmet {1 - k["stability"]:.0%}'
                f'</div>', unsafe_allow_html=True)
    with gauge_col:
        st.plotly_chart(build_gauges(k, cost_score), use_container_width=True)
    st.caption(f"Showing the **{live_src}**'s full-day result for the current scenario. "
               "Move the sidebar sliders to see this trade-off shift.")

    st.divider()
    st.markdown("##### Full-day dispatch at a glance")
    h_now = st.session_state.get("hour", 0)
    st.plotly_chart(build_dispatch_chart(live_df, h=h_now, compact=True), use_container_width=True)
    st.caption("Open **Live Dispatch** for hour-by-hour playback and the grid schematic, "
               "or **Decision Log** to inspect what the engine saw and did each hour.")

# =============================================================================
# Mode: Live Dispatch
# =============================================================================
def render_live_dispatch():
    st.subheader("⏱️ Live playback — watch a day unfold")
    st.markdown(
        '<span class="legend-pill" style="background:#1E2A22;color:#5BD99C;">● Renewable (solar, wind)</span>'
        '<span class="legend-pill" style="background:#2A1E1E;color:#E8743B;">● Conventional (gas, coal)</span>'
        '<span class="legend-pill" style="background:#221E2A;color:#9B8CFF;">● Battery storage</span>',
        unsafe_allow_html=True)

    if "hour" not in st.session_state:
        st.session_state.hour = 0
    if "playing" not in st.session_state:
        st.session_state.playing = False

    was_playing = st.session_state.playing

    c_play, c_reset, c_speed, c_scrub = st.columns([1, 1, 1, 4])
    with c_play:
        label = "⏸ Pause" if st.session_state.playing else "▶ Play"
        if st.button(label, use_container_width=True, key="play_pause_btn"):
            st.session_state.playing = not st.session_state.playing
    with c_reset:
        if st.button("⏮ Reset", use_container_width=True):
            st.session_state.hour = 0
            st.session_state.hour_scrub = 0
            st.session_state.playing = False
    with c_speed:
        speed = st.select_slider("Speed", options=list(SPEED_TO_SEC.keys()), value="1x")
    with c_scrub:
        if not st.session_state.playing:
            # Sync the scrub slider to the autoplay position the moment playback
            # stops (or on first render); otherwise leave it under user control.
            if was_playing or "hour_scrub" not in st.session_state:
                st.session_state.hour_scrub = st.session_state.hour
            st.session_state.hour = st.slider("Scrub to an hour", 0, HORIZON - 1,
                                               key="hour_scrub")
        else:
            st.markdown(f"**Playing…** hour {st.session_state.hour:02d}:00 "
                        f"(speed {speed})")

    interval = SPEED_TO_SEC[speed] if st.session_state.playing else None

    @st.fragment(run_every=interval)
    def live_panel():
        if st.session_state.playing:
            st.session_state.hour = (st.session_state.hour + 1) % HORIZON
        h = st.session_state.hour
        row = live_df.iloc[h]

        st.markdown(f"#### Hour {h:02d}:00 &nbsp;·&nbsp; {live_src}")

        emit_cum = live_df["emit"].cumsum()
        budget_left = cfg["emission_cap"] - emit_cum.iloc[h]
        cards = [
            ("💵", "Cost this hour", f"${row['cost']:,.0f}"),
            ("🌫️", "Emissions this hour", f"{row['emit']:.1f} tCO₂"),
            ("🎯", "Carbon budget left", f"{budget_left:,.0f} tCO₂"),
            ("🔋", "Battery level", f"{row['soc']:.0f}/{BATTERY_CAPACITY:.0f} MWh"),
            ("⚠️", "Unmet this hour", f"{row['unmet']:.1f} MWh"),
        ]
        cols = st.columns(5)
        for c, (icon, lbl, val) in zip(cols, cards):
            c.markdown(f'<div class="metric-card"><div class="metric-icon">{icon}</div>'
                        f'<div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div></div>',
                        unsafe_allow_html=True)

        schem_col, alloc_col = st.columns([3, 2])
        schem_col.plotly_chart(build_schematic(row), use_container_width=True, key=f"schem_{live_src}_{h}")
        alloc_col.plotly_chart(build_alloc_bar(row), use_container_width=True, key=f"alloc_{live_src}_{h}")

        st.markdown(f'<div class="insight-box">{hour_insight(row, cfg, live_src)}</div>',
                    unsafe_allow_html=True)

        chart_col, soc_col = st.columns([3, 2])
        chart_col.plotly_chart(build_dispatch_chart(live_df, h), use_container_width=True,
                                key=f"dispatch_{live_src}_{h}")
        soc_col.plotly_chart(build_soc_chart(live_df, h), use_container_width=True,
                              key=f"soc_{live_src}_{h}")

    live_panel()

# =============================================================================
# Mode: Engine Comparison
# =============================================================================
def render_engine_comparison():
    st.subheader("🆚 RL agent vs. constraint solver — full day")
    if agent_df is None:
        st.info("The SAC agent isn't active. Switch the sidebar **Dispatch engine** "
                "control to **\"Both — compare with SAC agent\"** to see this view.")
        return

    k_solver, k_agent = kpis(df), kpis(agent_df)

    cols = st.columns(2)
    for c, (label, kk) in zip(cols, [("Constraint solver", k_solver), ("SAC agent", k_agent)]):
        with c:
            st.markdown(f"##### {label}")
            d1, d2 = st.columns(2)
            with d1:
                fig, color = build_donut(kk["renew"], "Clean energy share", good=50, warn=25)
                st.plotly_chart(fig, use_container_width=True, key=f"cmp_clean_{label}")
            with d2:
                fig, color = build_donut(kk["stability"], "Grid reliability", good=98, warn=90)
                st.plotly_chart(fig, use_container_width=True, key=f"cmp_reli_{label}")

    comp = pd.DataFrame({
        "Metric": ["Cost ($)", "Emissions (tCO₂)", "Clean energy share", "Unmet (MWh)"],
        "Constraint solver": [f"{k_solver['cost']:,.0f}", f"{k_solver['emit']:.0f}",
                              f"{k_solver['renew']:.0%}", f"{k_solver['unmet']:.0f}"],
        "SAC agent (+projection)": [f"{k_agent['cost']:,.0f}", f"{k_agent['emit']:.0f}",
                                    f"{k_agent['renew']:.0%}", f"{k_agent['unmet']:.0f}"],
    })
    st.dataframe(comp, hide_index=True, use_container_width=True)
    st.caption("The LP is provably optimal for the modelled problem; the SAC agent "
               "learns a policy and is kept feasible by the projection layer, "
               "trading optimality for adaptability to unmodelled dynamics.")

    st.divider()
    st.markdown("##### Dispatch pattern comparison")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(f"**Constraint solver** — clean share {k_solver['renew']:.0%}, "
                     f"unmet {k_solver['unmet']:.1f} MWh")
        st.plotly_chart(build_dispatch_chart(df), use_container_width=True,
                        key="cmp_dispatch_solver")
    with d2:
        st.markdown(f"**SAC agent** — clean share {k_agent['renew']:.0%}, "
                     f"unmet {k_agent['unmet']:.1f} MWh")
        st.plotly_chart(build_dispatch_chart(agent_df), use_container_width=True,
                        key="cmp_dispatch_agent")
    st.caption("Same scenario, two dispatch policies — compare how each engine "
               "allocates renewable, fossil and battery output across the day.")

# =============================================================================
# Mode: Trade-offs
# =============================================================================
def render_tradeoffs():
    st.subheader("📊 Today's three-way trade-off")
    st.plotly_chart(build_gauges(k, cost_score), use_container_width=True)
    st.caption(f"Showing the **{live_src}**'s full-day result. "
               "🟢 healthy · 🟡 strained · 🔴 critical — move the sidebar sliders "
               "to see cost, emissions and reliability shift against each other.")

    st.divider()
    st.markdown("##### Headline sensitivity: emission cap vs. cost & unmet demand")
    st.caption("The emission cap is a **hard constraint** — sweeping it from tight to "
               "loose (the LP solver re-run at each level) shows the cost of stricter "
               "caps, and the point at which the cap starts forcing unmet demand.")

    sweep = run_cap_sweep(cfg, CAP_SWEEP_RANGE)
    fig_cap = make_subplots(specs=[[{"secondary_y": True}]])
    fig_cap.add_trace(go.Scatter(
        x=sweep["cap"], y=sweep["cost"], name="Total cost ($)",
        mode="lines+markers", line=dict(color="#F6C445", width=3)), secondary_y=False)
    fig_cap.add_trace(go.Scatter(
        x=sweep["cap"], y=sweep["unmet"], name="Unmet demand (MWh)",
        mode="lines+markers", line=dict(color="#F0626B", width=3, dash="dot")), secondary_y=True)
    fig_cap.add_vline(x=cfg["emission_cap"], line=dict(color="#F2F4F8", width=2, dash="dash"))
    fig_cap.add_annotation(x=cfg["emission_cap"], y=1.05, yref="paper",
                            text="Current cap", showarrow=False,
                            font=dict(color="#F2F4F8", size=11))
    fig_cap.update_layout(height=360, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=30, b=10),
                          legend=dict(orientation="h", y=-0.2),
                          xaxis_title="Daily carbon budget / emission cap (tCO₂) — tight → loose")
    fig_cap.update_yaxes(title_text="Total cost ($)", secondary_y=False)
    fig_cap.update_yaxes(title_text="Unmet demand (MWh)", secondary_y=True)
    st.plotly_chart(fig_cap, use_container_width=True)
    st.caption("Tightening the cap (moving left) pushes the solver toward costlier "
               "low-emission sources, raising total cost — and once the cap is tight "
               "enough, demand can no longer be fully served. The dashed line marks "
               "the cap chosen in the sidebar.")

    st.divider()
    st.markdown("##### Additional view: cost vs. emissions across carbon prices")

    @st.cache_data(show_spinner=False)
    def pareto(base_cfg):
        rows = []
        for p in [0, 10, 20, 40, 80, 120, 200]:
            c = {**base_cfg, "emission_penalty": p}
            d = pd.DataFrame(run_episode_solver(GridEnv(c)))
            rows.append({"penalty": p, "cost": d["cost"].sum(), "emit": d["emit"].sum()})
        return pd.DataFrame(rows)

    pf = pareto(cfg)
    figp = go.Figure()
    figp.add_trace(go.Scatter(x=pf["emit"], y=pf["cost"], mode="lines+markers+text",
                              text=[f"${p}" for p in pf["penalty"]], textposition="top center",
                              line=dict(color="#4FC3D9", width=2),
                              marker=dict(size=10, color="#F6C445")))
    figp.update_layout(height=340, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=10, b=10),
                       xaxis_title="Total emissions (tCO₂)", yaxis_title="Total cost ($)")
    st.plotly_chart(figp, use_container_width=True)
    st.caption("Each point is a different carbon price (the LP solver, run once per price). "
               "Moving down-left is better on both axes but infeasible past the "
               "renewable/storage limit — the classic Pareto frontier.")

# =============================================================================
# Mode: Decision Log
# =============================================================================
def render_decision_log():
    st.subheader("📋 Decision log / engine inspector")
    st.caption(f"What the **{live_src}** saw and did, hour by hour. "
               "'Action' describes recorded dispatch values — for the SAC agent this "
               "is its observable behaviour, not its internal computation.")

    log_df = build_decision_log(live_df, cfg, live_src)
    default_hour = st.session_state.get("hour", 0)

    list_col, detail_col = st.columns([2, 3])
    with list_col:
        display_cols = ["Hour", "Demand seen (MW)", "Clean output (MW)",
                         "Battery level (MWh)", "Carbon budget left (tCO₂)", "Lead"]
        event = st.dataframe(
            log_df[display_cols], hide_index=True, use_container_width=True,
            height=520, on_select="rerun", selection_mode="single-row", key="dlog_select")
        sel_rows = event.selection.rows if event and event.selection else []
        sel_hour = int(log_df.iloc[sel_rows[0]]["Hour"]) if sel_rows else default_hour

    with detail_col:
        r = log_df.iloc[sel_hour]
        live_row = live_df.iloc[sel_hour]
        st.markdown(f"### Hour {sel_hour:02d}:00")
        st.markdown(f'<div class="insight-box"><b>{r["Lead"]}</b> — {r["Detail"]}</div>',
                    unsafe_allow_html=True)

        st.markdown("**State the engine saw**")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Demand", f"{r['Demand seen (MW)']:.0f} MW")
        s2.metric("Clean output", f"{r['Clean output (MW)']:.0f} MW")
        s3.metric("Battery level", f"{r['Battery level (MWh)']:.0f} MWh")
        s4.metric("Carbon budget left", f"{r['Carbon budget left (tCO₂)']:.0f} tCO₂")

        st.markdown("**Action taken (MW per source)**")
        st.plotly_chart(build_alloc_bar(live_row), use_container_width=True,
                         key=f"dlog_alloc_{live_src}_{sel_hour}")

        st.markdown(f"**Dispatch summary** — {r['Action']}")
        if r["Constraints"]:
            st.caption(f"Binding constraints this hour: {r['Constraints']}")
        else:
            st.caption("No capacity or demand constraints were binding this hour.")

# ----------------------------- mode dispatch --------------------------------- #
if mode == "Overview":
    render_overview()
elif mode == "Live Dispatch":
    render_live_dispatch()
elif mode == "Engine Comparison":
    render_engine_comparison()
elif mode == "Trade-offs":
    render_tradeoffs()
elif mode == "Decision Log":
    render_decision_log()
