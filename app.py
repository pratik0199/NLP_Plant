import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

st.set_page_config(layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 0.8rem !important;
    padding-bottom: 0rem !important;
    padding-left: 1.2rem !important;
    padding-right: 1.2rem !important;
    max-width: 100% !important;
}
html, body, [class*="css"] {
    font-family: "Times New Roman", serif;
}
div.stButton > button {
    width: 100%;
    border-radius: 4px;
    height: 36px;
    font-weight: 600;
    font-size: 13px;
    background-color: #1e293b;
    color: #f1f5f9;
    border: 1px solid #334155;
}
div.stButton > button:hover { background-color: #0f172a; color: white; }
.plot-box {
    border: 2px solid black;
    border-radius: 6px;
    padding: 6px 6px 2px 6px;
    background: #f8fafc;
    margin-bottom: 6px;
}
.plot-title {
    font-size: 13px;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 2px;
    font-family: "Times New Roman", serif;
}
div[data-baseweb="select"] { max-width: 220px; }
header { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("processed_pump_data_final.csv")

df = load_data()
df["date"]     = pd.to_datetime(df["date"], errors="coerce")
df             = df[df["pump"].notna()]
df["month_dt"] = df["date"].dt.to_period("M").dt.to_timestamp()
df["month"]    = df["month_dt"].dt.strftime("%B-%Y")

FAILURE_TYPES = ["Seal_Issue", "Low_Level", "Pressure_Issue", "Trip_Issue", "Flow_Issue"]

COLORS = {
    "Seal_Issue":     "#2563eb",
    "Low_Level":      "#d97706",
    "Pressure_Issue": "#059669",
    "Trip_Issue":     "#dc2626",
    "Flow_Issue":     "#7c3aed",
}
SHIFT_COLORS = {"Day": "#0d9488", "Night": "#475569"}

PLOT_BG  = "#f8fafc"
GRID_CLR = "#cbd5e1"

BAR_FIGSIZE   = (6.3, 3.8)
DONUT_FIGSIZE = (6.3, 3.8)
BAR_SM        = (5.6, 3.2)   # monthly & pump bar
DONUT_SM      = (5.6, 3.2)   # monthly & pump donut


# ─────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────
def failure_patches():
    return [mpatches.Patch(color=COLORS[f], label=f) for f in FAILURE_TYPES]


def apply_black_border(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0)
        spine.set_color("black")


def style_bar(ax):
    ax.set_facecolor(PLOT_BG)
    # lighter grid: thinner line, lower alpha, softer colour
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6, color=GRID_CLR)
    ax.grid(axis='x', visible=False)
    ax.tick_params(colors="black", labelsize=8)
    apply_black_border(ax)


def style_donut(ax):
    ax.set_facecolor(PLOT_BG)
    apply_black_border(ax)


def get_shift(grouped, name, months_labels):
    try:
        return grouped.xs(name, level=1).reindex(months_labels).fillna(0)
    except KeyError:
        return pd.DataFrame(0, index=months_labels, columns=FAILURE_TYPES)


def plot_compartment(title, draw_fn, figsize=BAR_FIGSIZE):
    st.markdown(f'<div class="plot-box"><div class="plot-title">{title}</div>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=figsize, facecolor=PLOT_BG)
    fig.patch.set_facecolor(PLOT_BG)
    draw_fn(ax)
    plt.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────
# DONUT COMPARTMENT
# ─────────────────────────────────────────────────
def plot_donut_compartment(title, data, figsize=DONUT_FIGSIZE):
    data  = data.reindex(FAILURE_TYPES).fillna(0)
    total = data.sum()

    fig = plt.figure(figsize=figsize, facecolor=PLOT_BG)
    fig.patch.set_facecolor(PLOT_BG)

    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[0.65, 0.35],
        left=0.02, right=0.98,
        top=0.97, bottom=0.06,
        wspace=0.0
    )
    ax_pie = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])

    # ── Donut ──────────────────────────────────────────────────
    ax_pie.set_facecolor(PLOT_BG)
    for sp in ax_pie.spines.values():
        sp.set_visible(False)

    if total == 0:
        ax_pie.text(0.5, 0.5, "No data", ha="center", va="center",
                    fontsize=9, transform=ax_pie.transAxes)
    else:
        def autopct(pct):
            return f"{pct:.1f}%" if pct >= 5 else ""

        _, _, autotexts = ax_pie.pie(
            data,
            colors=[COLORS[k] for k in data.index],
            startangle=90,
            wedgeprops=dict(width=0.40, edgecolor="white", linewidth=0.8),
            autopct=autopct,
            pctdistance=0.76,
            radius=0.88,
            center=(0, 0.08),
        )
        for at in autotexts:
            at.set_fontsize(9)        # ← increased from 7
            at.set_color("white")
            at.set_fontweight("bold")

        ax_pie.text(0, 0.08, f"Total\n{int(total)}",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color="#1e293b")  # ← increased from 7

    # ── Legend ─────────────────────────────────────────────────
    ax_leg.set_facecolor(PLOT_BG)
    ax_leg.axis("off")
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)

    row_h = 0.13
    pad_t = 0.94
    x0    = 0.04
    hw    = 0.18
    hh    = 0.055

    for i, ft in enumerate(FAILURE_TYPES):
        yc = pad_t - i * row_h
        ax_leg.add_patch(mpatches.FancyArrow(0, 0, 0, 0))
        ax_leg.add_patch(plt.Rectangle(
            (x0, yc - hh / 2), hw, hh,
            transform=ax_leg.transAxes,
            color=COLORS[ft],
            clip_on=False, zorder=10
        ))
        label = ft.replace("_", " ")
        ax_leg.text(
            x0 + hw + 0.06, yc, label,
            transform=ax_leg.transAxes,
            fontsize=8.5, va="center", color="#1e293b",
            clip_on=False
        )

    st.markdown(f'<div class="plot-box"><div class="plot-title">{title}</div>',
                unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────
# TITLE + NAV
# ─────────────────────────────────────────────────
st.markdown(
    "<h2 style='text-align:center; color:#1e293b; margin-bottom:0.5rem;'>"
    "Pump Reliability Dashboard</h2>",
    unsafe_allow_html=True,
)

_, b1, b2, b3, _ = st.columns([2, 1, 1, 1, 2])

if "page" not in st.session_state:
    st.session_state.page = "Overall"
if b1.button("Overall Analysis"):  st.session_state.page = "Overall"
if b2.button("Monthly Analysis"):  st.session_state.page = "Monthly"
if b3.button("Pump Analysis"):     st.session_state.page = "Pump"

page = st.session_state.page

st.markdown(
    "<hr style='margin:0.3rem 0 0.6rem 0; border-color:#94a3b8;'>",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════
# OVERALL ANALYSIS
# ══════════════════════════════════════════════════
if page == "Overall":

    shift_toggle = st.toggle("Shift × Failure Bifurcation", value=False)

    months_sorted = sorted(df["month_dt"].dropna().unique())
    months_labels = [pd.to_datetime(m).strftime("%B-%Y") for m in months_sorted]
    x = np.arange(len(months_labels))

    col1, col2 = st.columns(2, gap="small")

    with col1:
        if not shift_toggle:
            def draw_single_dn(ax):
                grouped = df.groupby(["month", "shift"])[FAILURE_TYPES].sum()
                day   = get_shift(grouped, "Day",   months_labels)
                night = get_shift(grouped, "Night", months_labels)

                day_tot   = day.sum(axis=1).values
                night_tot = night.sum(axis=1).values

                ax.bar(x, day_tot, width=0.55,
                       color=SHIFT_COLORS["Day"], edgecolor="white",
                       linewidth=0.6, label="Day")
                ax.bar(x, night_tot, width=0.55, bottom=day_tot,
                       color=SHIFT_COLORS["Night"], edgecolor="white",
                       linewidth=0.6, hatch="//", label="Night")

                ax.plot(x, day_tot, "k_", markersize=14, markeredgewidth=1.4, zorder=5)

                for i in range(len(x)):
                    if day_tot[i] > 0:
                        ax.text(x[i], day_tot[i] / 2, "D",
                                ha="center", va="center",
                                fontsize=7, fontweight="bold", color="white", zorder=6)
                    if night_tot[i] > 0:
                        ax.text(x[i], day_tot[i] + night_tot[i] / 2, "N",
                                ha="center", va="center",
                                fontsize=7, fontweight="bold", color="white", zorder=6)

                ax.set_xticks(x)
                ax.set_xticklabels(months_labels, rotation=30, ha='right', fontsize=7.5)
                ax.set_ylabel("Failure Count", fontsize=8, color="black")
                ax.legend(fontsize=8, loc='upper left',
                          framealpha=0.9, edgecolor="black")
                style_bar(ax)

            plot_compartment("Monthly Failure Trend  (Day | Night)", draw_single_dn)

        else:
            def draw_dual_failure(ax):
                grouped = df.groupby(["month", "shift"])[FAILURE_TYPES].sum()
                day   = get_shift(grouped, "Day",   months_labels)
                night = get_shift(grouped, "Night", months_labels)

                w = 0.35
                bottom_d = np.zeros(len(months_labels))
                bottom_n = np.zeros(len(months_labels))

                for ft in FAILURE_TYPES:
                    ax.bar(x - w/2, day[ft].values, width=w,
                           bottom=bottom_d, color=COLORS[ft],
                           edgecolor="white", linewidth=0.4)
                    ax.bar(x + w/2, night[ft].values, width=w,
                           bottom=bottom_n, color=COLORS[ft],
                           edgecolor="white", linewidth=0.4)
                    bottom_d += day[ft].values
                    bottom_n += night[ft].values

                for i in range(len(x)):
                    if bottom_d[i] > 0:
                        ax.text(x[i] - w/2, bottom_d[i] + 0.3, "D",
                                ha="center", va="bottom",
                                fontsize=7, fontweight="bold", color="black", zorder=6)
                    if bottom_n[i] > 0:
                        ax.text(x[i] + w/2, bottom_n[i] + 0.3, "N",
                                ha="center", va="bottom",
                                fontsize=7, fontweight="bold", color="black", zorder=6)

                ax.set_xticks(x)
                ax.set_xticklabels(months_labels, rotation=30, ha='right', fontsize=7.5)
                ax.set_ylabel("Failure Count", fontsize=8, color="black")
                ax.legend(handles=failure_patches(), fontsize=7,
                          loc='upper left', framealpha=0.9, edgecolor="black")
                style_bar(ax)

            plot_compartment("Monthly Failure Trend  (Shift × Failure Type)", draw_dual_failure)

    with col2:
        plot_donut_compartment("Overall Failure Distribution",
                               df[FAILURE_TYPES].sum())


# ══════════════════════════════════════════════════
# MONTHLY ANALYSIS
# ══════════════════════════════════════════════════
elif page == "Monthly":

    colA, _ = st.columns([1, 5])
    with colA:
        month_list     = df.sort_values("month_dt")["month"].dropna().unique()
        selected_month = st.selectbox("Select Month", month_list)

    filtered_df = df[df["month"] == selected_month]
    grouped     = filtered_df.groupby("pump")[FAILURE_TYPES].sum()
    grouped["Total"] = grouped.sum(axis=1)
    top5 = grouped.sort_values("Total", ascending=False).head(5)[FAILURE_TYPES]

    col1, col2 = st.columns(2, gap="small")

    with col1:
        def draw_top5(ax):
            bottom = np.zeros(len(top5))
            for ft in FAILURE_TYPES:
                ax.bar(top5.index, top5[ft], bottom=bottom, width=0.55,
                       color=COLORS[ft], edgecolor="white", linewidth=0.5)
                bottom += top5[ft].values
            ax.set_xticklabels(top5.index, rotation=25, ha='right', fontsize=8)
            ax.set_ylabel("Failure Count", fontsize=8, color="black")
            ax.legend(handles=failure_patches(), fontsize=7,
                      loc='upper left', framealpha=0.9, edgecolor="black")
            style_bar(ax)

        plot_compartment(
            f"Top 5 Failure-Prone Pumps  —  {selected_month}",
            draw_top5, figsize=BAR_SM
        )

    with col2:
        plot_donut_compartment(
            f"Failure Distribution  —  {selected_month}",
            filtered_df[FAILURE_TYPES].sum(), figsize=DONUT_SM
        )


# ══════════════════════════════════════════════════
# PUMP ANALYSIS
# ══════════════════════════════════════════════════
elif page == "Pump":

    colA, _ = st.columns([1, 5])
    with colA:
        pumps_by_failures = (df.groupby("pump")[FAILURE_TYPES].sum().sum(axis=1).sort_values(ascending=False).index.tolist())
        selected_pump = st.selectbox("Select Pump", pumps_by_failures)

    pump_df = df[df["pump"] == selected_pump]
    col1, col2 = st.columns(2, gap="small")

    with col1:
        st.markdown(
            '<div class="plot-box"><div class="plot-title">Equipment History</div>',
            unsafe_allow_html=True
        )
        history = pump_df[["date", "shift", "event"]].sort_values("date")
        st.dataframe(history, use_container_width=True, height=280)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        plot_donut_compartment(
            f"Failure Distribution  —  {selected_pump}",
            pump_df[FAILURE_TYPES].sum(), figsize=DONUT_SM
        )
