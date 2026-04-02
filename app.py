import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

st.set_page_config(layout="wide")

# ─────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("classified_output.csv")   # <-- updated filename

df = load_data()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df[df["pump"].notna()]

df["month_dt"] = df["date"].dt.to_period("M").dt.to_timestamp()
df["month"] = df["month_dt"].dt.strftime("%B-%Y")


# ─────────────────────────────────────────────────
# UPDATED FAILURE TYPES (NEW COLUMNS)
# ─────────────────────────────────────────────────

FAILURE_TYPES = [
    "seal_failure",
    "low_level",
    "pump_swap",
    "startup",
    "shutdown",
    "trip_fault",
    "low_pressure",
    "oil_lubrication",
    "steam_issue",
    "strainer_clean",
    "maintenance_pm",
    "vibration"
]


# ─────────────────────────────────────────────────
# COLORS (Auto expand friendly)
# ─────────────────────────────────────────────────

COLORS = {
    "seal_failure": "#2563eb",
    "low_level": "#d97706",
    "pump_swap": "#059669",
    "startup": "#dc2626",
    "shutdown": "#7c3aed",
    "trip_fault": "#0ea5e9",
    "low_pressure": "#f59e0b",
    "oil_lubrication": "#10b981",
    "steam_issue": "#ef4444",
    "strainer_clean": "#6366f1",
    "maintenance_pm": "#14b8a6",
    "vibration": "#8b5cf6"
}


SHIFT_COLORS = {
    "Day": "#0d9488",
    "Night": "#475569"
}


PLOT_BG = "#f8fafc"
GRID_CLR = "#cbd5e1"
BAR_FIGSIZE = (6.0, 3.0)
DONUT_FIGSIZE = (6.0, 3.0)
BAR_WIDTH = 0.50


# ─────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────

def failure_patches():
    return [mpatches.Patch(color=COLORS[f], label=f.replace("_", " "))
            for f in FAILURE_TYPES]


def apply_black_border(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0)
        spine.set_color("black")


def style_bar(ax):
    ax.set_facecolor(PLOT_BG)
    ax.grid(axis='y', linestyle='--', linewidth=0.7,
            alpha=0.6, color=GRID_CLR)
    ax.grid(axis='x', visible=False)
    ax.tick_params(colors="black", labelsize=7)
    apply_black_border(ax)


def get_shift(grouped, name, months_labels):
    try:
        return grouped.xs(name, level=1).reindex(months_labels).fillna(0)
    except KeyError:
        return pd.DataFrame(0,
                            index=months_labels,
                            columns=FAILURE_TYPES)


# ─────────────────────────────────────────────────
# DONUT PLOT
# ─────────────────────────────────────────────────

def plot_donut_compartment(title, data):

    data = data.reindex(FAILURE_TYPES).fillna(0)
    total = data.sum()

    if total == 0:
        st.info("No failure data available")
        return

    fig, ax = plt.subplots(figsize=DONUT_FIGSIZE)

    ax.pie(
        data,
        labels=[i.replace("_", " ") for i in data.index],
        colors=[COLORS[k] for k in data.index],
        autopct='%1.1f%%',
        startangle=90
    )

    ax.set_title(title, fontsize=10)
    st.pyplot(fig)


# ─────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────

st.title("Pump Reliability Dashboard")


# ─────────────────────────────────────────────────
# NAVIGATION
# ─────────────────────────────────────────────────

page = st.sidebar.radio(
    "Select View",
    ["Overall", "Monthly", "Pump"]
)


# ══════════════════════════════════════════════════
# OVERALL
# ══════════════════════════════════════════════════

if page == "Overall":

    months_sorted = sorted(df["month_dt"].dropna().unique())
    months_labels = [pd.to_datetime(m).strftime("%B-%Y")
                     for m in months_sorted]

    grouped = df.groupby("month")[FAILURE_TYPES].sum()

    fig, ax = plt.subplots(figsize=(10, 4))

    bottom = np.zeros(len(grouped))

    for ft in FAILURE_TYPES:
        ax.bar(grouped.index,
               grouped[ft],
               bottom=bottom,
               label=ft.replace("_", " "))
        bottom += grouped[ft]

    ax.legend(fontsize=7)
    plt.xticks(rotation=30)

    st.pyplot(fig)

    plot_donut_compartment(
        "Overall Failure Distribution",
        df[FAILURE_TYPES].sum()
    )


# ══════════════════════════════════════════════════
# MONTHLY
# ══════════════════════════════════════════════════

elif page == "Monthly":

    month_list = df["month"].dropna().unique()

    selected_month = st.selectbox(
        "Select Month",
        month_list
    )

    filtered_df = df[df["month"] == selected_month]

    grouped = filtered_df.groupby("pump")[FAILURE_TYPES].sum()

    grouped["Total"] = grouped.sum(axis=1)

    top7 = grouped.sort_values(
        "Total",
        ascending=False
    ).head(7)

    fig, ax = plt.subplots(figsize=(10, 4))

    bottom = np.zeros(len(top7))

    for ft in FAILURE_TYPES:
        ax.bar(
            top7.index,
            top7[ft],
            bottom=bottom,
            label=ft.replace("_", " ")
        )
        bottom += top7[ft]

    ax.legend(fontsize=7)

    plt.xticks(rotation=30)

    st.pyplot(fig)

    plot_donut_compartment(
        f"{selected_month} Failure Distribution",
        filtered_df[FAILURE_TYPES].sum()
    )


# ══════════════════════════════════════════════════
# PUMP
# ══════════════════════════════════════════════════

elif page == "Pump":

    pumps = (
        df.groupby("pump")[FAILURE_TYPES]
        .sum()
        .sum(axis=1)
        .sort_values(ascending=False)
        .index
    )

    selected_pump = st.selectbox(
        "Select Pump",
        pumps
    )

    pump_df = df[df["pump"] == selected_pump]

    st.subheader("Equipment History")

    history = pump_df[
        ["date", "shift", "events"]
    ].sort_values("date")

    st.dataframe(history, height=300)

    plot_donut_compartment(
        f"{selected_pump} Failure Distribution",
        pump_df[FAILURE_TYPES].sum()
    )
