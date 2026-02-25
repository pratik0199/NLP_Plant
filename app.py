import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ==============================
# DARK THEME
# ==============================
st.markdown("""
<style>
.stApp { background-color: #0e1117; }

h1, h2, h3, h4 {
    color: #ffffff !important;
    font-weight: 700;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

label { color: white !important; }

div[data-baseweb="select"] > div {
    background-color: #1f2937 !important;
    color: white !important;
}

ul[role="listbox"] li {
    background-color: #1f2937 !important;
    color: white !important;
}

ul[role="listbox"] li:hover {
    background-color: #374151 !important;
}

li[aria-selected="true"] {
    background-color: #2563eb !important;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD CSV
# ==============================
file_path = "processed_pump_data.csv"
df = pd.read_csv(file_path)

# ==============================
# 🔥 FIX: CREATE MONTH COLUMN
# ==============================
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

df["month"] = df["date"].dt.strftime("%B-%Y")

# ==============================
# FAILURE TYPES
# ==============================
FAILURE_TYPES = [
    "Seal_Issue",
    "Low_Level",
    "Pressure_Issue",
    "Trip_Issue",
    "Flow_Issue"
]

COLORS = {
    "Seal_Issue": "#00E5FF",
    "Low_Level": "#FFEA00",
    "Pressure_Issue": "#00FF85",
    "Trip_Issue": "#FF1744",
    "Flow_Issue": "#D500F9"
}

# ==============================
# SIDEBAR FILTER
# ==============================
st.sidebar.title("📊 Filters")

selected_month = st.sidebar.selectbox(
    "Select Month",
    sorted(df["month"].unique())
)


filtered_df = df[
    (df["month"] == selected_month) &
    (df["shift"].isin(selected_shift))
]

# ==============================
# TITLE
# ==============================
st.title("⚙️ Pump Failure Dashboard")
st.markdown(f"### 📅 Selected Month: `{selected_month}`")

# ==============================
# LAYOUT
# ==============================
col1, col2 = st.columns(2)

# -------- KPI 1 --------
with col1:
    st.subheader("Top 7 Failure-Prone Pumps")

    top_pumps = (
        filtered_df.groupby("pump")["Total_Failure"]
        .sum()
        .sort_values(ascending=False)
        .head(7)
    )

    fig1, ax1 = plt.subplots()
    top_pumps.plot(kind="bar", ax=ax1, color="#00E5FF", edgecolor="white")

    ax1.tick_params(colors='white')
    ax1.set_facecolor("#0e1117")
    fig1.patch.set_facecolor("#0e1117")

    st.pyplot(fig1)

# -------- KPI 2 --------
with col2:
    st.subheader("Failure Distribution")

    failure_sum = filtered_df[FAILURE_TYPES].sum()

    fig2, ax2 = plt.subplots()

    ax2.pie(
        failure_sum,
        labels=failure_sum.index,
        autopct=lambda p: f'{p:.1f}%' if p > 5 else '',
        colors=[COLORS[k] for k in FAILURE_TYPES],
        textprops={'color': "white"}
    )

    fig2.patch.set_facecolor("#0e1117")
    st.pyplot(fig2)

# -------- KPI 3 & 4 --------
col3, col4 = st.columns(2)

with col3:
    st.subheader("Shift-wise Failures")

    monthly = filtered_df.groupby("shift")[FAILURE_TYPES].sum()

    fig3, ax3 = plt.subplots()

    monthly.plot(
        kind="bar",
        stacked=True,
        ax=ax3,
        color=[COLORS[c] for c in FAILURE_TYPES],
        edgecolor="white"
    )

    ax3.tick_params(colors='white')
    ax3.set_facecolor("#0e1117")
    fig3.patch.set_facecolor("#0e1117")

    st.pyplot(fig3)

with col4:
    st.subheader("Day vs Night Trend")

    monthly_all = df.groupby(["month", "shift"])[FAILURE_TYPES].sum().reset_index()
    months = sorted(df["month"].unique())

    day = monthly_all[monthly_all["shift"] == "Day"].set_index("month").reindex(months).fillna(0)
    night = monthly_all[monthly_all["shift"] == "Night"].set_index("month").reindex(months).fillna(0)

    x = np.arange(len(months))
    width = 0.25

    fig4, ax4 = plt.subplots(figsize=(10,5))

    bottom_d = np.zeros(len(months))
    bottom_n = np.zeros(len(months))

    for col in FAILURE_TYPES:
        ax4.bar(x - width, day[col], width, bottom=bottom_d, color=COLORS[col])
        ax4.bar(x + width, night[col], width, bottom=bottom_n, color=COLORS[col])

        bottom_d += day[col].values
        bottom_n += night[col].values

    ax4.set_xticks(x)
    ax4.set_xticklabels(months, rotation=30, color="white")

    ax4.tick_params(colors='white')
    ax4.set_facecolor("#0e1117")
    fig4.patch.set_facecolor("#0e1117")

    st.pyplot(fig4)

