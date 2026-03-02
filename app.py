import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(layout="wide")

# ==============================
# UI + FONT
# ==============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Times+New+Roman&display=swap');

html, body, [class*="css"]  {
    font-family: "Times New Roman", serif;
}

/* Main background */
.stApp { background-color: #0e1117; }

/* Titles */
h1 {
    color: white !important;
    text-align: center;
    font-weight: 800;
}

h2, h3 {
    color: #e5e7eb !important;
    font-weight: 600;
}

/* Sidebar WHITE */
section[data-testid="stSidebar"] {
    background-color: white !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: black !important;
}

/* Dropdown */
div[data-baseweb="select"] > div {
    background-color: #1f2937 !important;
    color: white !important;
}

/* Labels */
label { color: white !important; }

</style>
""", unsafe_allow_html=True)

st.cache_data.clear()

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    return pd.read_csv("processed_pump_data_final.csv")

df = load_data()

# ==============================
# PREPROCESS
# ==============================
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df[df["pump"].notna()]

df["month_dt"] = df["date"].dt.to_period("M").dt.to_timestamp()
df["month"] = df["month_dt"].dt.strftime("%B-%Y")

# ==============================
# FAILURE CONFIG
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
# SIDEBAR NAV
# ==============================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Overall Analysis", "Monthly Analysis", "Pump Analysis"]
)

# ==============================
# PAGE 1: OVERALL
# ==============================
if page == "Overall Analysis":

    st.title("Overall Plant Insights")

    col1, col2 = st.columns([2, 1])  # 🔥 bigger chart

    with col1:
        st.subheader("Month-Shift Failure Distribution")

        grouped = df.groupby(["month", "shift"])[FAILURE_TYPES].sum().reset_index()

        months_sorted = sorted(df["month_dt"].dropna().unique())
        months_labels = [pd.to_datetime(m).strftime("%B-%Y") for m in months_sorted]

        day = grouped[grouped["shift"] == "Day"].set_index("month").reindex(months_labels).fillna(0)
        night = grouped[grouped["shift"] == "Night"].set_index("month").reindex(months_labels).fillna(0)

        x = np.arange(len(months_labels))
        width = 0.35

        plt.rcParams["font.family"] = "Times New Roman"

        fig, ax = plt.subplots(figsize=(10,5))  # 🔥 bigger

        bottom_d = np.zeros(len(months_labels))
        bottom_n = np.zeros(len(months_labels))

        for col in FAILURE_TYPES:
            ax.bar(x - width/2, day[col], width, bottom=bottom_d, color=COLORS[col])
            ax.bar(x + width/2, night[col], width, bottom=bottom_n, color=COLORS[col])

            bottom_d += day[col].values
            bottom_n += night[col].values

        ax.set_xticks(x)
        ax.set_xticklabels(months_labels, rotation=30, color="white")

        ax.tick_params(colors='white')
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")

        st.pyplot(fig)

    with col2:
        st.subheader("Failure Distribution")

        failure_sum = df[FAILURE_TYPES].sum().fillna(0)

        if failure_sum.sum() == 0:
            st.info("No failure data available.")
        else:
            threshold = failure_sum.sum() * 0.05
            small = failure_sum[failure_sum < threshold].sum()
            large = failure_sum[failure_sum >= threshold].copy()

            if small > 0:
                large["Others"] = small

            fig2, ax2 = plt.subplots(figsize=(5,4))
            ax2.pie(
                large,
                labels=large.index,
                autopct='%1.1f%%',
                colors=[COLORS.get(k, "#888") for k in large.index],
                textprops={'color': "white"}
            )

            fig2.patch.set_facecolor("#0e1117")
            st.pyplot(fig2)

# ==============================
# PAGE 2: MONTHLY
# ==============================
elif page == "Monthly Analysis":

    st.title("Month-wise Analysis")

    month_list = df.sort_values("month_dt")["month"].dropna().unique()
    selected_month = st.selectbox("Select Month", month_list)

    filtered_df = df[df["month"] == selected_month]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 7 Failure-Prone Pumps")

        top_pumps = (
            filtered_df.groupby("pump")["Total_Failure"]
            .sum()
            .sort_values(ascending=False)
            .head(7)
        )

        fig1, ax1 = plt.subplots(figsize=(6,4))
        top_pumps.plot(kind="bar", ax=ax1, color="#00E5FF")

        ax1.tick_params(colors='white')
        ax1.set_facecolor("#0e1117")
        fig1.patch.set_facecolor("#0e1117")

        st.pyplot(fig1)

    with col2:
        st.subheader("Failure Type Distribution")

        failure_sum = filtered_df[FAILURE_TYPES].sum().fillna(0)

        if failure_sum.sum() == 0:
            st.info("No failure data for this month.")
        else:
            threshold = failure_sum.sum() * 0.05
            small = failure_sum[failure_sum < threshold].sum()
            large = failure_sum[failure_sum >= threshold].copy()

            if small > 0:
                large["Others"] = small

            fig2, ax2 = plt.subplots(figsize=(6,4))
            ax2.pie(
                large,
                labels=large.index,
                autopct='%1.1f%%',
                colors=[COLORS.get(k, "#888") for k in large.index],
                textprops={'color': "white"}
            )

            fig2.patch.set_facecolor("#0e1117")
            st.pyplot(fig2)

# ==============================
# PAGE 3: PUMP ANALYSIS
# ==============================
elif page == "Pump Analysis":

    st.title("Pump Level Analysis")

    valid_pumps = df[df["history"].notna() & (df["history"] != "")]
    valid_pumps = valid_pumps["pump"].unique()

    selected_pump = st.selectbox("Select Pump", sorted(valid_pumps))

    pump_df = df[df["pump"] == selected_pump]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Equipment History")

        history = pump_df[
            (pump_df["history"].notna()) &
            (pump_df["history"] != "")
        ][["date", "shift", "event"]].sort_values("date")

        st.dataframe(history, use_container_width=True)

    with col2:
        st.subheader("Failure Split")

        failure_sum = pump_df[FAILURE_TYPES].sum().fillna(0)

        if failure_sum.sum() == 0:
            st.info("No failure data available for this pump.")
        else:
            threshold = failure_sum.sum() * 0.05
            small = failure_sum[failure_sum < threshold].sum()
            large = failure_sum[failure_sum >= threshold].copy()

            if small > 0:
                large["Others"] = small

            fig, ax = plt.subplots(figsize=(5,4))
            ax.pie(
                large,
                labels=large.index,
                autopct='%1.1f%%',
                colors=[COLORS.get(k, "#888") for k in large.index],
                textprops={'color': "white"}
            )

            fig.patch.set_facecolor("#0e1117")
            st.pyplot(fig)
