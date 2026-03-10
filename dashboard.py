import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pydeck as pdk

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Urban Infrastructure AI Monitoring System",
    layout="wide"
)

st.title("AI System for Predictive Monitoring of Urban Infrastructure")
st.markdown("Flood-Induced Cascading Failure Analysis for Pune Road Network")

# -----------------------------
# LOAD DATA
# -----------------------------

cascade_path = BASE_DIR / "data" / "graphs" / "run_results_weather_2.5.json"
repair_path = BASE_DIR / "data" / "graphs" / "repair_impact_scale_1.2.json"
opt_path = BASE_DIR / "data" / "graphs" / "opt_vs_random_scale_1.2.json"
roads_path = BASE_DIR / "data" / "gis" / "roads_final.geojson"

# -----------------------------
# CREATE TABS
# -----------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Cascade Simulation",
    "Resilience Analysis",
    "Repair Optimization",
    "Infrastructure Map"
])

# =============================
# TAB 1 : OVERVIEW
# =============================

with tab1:

    st.subheader("System Risk Overview")

    if cascade_path.exists() and repair_path.exists():

        with open(cascade_path) as f:
            cascade_data = json.load(f)

        with open(repair_path) as f:
            repair_data = json.load(f)

        failures = cascade_data["capacity_cascade"]

        total_failures = sum(failures)
        max_failures = max(failures)
        cascade_steps = len(failures)

        resilience_values = [1 - r["frac_failed"] for r in repair_data]
        resilience_score = max(resilience_values)

        # Risk classification
        if total_failures > 50000:
            risk_level = "HIGH"
        elif total_failures > 20000:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "System Risk Level",
            risk_level
        )

        col2.metric(
            "Total Failed Roads",
            int(total_failures)
        )

        col3.metric(
            "Resilience Score",
            round(resilience_score, 3)
        )

        col4.metric(
            "Cascade Steps",
            cascade_steps
        )

        st.markdown("### System Status")

        if risk_level == "HIGH":
            st.error("Critical infrastructure stress detected. Immediate repairs recommended.")
        elif risk_level == "MEDIUM":
            st.warning("Moderate infrastructure risk detected.")
        else:
            st.success("Infrastructure network operating within safe limits.")

    else:

        st.warning("Simulation data not available")

    st.markdown("""
### System Description

This AI system monitors urban infrastructure networks and predicts cascading failures
during flood events.

Key capabilities:

• Disaster scenario simulation  
• Graph neural network failure prediction  
• Repair strategy optimization  
• Infrastructure resilience analysis  
• Smart-city monitoring dashboard
""")
# =============================
# TAB 2 : CASCADE SIMULATION
# =============================

with tab2:

    st.subheader("Cascade Failure Simulation")

    if cascade_path.exists():

        with open(cascade_path) as f:
            data = json.load(f)

        failures = data["capacity_cascade"]
        steps = list(range(1, len(failures)+1))

        step = st.slider(
            "Select Cascade Step",
            min_value=1,
            max_value=len(failures),
            value=1
        )

        st.metric(
            "Failures at Selected Step",
            int(failures[step-1])
        )

        fig, ax = plt.subplots(figsize=(6,4))

        ax.plot(steps, failures, marker="o")
        ax.axvline(step, linestyle="--")

        ax.set_xlabel("Cascade Step")
        ax.set_ylabel("New Failures")
        ax.set_title("Cascade Propagation")

        st.pyplot(fig)

    else:
        st.warning("Cascade results not found")

# =============================
# TAB 3 : RESILIENCE ANALYSIS
# =============================

with tab3:

    st.subheader("Infrastructure Resilience")

    if repair_path.exists():

        with open(repair_path) as f:
            results = json.load(f)

        budgets = [r["K"] for r in results]
        resilience = [1 - r["frac_failed"] for r in results]

        fig, ax = plt.subplots(figsize=(6,4))

        ax.plot(budgets, resilience, marker="o")

        ax.set_xlabel("Repair Budget")
        ax.set_ylabel("Network Resilience")
        ax.set_title("Resilience vs Repair Budget")

        st.pyplot(fig)

    else:
        st.warning("Repair impact data not found")

# =============================
# TAB 4 : REPAIR OPTIMIZATION
# =============================

with tab4:

    st.subheader("Repair Strategy Comparison")

    if opt_path.exists():

        with open(opt_path) as f:
            data = json.load(f)

        budgets = [r["K"] for r in data]
        greedy = [r["greedy_failed"] for r in data]
        random = [r["random_failed"] for r in data]

        fig, ax = plt.subplots(figsize=(6,4))

        ax.plot(budgets, greedy, marker="o", label="Greedy Optimizer")
        ax.plot(budgets, random, marker="o", label="Random Strategy")

        ax.set_xlabel("Repair Budget")
        ax.set_ylabel("Number of Failed Roads")
        ax.set_title("Repair Strategy Comparison")

        ax.legend()

        st.pyplot(fig)

        # -----------------------------
        # AI REPAIR RECOMMENDATION
        # -----------------------------

        st.subheader("AI Repair Recommendation")

        improvements = []

        for i in range(len(budgets)):
            improvement = random[i] - greedy[i]
            improvements.append((budgets[i], improvement))

        improvements.sort(key=lambda x: x[1], reverse=True)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Top Repair Budget",
            f"{improvements[0][0]} roads"
        )

        col2.metric(
            "Failure Reduction",
            f"{int(improvements[0][1])} roads saved"
        )

        col3.metric(
            "Best Strategy",
            "Greedy Optimizer"
        )

        st.markdown("### Recommended Repair Priority")

        recommended = sorted(data, key=lambda x: x["greedy_failed"])[:3]

        for i, r in enumerate(recommended):
            st.write(f"{i+1}. Repair budget {r['K']} → expected failures {int(r['greedy_failed'])}")

    else:
        st.warning("Optimization results not found")

    # -----------------------------
    # MODEL PERFORMANCE
    # -----------------------------

    st.subheader("Model Performance")

    perf_data = {
        "Model":[
            "Random Forest",
            "Temporal Neural Network",
            "Graph Neural Network",
            "Spatio-Temporal GNN"
        ],
        "ROC-AUC":[
            0.62,
            0.84,
            0.51,
            0.63
        ]
    }

    df = pd.DataFrame(perf_data)

    st.dataframe(df, use_container_width=True)

# =============================
# TAB 5 : INFRASTRUCTURE MAP
# =============================

with tab5:

    st.subheader("Pune Road Infrastructure Map")

    if roads_path.exists():

        with open(roads_path) as f:
            roads_geojson = json.load(f)

        layer = pdk.Layer(
            "GeoJsonLayer",
            roads_geojson,
            pickable=True,
            stroked=True,
            filled=False,
            get_line_color=[200,30,0],
            get_line_width=2
        )

        view_state = pdk.ViewState(
            latitude=18.5204,
            longitude=73.8567,
            zoom=11
        )

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text":"Road Segment"}
        )

        st.pydeck_chart(deck)

    else:
        st.warning("Road GIS data not found")