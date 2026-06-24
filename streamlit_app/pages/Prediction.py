import streamlit as st
import requests
import plotly.graph_objects as go
from services.api_client import fetch_cluster_results, get_customers_df
from utils.helpers import CLUSTER_PERSONAS, CLUSTER_COLORS

API_BASE = "https://data-psi-blue.vercel.app"

st.title("Customer Segment Prediction")
st.markdown("Enter a customer's **Annual Income** and **Spending Score** to predict which segment they belong to.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    income = st.slider("Annual Income (k$)", min_value=1, max_value=200, value=60, step=1)
with col2:
    score = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50, step=1)

st.markdown("---")

if st.button("Predict Segment", type="primary", use_container_width=True):
    with st.spinner("Calling prediction API..."):
        try:
            resp = requests.get(
                f"{API_BASE}/api/predict",
                params={"income": income, "score": score},
                timeout=15,
            )
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

    cluster_id = result["cluster"]
    segment_name = result["segment_name"]
    persona = CLUSTER_PERSONAS.get(cluster_id, {})

    st.markdown("---")

    st.subheader(f"{persona.get('icon', '📌')} Predicted Segment: {segment_name}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cluster ID", cluster_id)
    with col2:
        st.metric("Annual Income", f"${income}k")
    with col3:
        st.metric("Spending Score", f"{score}/100")

    st.info(f"**Profile:** {persona.get('description', 'N/A')}")
    st.success(f"**Recommended Strategy:** {persona.get('strategy', 'N/A')}")

    st.markdown("---")
    st.subheader("Your Customer on the Segment Map")

    try:
        results = fetch_cluster_results()
        customers_df = get_customers_df(results)

        fig = go.Figure()

        for cid in sorted(customers_df["Cluster"].unique()):
            mask = customers_df["Cluster"] == cid
            cluster_data = customers_df[mask]
            fig.add_trace(go.Scatter(
                x=cluster_data["Annual Income (k$)"],
                y=cluster_data["Spending Score (1-100)"],
                mode="markers",
                name=f"Cluster {cid}: {CLUSTER_PERSONAS.get(cid, {}).get('name', '')}",
                marker=dict(size=8, color=CLUSTER_COLORS[cid], opacity=0.5),
            ))

        fig.add_trace(go.Scatter(
            x=[income],
            y=[score],
            mode="markers+text",
            name="Your Customer",
            text=["YOU"],
            textposition="top center",
            marker=dict(size=18, color="white", symbol="star", line=dict(width=2, color=CLUSTER_COLORS[cluster_id])),
        ))

        fig.update_layout(
            title=f"Predicted: {segment_name} (Cluster {cluster_id})",
            xaxis_title="Annual Income (k$)",
            yaxis_title="Spending Score (1-100)",
            template="plotly_dark",
            height=550,
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception:
        st.warning("Could not load cluster map for visualization.")

st.markdown("---")

st.subheader("Segment Reference Guide")

for cid in sorted(CLUSTER_PERSONAS.keys()):
    p = CLUSTER_PERSONAS[cid]
    with st.expander(f"{p['icon']} Cluster {cid}: {p['name']}"):
        st.markdown(f"**Profile:** {p['description']}")
        st.markdown(f"**Strategy:** {p['strategy']}")
