import streamlit as st
from services.api_client import fetch_cluster_results, get_customers_df, get_clusters_df
from utils.helpers import CLUSTER_PERSONAS, CLUSTER_COLORS, create_radar_chart

st.title("Analytics & Business Insights")
st.markdown("Actionable intelligence derived from customer segmentation.")

st.markdown("---")

with st.spinner("Loading cluster data..."):
    try:
        results = fetch_cluster_results()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

customers_df = get_customers_df(results)
clusters_df = get_clusters_df(results)

st.subheader("Cluster Comparison")
st.plotly_chart(create_radar_chart(clusters_df), use_container_width=True)

st.markdown("---")

st.subheader("Customer Personas & Strategies")

for _, row in clusters_df.iterrows():
    cluster_id = int(row["Cluster"])
    persona = CLUSTER_PERSONAS.get(cluster_id, {})

    with st.expander(
        f"{persona.get('icon', '📌')} Cluster {cluster_id}: {persona.get('name', 'Unknown')} — {int(row['Size'])} customers",
        expanded=True,
    ):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Customers", int(row["Size"]))
        with col2:
            st.metric("Avg Income", f"${row['Avg Annual Income (k$)']:.0f}k")
        with col3:
            st.metric("Avg Spending", f"{row['Avg Spending Score']:.0f}/100")
        with col4:
            st.metric("Silhouette", f"{row['Silhouette Score']:.4f}")

        st.markdown(f"**Profile:** {persona.get('description', 'N/A')}")
        st.success(f"**Recommended Strategy:** {persona.get('strategy', 'N/A')}")

st.markdown("---")

st.subheader("Key Business Takeaways")

st.markdown("""
**1. The Majority is Moderate (Cluster 0 — 78.5% of customers)**

Most customers have average income and average spending. This is the core customer base.
Marketing should focus on incremental engagement — loyalty programs, personalized recommendations,
and cross-selling to gradually increase their spending score.

**2. Untapped High-Income Potential (Clusters 1 & 3)**

Customers with high income but moderate spending represent an opportunity gap.
They have the purchasing power but aren't fully engaged. Premium experiences,
exclusive memberships, and personalized outreach can convert them into high spenders.

**3. High-Engagement Shoppers (Cluster 4 — 10% of customers)**

These customers spend aggressively relative to their income. They are the most
engaged segment and likely brand advocates. Reward their loyalty with early access,
referral bonuses, and installment options to sustain their spending.

**4. At-Risk Savers (Cluster 2 — 8.5% of customers)**

Low spending despite moderate income signals disengagement or price sensitivity.
Targeted promotions, flash sales, and value-focused product bundles can re-engage
this segment before they churn entirely.
""")

st.markdown("---")

st.subheader("Segment Breakdown Summary")

summary_data = []
total = customers_df.shape[0]
for _, row in clusters_df.iterrows():
    cluster_id = int(row["Cluster"])
    persona = CLUSTER_PERSONAS.get(cluster_id, {})
    summary_data.append({
        "Cluster": cluster_id,
        "Persona": persona.get("name", "—"),
        "Size": int(row["Size"]),
        "% of Total": f"{row['Size'] / total * 100:.1f}%",
        "Avg Income (k$)": f"${row['Avg Annual Income (k$)']:.0f}k",
        "Avg Spending": f"{row['Avg Spending Score']:.0f}/100",
        "Strategy": persona.get("strategy", "—"),
    })

import pandas as pd
st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
