import streamlit as st
from services.api_client import (
    fetch_cluster_results,
    get_customers_df,
    get_clusters_df,
    get_k_evaluation_df,
)
from utils.helpers import (
    create_scatter_plot,
    create_cluster_bar,
    create_elbow_plot,
    create_silhouette_plot,
    create_income_distribution,
    create_spending_distribution,
    df_to_csv,
)

st.title("Clustering Dashboard")
st.markdown("Live results from the deployed Vercel API.")

st.markdown("---")

with st.spinner("Fetching clustering results from API..."):
    try:
        results = fetch_cluster_results()
    except Exception as e:
        st.error(f"Failed to fetch API results: {e}")
        st.stop()

model = results["model"]
customers_df = get_customers_df(results)
clusters_df = get_clusters_df(results)
k_eval_df = get_k_evaluation_df(results)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Optimal K", model["k"])
with col2:
    st.metric("Silhouette Score", f"{model['overall_silhouette_score']:.4f}")
with col3:
    st.metric("Total Customers", model["total_customers"])

st.markdown("---")

st.subheader("Customer Segments")
st.plotly_chart(create_scatter_plot(customers_df), use_container_width=True)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Elbow Method")
    st.plotly_chart(create_elbow_plot(k_eval_df), use_container_width=True)

with col2:
    st.subheader("Silhouette Score vs K")
    st.plotly_chart(create_silhouette_plot(k_eval_df), use_container_width=True)

st.markdown("---")

st.subheader("Cluster Distribution")
st.plotly_chart(create_cluster_bar(clusters_df), use_container_width=True)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Income by Cluster")
    st.plotly_chart(create_income_distribution(customers_df), use_container_width=True)

with col2:
    st.subheader("Spending by Cluster")
    st.plotly_chart(create_spending_distribution(customers_df), use_container_width=True)

st.markdown("---")

st.subheader("Cluster Profiles")
st.dataframe(
    clusters_df.style.format({
        "Silhouette Score": "{:.4f}",
        "Avg Annual Income (k$)": "{:.2f}",
        "Avg Spending Score": "{:.2f}",
    }),
    use_container_width=True,
)

st.markdown("---")

st.subheader("K Evaluation Table")
st.dataframe(
    k_eval_df.style.format({
        "Inertia": "{:.2f}",
        "Silhouette Score": "{:.4f}",
    }),
    use_container_width=True,
)

st.markdown("---")

st.subheader("Download Results")

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        label="Download Customer Assignments (CSV)",
        data=df_to_csv(customers_df),
        file_name="customer_clusters.csv",
        mime="text/csv",
    )
with col2:
    st.download_button(
        label="Download Cluster Profiles (CSV)",
        data=df_to_csv(clusters_df),
        file_name="cluster_profiles.csv",
        mime="text/csv",
    )
