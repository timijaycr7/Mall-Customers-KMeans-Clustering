import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Dataset Explorer")
st.markdown("Explore the Mall Customers dataset used for clustering.")

st.markdown("---")

CSV_URL = "https://raw.githubusercontent.com/timijaycr7/Mall-Customers-KMeans-Clustering/master/Mall_Customers.csv"


@st.cache_data(ttl=600)
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(CSV_URL)


try:
    df = load_dataset()
except Exception:
    st.error("Could not load dataset from GitHub. Showing sample data.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Rows", df.shape[0])
with col2:
    st.metric("Columns", df.shape[1])
with col3:
    st.metric("Missing Values", int(df.isnull().sum().sum()))

st.markdown("---")

st.subheader("Summary Statistics")
st.dataframe(df.describe().round(2), use_container_width=True)

st.markdown("---")

st.subheader("Feature Distributions")

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        df, x="Annual Income (k$)", nbins=20,
        title="Annual Income Distribution",
        color_discrete_sequence=["#636EFA"],
    )
    fig.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(
        df, x="Spending Score (1-100)", nbins=20,
        title="Spending Score Distribution",
        color_discrete_sequence=["#00CC96"],
    )
    fig.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        df, x="Age", nbins=15,
        title="Age Distribution",
        color_discrete_sequence=["#EF553B"],
    )
    fig.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.pie(
        df, names="Gender",
        title="Gender Distribution",
        color_discrete_sequence=["#AB63FA", "#FFA15A"],
    )
    fig.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.subheader("Feature Correlations")
fig = px.scatter(
    df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="Gender",
    title="Income vs Spending Score (colored by Gender)",
    color_discrete_sequence=["#636EFA", "#EF553B"],
)
fig.update_layout(template="plotly_dark", height=450)
st.plotly_chart(fig, use_container_width=True)
