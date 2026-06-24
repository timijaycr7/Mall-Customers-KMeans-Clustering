import streamlit as st

st.title("Customer Segmentation using K-Means Clustering")

st.markdown("""
A production-grade customer segmentation system that identifies distinct customer groups
from mall shopping data using **unsupervised machine learning**. This project demonstrates
the full ML lifecycle: data preprocessing, model training, API deployment, and interactive visualization.
""")

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Customers", "200")
with col2:
    st.metric("Segments Found", "5")
with col3:
    st.metric("Silhouette Score", "0.7438")

st.markdown("---")

st.subheader("Architecture")
st.code("""
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Mall Customers  │────▶│  ML Pipeline     │────▶│  Precomputed JSON   │
│  Dataset (CSV)   │     │  (scikit-learn)  │     │  (cluster results)  │
└─────────────────┘     └──────────────────┘     └──────────┬──────────┘
                                                            │
                         ┌──────────────────┐               │
                         │  Streamlit App   │◀──── API ◀────┘
                         │  (This Dashboard)│     (Vercel)
                         └──────────────────┘
""", language=None)

st.markdown("---")

st.subheader("Methodology")

st.markdown("""
**1. Data Preprocessing**
- Selected **Annual Income** and **Spending Score** as clustering features
- Applied power transformation (p=5) to amplify group separation
- Standardized features with StandardScaler

**2. Optimal K Selection**
- **Elbow Method:** Plots inertia (WCSS) vs K to find the diminishing returns point
- **Silhouette Analysis:** Measures cluster cohesion and separation (-1 to +1, higher is better)
- Best K = **5** with silhouette score = **0.7438**

**3. Model Training**
- K-Means++ initialization with 50 restarts per seed
- Evaluated across 20 random seeds to find the globally best partition
- Final model produces 5 distinct customer segments

**4. Deployment**
- Results served via **Vercel serverless API** (zero cold-start ML dependencies)
- Interactive dashboard built with **Streamlit** consuming the live API
""")

st.markdown("---")

st.subheader("API Endpoints")

st.markdown("""
| Endpoint | Description |
|----------|-------------|
| `GET /api` | API documentation |
| `GET /api/cluster` | Full clustering results (model, clusters, customers) |
""")

st.info("Navigate using the sidebar to explore the dataset, view clustering results, and read business insights.")
