import streamlit as st

st.set_page_config(
    page_title="Customer Segmentation | K-Means",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

home = st.Page("pages/Home.py", title="Home", icon="🏠", default=True)
dataset = st.Page("pages/Dataset.py", title="Dataset Explorer", icon="📋")
clustering = st.Page("pages/Clustering.py", title="Clustering Dashboard", icon="📊")
analytics = st.Page("pages/Analytics.py", title="Analytics & Insights", icon="💡")
prediction = st.Page("pages/Prediction.py", title="Customer Prediction", icon="🔮")

pg = st.navigation([home, dataset, clustering, analytics, prediction])

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Customer Segmentation**\n\n"
    "K-Means Clustering on Mall Customers\n\n"
    "[GitHub](https://github.com/timijaycr7/Mall-Customers-KMeans-Clustering) | "
    "[API](https://data-psi-blue.vercel.app/api)"
)

pg.run()
