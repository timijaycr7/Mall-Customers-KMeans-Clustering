import requests
import streamlit as st
import pandas as pd

API_BASE = "https://data-psi-blue.vercel.app"


@st.cache_data(ttl=300)
def fetch_cluster_results() -> dict:
    response = requests.get(f"{API_BASE}/api/cluster", timeout=15)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=300)
def fetch_api_info() -> dict:
    response = requests.get(f"{API_BASE}/api", timeout=15)
    response.raise_for_status()
    return response.json()


def get_customers_df(results: dict) -> pd.DataFrame:
    df = pd.DataFrame(results["customers"])
    df.columns = ["Customer ID", "Annual Income (k$)", "Spending Score (1-100)", "Cluster"]
    return df


def get_clusters_df(results: dict) -> pd.DataFrame:
    df = pd.DataFrame(results["clusters"])
    df.columns = ["Cluster", "Size", "Silhouette Score", "Avg Annual Income (k$)", "Avg Spending Score"]
    return df


def get_k_evaluation_df(results: dict) -> pd.DataFrame:
    df = pd.DataFrame(results["k_evaluation"])
    df.columns = ["K", "Inertia", "Silhouette Score"]
    return df
