import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

CLUSTER_COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

CLUSTER_PERSONAS = {
    0: {
        "name": "Average Joes",
        "icon": "👥",
        "description": "The largest group with moderate income and moderate spending. They represent the mainstream customer base.",
        "strategy": "Broad marketing campaigns, loyalty programs, and cross-selling to increase basket size.",
    },
    1: {
        "name": "Elite Outliers",
        "icon": "💎",
        "description": "Very high income with moderate spending. A tiny but extremely valuable niche segment.",
        "strategy": "White-glove VIP treatment, exclusive product previews, and personal shopping assistants.",
    },
    2: {
        "name": "Budget Savers",
        "icon": "💰",
        "description": "Moderate income but very low spending scores. These customers are cautious spenders.",
        "strategy": "Targeted discounts, value bundles, flash sales, and budget-friendly product lines.",
    },
    3: {
        "name": "Affluent Moderates",
        "icon": "🏦",
        "description": "High income with moderate spending. They have purchasing power but aren't fully engaged.",
        "strategy": "Premium brand partnerships, experiential marketing, and status-driven loyalty tiers.",
    },
    4: {
        "name": "Big Spenders",
        "icon": "🛍️",
        "description": "Moderate income but very high spending scores. They love to shop and are highly engaged.",
        "strategy": "Reward programs, early access to sales, installment plans, and referral bonuses.",
    },
}


def create_scatter_plot(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color="Cluster",
        color_discrete_sequence=CLUSTER_COLORS,
        title="Customer Segments: Income vs Spending",
        hover_data=["Customer ID"],
        category_orders={"Cluster": sorted(df["Cluster"].unique())},
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    fig.update_layout(
        template="plotly_dark",
        height=550,
        legend_title_text="Cluster",
    )
    return fig


def create_cluster_bar(clusters_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        clusters_df,
        x="Cluster",
        y="Size",
        color="Cluster",
        color_discrete_sequence=CLUSTER_COLORS,
        title="Customers per Cluster",
        text="Size",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(template="plotly_dark", height=400, showlegend=False)
    return fig


def create_elbow_plot(k_eval_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_eval_df["K"], y=k_eval_df["Inertia"],
        mode="lines+markers", name="Inertia",
        line=dict(color="#636EFA", width=3),
        marker=dict(size=10),
    ))
    fig.add_vline(x=5, line_dash="dash", line_color="red", annotation_text="Best K=5")
    fig.update_layout(
        title="Elbow Method: Inertia vs K",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Inertia (WCSS)",
        template="plotly_dark",
        height=400,
    )
    return fig


def create_silhouette_plot(k_eval_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_eval_df["K"], y=k_eval_df["Silhouette Score"],
        mode="lines+markers", name="Silhouette",
        line=dict(color="#00CC96", width=3),
        marker=dict(size=10),
    ))
    fig.add_vline(x=5, line_dash="dash", line_color="red", annotation_text="Best K=5")
    fig.update_layout(
        title="Silhouette Score vs K",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Silhouette Score",
        template="plotly_dark",
        height=400,
    )
    return fig


def create_radar_chart(clusters_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for _, row in clusters_df.iterrows():
        cluster_id = int(row["Cluster"])
        fig.add_trace(go.Scatterpolar(
            r=[row["Avg Annual Income (k$)"], row["Avg Spending Score"], row["Size"], row["Silhouette Score"] * 100],
            theta=["Avg Income", "Avg Spending", "Cluster Size", "Silhouette (x100)"],
            fill="toself",
            name=f"Cluster {cluster_id}",
            line=dict(color=CLUSTER_COLORS[cluster_id]),
        ))
    fig.update_layout(
        title="Cluster Comparison Radar",
        template="plotly_dark",
        height=500,
        polar=dict(radialaxis=dict(visible=True)),
    )
    return fig


def create_income_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="Annual Income (k$)",
        color="Cluster",
        color_discrete_sequence=CLUSTER_COLORS,
        title="Income Distribution by Cluster",
        barmode="overlay",
        opacity=0.7,
        category_orders={"Cluster": sorted(df["Cluster"].unique())},
    )
    fig.update_layout(template="plotly_dark", height=400)
    return fig


def create_spending_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="Spending Score (1-100)",
        color="Cluster",
        color_discrete_sequence=CLUSTER_COLORS,
        title="Spending Score Distribution by Cluster",
        barmode="overlay",
        opacity=0.7,
        category_orders={"Cluster": sorted(df["Cluster"].unique())},
    )
    fig.update_layout(template="plotly_dark", height=400)
    return fig


def df_to_csv(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)
