import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from preprocess import load_data, prepare_features, scale_features


def find_optimal_k(X: np.ndarray, k_range: range = range(3, 11)) -> dict:
    inertias = []
    silhouette_scores = []

    print("\nEvaluating K values...")
    print(f"{'K':>3} | {'Inertia':>12} | {'Silhouette':>10}")
    print("-" * 35)

    for k in k_range:
        best_sil_k = -1
        for seed in range(10):
            km = KMeans(n_clusters=k, init="k-means++", n_init=50, random_state=seed, max_iter=500)
            lbl = km.fit_predict(X)
            s = silhouette_score(X, lbl)
            if s > best_sil_k:
                best_sil_k = s
                best_km = km
                best_lbl = lbl
        kmeans = best_km
        labels = best_lbl
        sil = best_sil_k

        inertias.append(kmeans.inertia_)
        silhouette_scores.append(sil)
        print(f"{k:>3} | {kmeans.inertia_:>12.2f} | {sil:>10.4f}")

    best_idx = np.argmax(silhouette_scores)
    best_k = list(k_range)[best_idx]
    best_sil = silhouette_scores[best_idx]
    print(f"\nBest K = {best_k} (Silhouette Score = {best_sil:.4f})")

    return {
        "k_range": list(k_range),
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
        "best_k": best_k,
        "best_silhouette": best_sil,
    }


def plot_elbow_and_silhouette(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(results["k_range"], results["inertias"], "bo-", linewidth=2, markersize=8)
    axes[0].axvline(x=results["best_k"], color="r", linestyle="--", alpha=0.7, label=f"Best K={results['best_k']}")
    axes[0].set_xlabel("Number of Clusters (K)", fontsize=12)
    axes[0].set_ylabel("Inertia (WCSS)", fontsize=12)
    axes[0].set_title("Elbow Method", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(results["k_range"], results["silhouette_scores"], "go-", linewidth=2, markersize=8)
    axes[1].axvline(x=results["best_k"], color="r", linestyle="--", alpha=0.7, label=f"Best K={results['best_k']}")
    axes[1].set_xlabel("Number of Clusters (K)", fontsize=12)
    axes[1].set_ylabel("Silhouette Score", fontsize=12)
    axes[1].set_title("Silhouette Score vs K", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("elbow_silhouette.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: elbow_silhouette.png")


def fit_final_model(X: np.ndarray, k: int) -> tuple[KMeans, np.ndarray]:
    best_sil = -1
    best_labels = None
    best_model = None

    for seed in range(20):
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=50, random_state=seed, max_iter=500)
        labels = kmeans.fit_predict(X)
        sil = silhouette_score(X, labels)
        if sil > best_sil:
            best_sil = sil
            best_labels = labels
            best_model = kmeans

    print(f"\nBest model from 20 random seeds: Silhouette = {best_sil:.4f}")
    return best_model, best_labels


def evaluate_clusters(X: np.ndarray, labels: np.ndarray, features, feature_names: list) -> None:
    sil_avg = silhouette_score(X, labels)
    sil_samples = silhouette_samples(X, labels)

    print(f"\n{'='*50}")
    print(f"FINAL MODEL EVALUATION")
    print(f"{'='*50}")
    print(f"Number of clusters: {len(set(labels))}")
    print(f"Overall Silhouette Score: {sil_avg:.4f}")
    print()

    for cluster in sorted(set(labels)):
        mask = labels == cluster
        cluster_sil = sil_samples[mask].mean()
        print(f"Cluster {cluster}: {mask.sum():>4} samples | Silhouette: {cluster_sil:.4f}")

    print(f"\n{'='*50}")
    print("CLUSTER PROFILES (Original Scale)")
    print(f"{'='*50}")
    cluster_df = features.copy()
    cluster_df["Cluster"] = labels
    profile = cluster_df.groupby("Cluster")[feature_names].mean().round(2)
    print(profile.to_string())


def plot_silhouette_analysis(X: np.ndarray, labels: np.ndarray, k: int) -> None:
    sil_samples = silhouette_samples(X, labels)
    sil_avg = silhouette_score(X, labels)

    fig, ax = plt.subplots(figsize=(10, 7))
    y_lower = 10

    colors = sns.color_palette("husl", k)
    for i in range(k):
        cluster_sil = np.sort(sil_samples[labels == i])
        cluster_size = cluster_sil.shape[0]
        y_upper = y_lower + cluster_size

        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                         facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * cluster_size, str(i), fontsize=12, fontweight="bold")
        y_lower = y_upper + 10

    ax.axvline(x=sil_avg, color="red", linestyle="--", linewidth=2, label=f"Avg: {sil_avg:.4f}")
    ax.set_xlabel("Silhouette Coefficient", fontsize=12)
    ax.set_ylabel("Cluster", fontsize=12)
    ax.set_title(f"Silhouette Analysis (K={k})", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("silhouette_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: silhouette_analysis.png")


def plot_clusters_2d(features, labels: np.ndarray, k: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette("husl", k)

    for cluster in range(k):
        mask = labels == cluster
        ax.scatter(features.loc[mask, "Annual_Income"], features.loc[mask, "Spending_Score"],
                   c=[colors[cluster]], label=f"Cluster {cluster}",
                   s=80, alpha=0.7, edgecolors="w", linewidth=0.5)

    ax.set_xlabel("Annual Income (k$)", fontsize=13)
    ax.set_ylabel("Spending Score (1-100)", fontsize=13)
    ax.set_title(f"K-Means Clustering (K={k})", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("cluster_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: cluster_scatter.png")


def main():
    df = load_data()
    features, feature_names = prepare_features(df)
    X_scaled, scaler = scale_features(features)

    results = find_optimal_k(X_scaled)

    plot_elbow_and_silhouette(results)

    best_k = results["best_k"]
    kmeans, labels = fit_final_model(X_scaled, best_k)

    evaluate_clusters(X_scaled, labels, features, feature_names)

    plot_silhouette_analysis(X_scaled, labels, best_k)
    plot_clusters_2d(features, labels, best_k)

    print(f"\nDone. All plots saved to current directory.")


if __name__ == "__main__":
    main()
