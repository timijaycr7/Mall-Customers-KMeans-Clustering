from http.server import BaseHTTPRequestHandler
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from urllib.parse import parse_qs, urlparse

POWER = 5
DATA_URL = "https://raw.githubusercontent.com/timijaycr7/Mall-Customers-KMeans-Clustering/master/Mall_Customers.csv"


def run_clustering(k=None, k_min=3, k_max=10):
    df = pd.read_csv(DATA_URL)

    features = df[["Annual Income (k$)", "Spending Score (1-100)"]].copy()
    features.columns = ["Annual_Income", "Spending_Score"]
    features = features.dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    X_power = np.sign(X_scaled) * np.abs(X_scaled) ** POWER
    X_final = StandardScaler().fit_transform(X_power)

    if k is None:
        k_range = range(k_min, k_max + 1)
        eval_results = []
        best_k = k_min
        best_sil = -1

        for ki in k_range:
            km = KMeans(n_clusters=ki, init="k-means++", n_init=20, random_state=42, max_iter=500)
            labels = km.fit_predict(X_final)
            sil = silhouette_score(X_final, labels)
            eval_results.append({"k": ki, "inertia": round(km.inertia_, 2), "silhouette_score": round(sil, 4)})
            if sil > best_sil:
                best_sil = sil
                best_k = ki

        k = best_k
    else:
        eval_results = None

    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=42, max_iter=500)
    labels = kmeans.fit_predict(X_final)
    sil_avg = silhouette_score(X_final, labels)
    sil_samples = silhouette_samples(X_final, labels)

    clusters = []
    for c in range(k):
        mask = labels == c
        cluster_data = features[mask]
        clusters.append({
            "cluster_id": int(c),
            "size": int(mask.sum()),
            "silhouette_score": round(float(sil_samples[mask].mean()), 4),
            "avg_annual_income": round(float(cluster_data["Annual_Income"].mean()), 2),
            "avg_spending_score": round(float(cluster_data["Spending_Score"].mean()), 2),
        })

    customers = []
    for i, row in features.iterrows():
        customers.append({
            "customer_id": int(i) + 1,
            "annual_income": float(row["Annual_Income"]),
            "spending_score": float(row["Spending_Score"]),
            "cluster": int(labels[features.index.get_loc(i)]),
        })

    response = {
        "model": {
            "k": int(k),
            "overall_silhouette_score": round(float(sil_avg), 4),
            "total_customers": len(features),
        },
        "clusters": clusters,
        "customers": customers,
    }

    if eval_results is not None:
        response["k_evaluation"] = eval_results

    return response


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            params = parse_qs(urlparse(self.path).query)
            k = params.get("k", [None])[0]
            if k is not None:
                k = int(k)
                if k < 2 or k > 15:
                    raise ValueError("k must be between 2 and 15")

            result = run_clustering(k=k)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result, indent=2).encode())

        except ValueError as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
