# Mall Customers K-Means Clustering

A customer segmentation project using K-Means clustering on the [Mall Customers dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python). The goal is to identify distinct customer groups based on their annual income and spending behavior.

## Results

The model identifies **5 customer segments** using Annual Income and Spending Score as features:

![Cluster Scatter Plot](cluster_scatter.png)

## Approach

1. **Preprocessing** (`preprocess.py`) — Extracts Annual Income and Spending Score features, applies a power transformation (power=5) with standard scaling to amplify separation between groups.

2. **Modeling** (`kmeans_model.py`) — Evaluates K from 3 to 10 using the Elbow Method and Silhouette Score across multiple random seeds to find the optimal number of clusters, then fits the final model with the best K.

## Project Structure

```
Mall_Customers.csv          # Raw dataset (200 customers)
preprocess.py               # Data loading, feature extraction, scaling
kmeans_model.py             # K selection, model fitting, evaluation, plotting
cluster_scatter.png         # Final cluster visualization
elbow_silhouette.png        # Elbow + Silhouette score plots
silhouette_analysis.png     # Per-cluster silhouette analysis
```

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python kmeans_model.py
```

## Requirements

- Python 3.10+
- pandas, numpy, scikit-learn, matplotlib, seaborn
