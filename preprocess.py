import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

POWER = 5


def load_data(path: str = "Mall_Customers.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"\nNull counts:\n{df.isnull().sum()}")
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    features = df[["Annual Income (k$)", "Spending Score (1-100)"]].copy()
    features.columns = ["Annual_Income", "Spending_Score"]

    features = features.dropna()
    feature_names = list(features.columns)

    print(f"\nFeature matrix: {features.shape[0]} samples x {features.shape[1]} features")
    print(f"\nFeature stats:\n{features.describe().T[['mean', 'std', 'min', 'max']]}")
    return features, feature_names


def scale_features(features: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    X_power = np.sign(X_scaled) * np.abs(X_scaled) ** POWER
    X_final = StandardScaler().fit_transform(X_power)

    print(f"\nTransformation: StandardScale -> Power({POWER}) -> StandardScale")
    print(f"Scaled feature matrix shape: {X_final.shape}")
    return X_final, scaler


if __name__ == "__main__":
    df = load_data()
    features, feature_names = prepare_features(df)
    X_scaled, scaler = scale_features(features)
