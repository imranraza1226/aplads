"""
Anomaly detection model for the AI Log Anomaly Detection System.

Uses Isolation Forest — an unsupervised algorithm well-suited for
high-dimensional log data. No labeled data required.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# Default model hyperparameters
CONTAMINATION = 0.05   # Expected fraction of anomalies in data
RANDOM_STATE = 42
N_ESTIMATORS = 100


def train_model(
    X: pd.DataFrame,
    contamination: float = CONTAMINATION,
    random_state: int = RANDOM_STATE,
) -> tuple[IsolationForest, StandardScaler]:
    """
    Scale features and train an Isolation Forest model.

    Args:
        X: Feature matrix (numeric only).
        contamination: Expected proportion of outliers.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (trained IsolationForest, fitted StandardScaler).
    """
    print(f"[->] Training Isolation Forest (contamination={contamination})...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    print("[OK] Model training complete.")
    return model, scaler


def predict(
    model: IsolationForest,
    scaler: StandardScaler,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Generate anomaly predictions for new data.

    Args:
        model: Trained IsolationForest.
        scaler: Fitted StandardScaler used during training.
        X: Feature matrix to predict on.

    Returns:
        Array of labels: -1 (anomaly) or 1 (normal).
    """
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)


def anomaly_scores(
    model: IsolationForest,
    scaler: StandardScaler,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Return raw anomaly scores (lower = more anomalous).

    Args:
        model: Trained IsolationForest.
        scaler: Fitted StandardScaler.
        X: Feature matrix.

    Returns:
        Array of float scores.
    """
    X_scaled = scaler.transform(X)
    # decision_function returns negative scores; we negate so higher = more anomalous
    return -model.decision_function(X_scaled)


def evaluate(df_original: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Attach predictions to the original DataFrame and print a summary.

    Args:
        df_original: Original (preprocessed) DataFrame.
        labels: Prediction array from predict().

    Returns:
        DataFrame with a new 'anomaly_label' column (-1 or 1).
    """
    df = df_original.copy()
    df["anomaly_label"] = labels

    total = len(df)
    n_anomalies = (labels == -1).sum()
    n_normal = (labels == 1).sum()

    print(f"\n{'='*50}")
    print(f"  Detection Results")
    print(f"{'='*50}")
    print(f"  Total records  : {total}")
    print(f"  Normal         : {n_normal} ({n_normal/total*100:.1f}%)")
    print(f"  Anomalies      : {n_anomalies} ({n_anomalies/total*100:.1f}%)")
    print(f"{'='*50}")

    # Show a sample of anomalous records (original columns only)
    display_cols = [c for c in ["ip_address", "timestamp", "login_status", "request_type"] if c in df.columns]
    anomalies = df[df["anomaly_label"] == -1][display_cols].head(10)
    print("\n  Sample anomalous records:")
    print(anomalies.to_string(index=False))

    return df


def plot_anomalies(
    X: pd.DataFrame,
    labels: np.ndarray,
    scores: np.ndarray,
    save_path: str | None = None,
) -> None:
    """
    Create a 2-panel visualization: scatter plot + score distribution.

    Args:
        X: Feature matrix used for prediction.
        labels: Prediction labels (-1 / 1).
        scores: Anomaly scores (higher = more anomalous).
        save_path: If provided, save the figure to this path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("AI Log Anomaly Detection — Results", fontsize=14, fontweight="bold")

    colors = np.where(labels == -1, "#e74c3c", "#2ecc71")

    # Panel 1: scatter of first two features
    axes[0].scatter(
        X.iloc[:, 0], X.iloc[:, 1],
        c=colors, alpha=0.6, edgecolors="none", s=30
    )
    axes[0].set_xlabel(X.columns[0])
    axes[0].set_ylabel(X.columns[1])
    axes[0].set_title("Anomaly Scatter Plot")
    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="Anomaly"),
        Patch(facecolor="#2ecc71", label="Normal"),
    ]
    axes[0].legend(handles=legend_elements, loc="upper right")

    # Panel 2: anomaly score distribution
    axes[1].hist(scores[labels == 1],  bins=40, color="#2ecc71", alpha=0.7, label="Normal")
    axes[1].hist(scores[labels == -1], bins=40, color="#e74c3c", alpha=0.7, label="Anomaly")
    axes[1].set_xlabel("Anomaly Score (higher = more anomalous)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Anomaly Score Distribution")
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Plot saved to: {save_path}")

    plt.show()


def save_model(model: IsolationForest, scaler: StandardScaler, directory: str) -> None:
    """Persist the trained model and scaler to disk."""
    os.makedirs(directory, exist_ok=True)
    joblib.dump(model,  os.path.join(directory, "isolation_forest.pkl"))
    joblib.dump(scaler, os.path.join(directory, "scaler.pkl"))
    print(f"[OK] Model and scaler saved to: {directory}")


def load_model(directory: str) -> tuple[IsolationForest, StandardScaler]:
    """Load a persisted model and scaler from disk."""
    model  = joblib.load(os.path.join(directory, "isolation_forest.pkl"))
    scaler = joblib.load(os.path.join(directory, "scaler.pkl"))
    print(f"[OK] Model loaded from: {directory}")
    return model, scaler
