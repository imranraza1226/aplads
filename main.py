"""
CLI entry point for the AI Log Anomaly Detection System.

Usage:
    python main.py                        # Generate synthetic data, train, evaluate
    python main.py --data data/logs.csv  # Use an existing dataset
    python main.py --simulate            # Stream real-time log simulation after training
"""

import argparse
import os
import time
import random

import pandas as pd
import numpy as np

from src.utils import generate_synthetic_logs, save_dataset, load_dataset, print_summary
from src.preprocessing import preprocess
from src.features import build_feature_matrix
from src.model import (
    train_model,
    predict,
    anomaly_scores,
    evaluate,
    plot_anomalies,
    save_model,
)


DATA_PATH  = "data/logs.csv"
MODEL_DIR  = "data/model"
PLOT_PATH  = "data/anomaly_plot.png"


# ---------------------------------------------------------------------------
# Real-time simulation
# ---------------------------------------------------------------------------

def simulate_realtime(model, scaler, n_events: int = 20, delay: float = 0.5) -> None:
    """
    Stream synthetic log events one-by-one and alert on anomalies.

    Args:
        model: Trained IsolationForest.
        scaler: Fitted StandardScaler.
        n_events: Number of events to simulate.
        delay: Seconds between events.
    """
    print("\n" + "="*60)
    print("  REAL-TIME LOG SIMULATION  (Ctrl+C to stop)")
    print("="*60)

    from src.utils import generate_synthetic_logs
    from src.preprocessing import preprocess
    from src.features import build_feature_matrix

    # Pre-generate a pool of events so each stream event has correct IP features
    pool = generate_synthetic_logs(n_normal=200, n_anomalous=20)
    pool_processed = preprocess(pool)
    pool_features = build_feature_matrix(pool_processed)

    for i in range(n_events):
        idx = random.randint(0, len(pool) - 1)
        row_raw = pool.iloc[[idx]]
        row_feat = pool_features.iloc[[idx]]

        label = model.predict(scaler.transform(row_feat))[0]
        score = -model.decision_function(scaler.transform(row_feat))[0]

        ip      = row_raw["ip_address"].values[0]
        ts      = row_raw["timestamp"].values[0]
        status  = row_raw["login_status"].values[0]
        reqtype = row_raw["request_type"].values[0]

        tag = "  [NORMAL ]" if label == 1 else "!  [ANOMALY]"
        alert = " <<<  ALERT: Suspicious activity detected!" if label == -1 else ""

        print(f"{tag}  IP={ip:<14}  {ts}  {status:<8}  {reqtype:<6}  score={score:.3f}{alert}")
        time.sleep(delay)

    print("\n[OK] Simulation complete.")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(data_path: str | None = None, simulate: bool = False) -> None:
    print("\n" + "="*60)
    print("  AI Log Anomaly Detection System")
    print("="*60 + "\n")

    # Step 1 -- Load or generate data
    if data_path and os.path.exists(data_path):
        df = load_dataset(data_path)
    else:
        print("[->] No dataset provided -- generating synthetic logs...")
        df = generate_synthetic_logs(n_normal=950, n_anomalous=50)
        save_dataset(df, DATA_PATH)

    print_summary(df, label="Raw Dataset")

    # Step 2 -- Preprocess
    df_processed = preprocess(df)

    # Step 3 -- Feature engineering
    X = build_feature_matrix(df_processed)

    # Step 4 -- Train model
    model, scaler = train_model(X)

    # Step 5 -- Predict and evaluate
    labels = predict(model, scaler, X)
    scores = anomaly_scores(model, scaler, X)
    df_results = evaluate(df_processed, labels)

    # Step 6 -- Save results
    results_path = "data/results.csv"
    df_results.to_csv(results_path, index=False)
    print(f"\n[OK] Full results saved to: {results_path}")

    # Step 7 -- Visualize
    plot_anomalies(X, labels, scores, save_path=PLOT_PATH)

    # Step 8 -- Persist model
    save_model(model, scaler, MODEL_DIR)

    # Step 9 -- Optional real-time simulation
    if simulate:
        simulate_realtime(model, scaler)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-Powered Log Anomaly Detection System"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to a CSV log file. If omitted, synthetic data is generated.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="After training, stream a real-time log simulation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(data_path=args.data, simulate=args.simulate)
