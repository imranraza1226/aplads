"""
Utility functions for the AI Log Anomaly Detection System.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os


def generate_synthetic_logs(n_normal: int = 950, n_anomalous: int = 50, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic log dataset with embedded anomalies.

    Args:
        n_normal: Number of normal log entries to generate.
        n_anomalous: Number of anomalous log entries to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: ip_address, timestamp, login_status, request_type
    """
    np.random.seed(seed)
    random.seed(seed)

    # --- Normal traffic ---
    normal_ips = [f"192.168.1.{i}" for i in range(1, 51)]  # internal IPs
    normal_request_types = ["GET", "POST", "GET", "GET", "PUT"]  # GET-heavy
    base_time = datetime(2024, 1, 1, 8, 0, 0)  # business hours start

    normal_records = []
    for _ in range(n_normal):
        # Business hours: 8am - 6pm, Mon-Fri
        days_offset = np.random.randint(0, 90)
        hour = np.random.choice(range(8, 19), p=[0.05, 0.10, 0.12, 0.12, 0.10, 0.10, 0.10, 0.08, 0.08, 0.08, 0.07])
        minute = np.random.randint(0, 60)
        ts = base_time + timedelta(days=int(days_offset), hours=int(hour - 8), minutes=int(minute))

        normal_records.append({
            "ip_address": random.choice(normal_ips),
            "timestamp": ts,
            "login_status": np.random.choice(["success", "failure"], p=[0.90, 0.10]),
            "request_type": random.choice(normal_request_types),
            "is_anomaly_ground_truth": 0
        })

    # --- Anomalous traffic ---
    anomaly_ips = [f"10.0.0.{i}" for i in range(1, 11)]  # suspicious external IPs
    anomaly_records = []

    anomaly_patterns = [
        # Brute-force: many failures from same IP at odd hours
        {"login_status": "failure", "request_type": "POST", "hour_range": (0, 5)},
        # Data exfiltration: many GETs at night
        {"login_status": "success", "request_type": "GET",  "hour_range": (1, 4)},
        # Scanning: DELETE/PUT at odd hours
        {"login_status": "failure", "request_type": "DELETE", "hour_range": (2, 6)},
    ]

    for _ in range(n_anomalous):
        pattern = random.choice(anomaly_patterns)
        days_offset = np.random.randint(0, 90)
        hour = np.random.randint(*pattern["hour_range"])
        minute = np.random.randint(0, 60)
        ts = base_time + timedelta(days=int(days_offset), hours=int(hour), minutes=int(minute))

        anomaly_records.append({
            "ip_address": random.choice(anomaly_ips),
            "timestamp": ts,
            "login_status": pattern["login_status"],
            "request_type": pattern["request_type"],
            "is_anomaly_ground_truth": 1
        })

    df = pd.DataFrame(normal_records + anomaly_records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def save_dataset(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[OK] Dataset saved to: {path}")


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a log dataset from CSV.

    Args:
        path: Path to the CSV file.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    print(f"[OK] Loaded {len(df)} records from {path}")
    return df


def print_summary(df: pd.DataFrame, label: str = "Dataset") -> None:
    """Print a concise summary of a DataFrame."""
    print(f"\n{'='*50}")
    print(f"  {label} Summary")
    print(f"{'='*50}")
    print(f"  Rows:    {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Missing: {df.isnull().sum().sum()}")
    print(f"{'='*50}\n")
