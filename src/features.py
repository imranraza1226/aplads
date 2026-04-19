"""
Feature engineering for the AI Log Anomaly Detection System.

Builds behavioral features per source (IP or Windows Event Source) that help
the model identify suspicious patterns regardless of the input log format.
"""

import pandas as pd
import numpy as np


def compute_ip_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-source aggregate behavioral features and merge them back.

    Features created:
        - login_freq        : total events from this source
        - failure_ratio     : fraction of failure/error-level events (0.0 - 1.0)
        - off_hours_ratio   : fraction of events outside business hours
        - unique_req_types  : number of distinct event types / request types used

    Args:
        df: Preprocessed DataFrame with ip_address, login_encoded, is_off_hours.

    Returns:
        DataFrame with four new per-source feature columns.
    """
    ip_stats = (
        df.groupby("ip_address")
        .agg(
            login_freq=("login_encoded", "count"),
            failure_ratio=("login_encoded", lambda x: 1 - x.mean()),
            off_hours_ratio=("is_off_hours", "mean"),
            unique_req_types=("request_type", "nunique"),
        )
        .reset_index()
    )

    df = df.merge(ip_stats, on="ip_address", how="left")
    return df


def compute_time_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute how far each event's hour deviates from that source's mean activity hour.

    A large deviation suggests the source is active at unusual times compared to
    its own historical pattern.

    Feature created:
        - hour_deviation : abs(hour_of_day - mean_hour_for_this_source)

    Args:
        df: DataFrame with ip_address and hour_of_day columns.

    Returns:
        DataFrame with hour_deviation column added.
    """
    mean_hour = df.groupby("ip_address")["hour_of_day"].transform("mean")
    df = df.copy()
    df["hour_deviation"] = (df["hour_of_day"] - mean_hour).abs()
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all feature engineering steps and return the final numeric feature matrix.

    Args:
        df: Preprocessed DataFrame (output of preprocessing.preprocess()).

    Returns:
        DataFrame with the following feature columns:
        hour_of_day, day_of_week, is_weekend, is_off_hours,
        login_encoded, ip_encoded, request_encoded,
        login_freq, failure_ratio, off_hours_ratio, unique_req_types,
        hour_deviation
    """
    print("[->] Computing per-source behavioral features...")
    df = compute_ip_features(df)

    print("[->] Computing hour deviation feature...")
    df = compute_time_deviation(df)

    feature_columns = [
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "is_off_hours",
        "login_encoded",
        "ip_encoded",
        "request_encoded",
        "login_freq",
        "failure_ratio",
        "off_hours_ratio",
        "unique_req_types",
        "hour_deviation",
    ]

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Feature engineering produced missing columns: {missing}")

    print(f"[OK] Feature matrix ready. Features: {feature_columns}")
    return df[feature_columns]
