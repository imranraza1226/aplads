"""
Data preprocessing for the AI Log Anomaly Detection System.

Supports two input formats:
  1. Synthetic / custom logs  : ip_address, timestamp, login_status, request_type
  2. Windows Event Logs       : Level, Date and Time, Source, Event ID, Task Category
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

SYNTHETIC_COLS  = {"ip_address", "timestamp", "login_status", "request_type"}
WINDOWS_EV_COLS = {"Level", "Date and Time", "Source", "Event ID"}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip all whitespace variants (incl. non-breaking spaces) from column names."""
    import re
    df.columns = [re.sub(r"[\s\xa0\ufeff]+", " ", c).strip() for c in df.columns]
    return df


def _fuzzy_rename(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Rename columns using case-insensitive matching so minor typos or
    encoding quirks in the CSV header don't silently break the rename.
    """
    col_lower = {c.lower(): c for c in df.columns}
    resolved = {}
    for target_lower, new_name in {k.lower(): v for k, v in mapping.items()}.items():
        actual = col_lower.get(target_lower)
        if actual:
            resolved[actual] = new_name
    return df.rename(columns=resolved)


def detect_format(df: pd.DataFrame) -> str:
    """
    Detect the log format of the DataFrame.

    Returns:
        'synthetic' if it looks like the custom/synthetic format.
        'windows'   if it looks like a Windows Event Log export.

    Raises:
        ValueError if neither format is recognised.
    """
    df = _normalise_columns(df)
    cols_lower = {c.lower() for c in df.columns}
    if {c.lower() for c in SYNTHETIC_COLS}.issubset(cols_lower):
        return "synthetic"
    if {c.lower() for c in WINDOWS_EV_COLS}.issubset(cols_lower):
        return "windows"
    raise ValueError(
        f"Unrecognised log format.\n"
        f"  Found columns    : {sorted(df.columns.tolist())}\n"
        f"  Expected (option 1 - synthetic): {sorted(SYNTHETIC_COLS)}\n"
        f"  Expected (option 2 - Windows Event Log): {sorted(WINDOWS_EV_COLS)}"
    )


# ---------------------------------------------------------------------------
# Windows Event Log adapter
# ---------------------------------------------------------------------------

# Map Windows severity levels to a binary success/failure analogue.
# Error and Critical are treated as "failure" events; everything else is "success".
_LEVEL_TO_STATUS = {
    "information": "success",
    "verbose":     "success",
    "warning":     "failure",
    "error":       "failure",
    "critical":    "failure",
    "audit success": "success",
    "audit failure": "failure",
}


def adapt_windows_event_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a Windows Event Log export into the internal schema:
        ip_address    <- Source          (event source / component name)
        timestamp     <- Date and Time   (parsed datetime)
        login_status  <- Level           (Error/Critical/Warning -> failure, else success)
        request_type  <- Event ID        (numeric event identifier, cast to string)

    Also preserves 'Task Category' as an extra context column.

    Args:
        df: Raw Windows Event Log DataFrame.

    Returns:
        DataFrame with standardised column names.
    """
    df = df.copy()
    df = _normalise_columns(df)

    # Rename columns to internal schema (case-insensitive, tolerates encoding quirks)
    df = _fuzzy_rename(df, {
        "Source":        "ip_address",
        "Date and Time": "timestamp",
        "Event ID":      "request_type",
    })

    missing = {"ip_address", "timestamp", "request_type"} - set(df.columns)
    if missing:
        raise ValueError(
            f"Windows Event Log adapter could not map columns: {missing}.\n"
            f"Actual columns found: {sorted(df.columns.tolist())}"
        )

    # Map Level -> login_status (case-insensitive)
    df["login_status"] = (
        df["Level"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(_LEVEL_TO_STATUS)
        .fillna("success")   # unknown levels treated as success
    )

    # Keep Event ID as a string category (e.g. "4625", "4624")
    df["request_type"] = df["request_type"].astype(str)

    # Drop the original Level column (already captured in login_status)
    df = df.drop(columns=["Level"], errors="ignore")

    print(f"[OK] Windows Event Log adapted: {len(df)} records.")
    print(f"     Unique sources  : {df['ip_address'].nunique()}")
    print(f"     Unique event IDs: {df['request_type'].nunique()}")
    print(f"     Failure events  : {(df['login_status'] == 'failure').sum()}")

    return df


# ---------------------------------------------------------------------------
# Core preprocessing steps (shared by both formats)
# ---------------------------------------------------------------------------

def load_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that required columns exist and drop rows with critical nulls.

    Args:
        df: Raw log DataFrame (already normalised to internal schema).

    Returns:
        Validated DataFrame.
    """
    required_columns = {"ip_address", "timestamp", "login_status", "request_type"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    before = len(df)
    df = df.dropna(subset=list(required_columns))
    dropped = before - len(df)
    if dropped > 0:
        print(f"[!] Dropped {dropped} rows with missing values in required columns.")

    if len(df) == 0:
        raise ValueError(
            "Dataset is empty after dropping rows with missing required columns. "
            "Check that your CSV has data in: " + str(sorted(required_columns))
        )

    return df.reset_index(drop=True)


def _parse_timestamp_column(series: pd.Series) -> pd.Series:
    """
    Try multiple strategies to parse timestamps from Windows Event Viewer exports.

    Common Windows Event Viewer date formats:
      - "4/19/2025 10:15:00 AM"   (US locale, 12-hour)
      - "4/19/2025 14:15:00"      (US locale, 24-hour)
      - "19/04/2025 10:15:00"     (UK/EU locale, 24-hour)
      - "2025-04-19 10:15:00"     (ISO format)

    Tries strategies in order, picks the one that parses the most rows.
    """
    EXPLICIT_FORMATS = [
        "%m/%d/%Y %I:%M:%S %p",   # 4/19/2025 10:15:00 AM  (US 12-hour)  ← most common
        "%m/%d/%Y %H:%M:%S",      # 4/19/2025 14:15:00     (US 24-hour)
        "%d/%m/%Y %H:%M:%S",      # 19/04/2025 10:15:00    (EU 24-hour)
        "%d/%m/%Y %I:%M:%S %p",   # 19/04/2025 10:15:00 AM (EU 12-hour)
        "%Y-%m-%d %H:%M:%S",      # 2025-04-19 10:15:00    (ISO)
        "%Y/%m/%d %H:%M:%S",      # 2025/04/19 10:15:00
    ]

    best_parsed = pd.Series([pd.NaT] * len(series), dtype="datetime64[ns]")
    best_ok = 0

    # Strategy 1: pandas "mixed" mode — infers format per row (pandas >= 2.0)
    try:
        candidate = pd.to_datetime(series, format="mixed", dayfirst=False, errors="coerce")
        n_ok = candidate.notna().sum()
        if n_ok > best_ok:
            best_parsed, best_ok = candidate, n_ok
    except Exception:
        pass

    # Strategy 2: explicit format list — try each and keep the best
    for fmt in EXPLICIT_FORMATS:
        if best_ok == len(series):
            break  # already perfect
        candidate = pd.to_datetime(series, format=fmt, errors="coerce")
        n_ok = candidate.notna().sum()
        if n_ok > best_ok:
            best_parsed, best_ok = candidate, n_ok

    # Strategy 3: pandas auto-detect fallback
    if best_ok < len(series) * 0.5:
        candidate = pd.to_datetime(series, errors="coerce")
        n_ok = candidate.notna().sum()
        if n_ok > best_ok:
            best_parsed, best_ok = candidate, n_ok

    return best_parsed


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the timestamp column and extract time-based features.

    Adds:
        - hour_of_day  (0-23)
        - day_of_week  (0=Monday, 6=Sunday)
        - is_weekend   (1 if Saturday/Sunday, else 0)
        - is_off_hours (1 if outside 8am-6pm, else 0)
    """
    df = df.copy()
    raw_sample = df["timestamp"].dropna().iloc[0] if df["timestamp"].notna().any() else "N/A"
    print(f"[->] Sample timestamp value: '{raw_sample}'")

    df["timestamp"] = _parse_timestamp_column(df["timestamp"])

    n_failed = df["timestamp"].isna().sum()
    if n_failed > 0:
        print(f"[!] {n_failed}/{len(df)} timestamps could not be parsed and will be dropped.")
        df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(
            "All timestamps failed to parse. "
            f"Sample raw value was: '{raw_sample}'. "
            "Please check your CSV date format."
        )

    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["is_off_hours"] = (~df["hour_of_day"].between(8, 18)).astype(int)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode categorical columns: ip_address, login_status, request_type.

    Encoded columns added:
        - ip_encoded
        - login_encoded   (success=1, failure=0)
        - request_encoded
    """
    df = df.copy()

    df["login_encoded"] = df["login_status"].map({"success": 1, "failure": 0})
    if df["login_encoded"].isnull().any():
        le = LabelEncoder()
        df["login_encoded"] = le.fit_transform(df["login_status"].astype(str))

    le_ip = LabelEncoder()
    df["ip_encoded"] = le_ip.fit_transform(df["ip_address"].astype(str))

    le_req = LabelEncoder()
    df["request_encoded"] = le_req.fit_transform(df["request_type"].astype(str))

    return df


# ---------------------------------------------------------------------------
# Public pipeline entry point
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline. Auto-detects log format.

    Steps:
        1. Detect format (synthetic vs Windows Event Log)
        2. Adapt Windows Event Log columns if needed
        3. Validate required columns
        4. Parse timestamps and extract time features
        5. Encode categorical variables

    Args:
        df: Raw log DataFrame (any supported format).

    Returns:
        Fully preprocessed DataFrame ready for feature engineering.
    """
    df = _normalise_columns(df)
    fmt = detect_format(df)
    print(f"[->] Detected log format: {fmt}")

    if fmt == "windows":
        df = adapt_windows_event_log(df)

    print("[->] Validating data...")
    df = load_and_validate(df)

    print("[->] Parsing timestamps...")
    df = parse_timestamps(df)

    print("[->] Encoding categorical variables...")
    df = encode_categoricals(df)

    print(f"[OK] Preprocessing complete. Shape: {df.shape}")
    return df
