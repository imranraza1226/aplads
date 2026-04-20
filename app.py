"""
Streamlit dashboard for the AI Log Anomaly Detection System.

Run with:
    streamlit run app.py
"""

import os
import time
import random

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.utils import generate_synthetic_logs, save_dataset
from src.preprocessing import preprocess
from src.features import build_feature_matrix
from src.model import train_model, predict, anomaly_scores, save_model


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Log Anomaly Detector",
    page_icon="🔍",
    layout="wide",
)

DATA_PATH = "data/logs.csv"
MODEL_DIR = "data/model"


# ---------------------------------------------------------------------------
# Helpers (cached for performance)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_or_generate_data() -> pd.DataFrame:
    """Return the dataset, generating synthetic data if needed."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    df = generate_synthetic_logs(n_normal=950, n_anomalous=50)
    save_dataset(df, DATA_PATH)
    return df


@st.cache_data(show_spinner=False)
def run_detection(df: pd.DataFrame, contamination: float = 0.05):
    """Full ML pipeline: preprocess -> features -> train -> predict."""
    df_processed = preprocess(df)
    X = build_feature_matrix(df_processed)
    model, scaler = train_model(X, contamination=contamination)
    labels = predict(model, scaler, X)
    scores = anomaly_scores(model, scaler, X)
    save_model(model, scaler, MODEL_DIR)
    return df_processed, X, model, scaler, labels, scores


def build_results_df(df_processed: pd.DataFrame, labels: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    display_cols = [c for c in ["ip_address", "timestamp", "login_status", "request_type",
                                "hour_of_day", "day_of_week", "is_off_hours", "failure_ratio"] if c in df_processed.columns]
    df = df_processed[display_cols].copy()
    df["anomaly_label"] = labels
    df["anomaly_score"] = scores.round(4)
    df["status"] = df["anomaly_label"].map({-1: "🔴 Anomaly", 1: "🟢 Normal"})
    return df


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main():
    # ── Header ──────────────────────────────────────────────────────────────
    st.title("🔍 AI Log Anomaly Detector")
    st.markdown(
        "Unsupervised anomaly detection for system logs using **Isolation Forest**. "
        "Upload your own CSV or use the built-in synthetic dataset."
    )
    st.divider()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        uploaded = st.file_uploader(
            "Upload log CSV (optional)",
            type=["csv"],
            help=(
                "Supported formats:\n"
                "1. Custom/synthetic: ip_address, timestamp, login_status, request_type\n"
                "2. Windows Event Log export: Level, Date and Time, Source, Event ID, Task Category"
            ),
        )

        contamination = st.slider(
            "Contamination (% anomalies expected)",
            min_value=1, max_value=20, value=5, step=1,
            help="Set to ~5% for most real datasets.",
        ) / 100.0

        run_btn = st.button("▶  Run Detection", use_container_width=True, type="primary")
        run_sim = st.checkbox("Show real-time simulation", value=False)
        st.divider()
        st.caption("AI Log Anomaly Detection System")

    # ── Load data ────────────────────────────────────────────────────────────
    with st.spinner("Loading dataset..."):
        if uploaded is not None:
            df_raw = None
            # Windows Event Viewer exports are often UTF-16 tab-separated;
            # try encoding + separator combinations until one works.
            for enc, sep in [
                ("utf-8",     ","),
                ("utf-8-sig", ","),   # UTF-8 with BOM
                ("utf-16",    "\t"),  # Windows Event Viewer default
                ("utf-16",    ","),
                ("cp1252",    ","),
                ("cp1252",    "\t"),
                ("latin-1",   ","),
                ("latin-1",   "\t"),
            ]:
                try:
                    uploaded.seek(0)
                    df_raw = pd.read_csv(uploaded, encoding=enc, sep=sep)
                    if df_raw.shape[1] > 1:   # at least 2 columns → valid parse
                        break
                    df_raw = None
                except Exception:
                    continue
            if df_raw is None:
                st.error(
                    "Could not decode the CSV file. "
                    "Open it in Excel → Save As → CSV UTF-8 (comma delimited), then re-upload."
                )
                st.stop()
            from src.preprocessing import detect_format
            try:
                fmt = detect_format(df_raw)
                fmt_label = "Windows Event Log" if fmt == "windows" else "Custom/Synthetic"
                st.success(f"Loaded {len(df_raw):,} records — detected format: **{fmt_label}**")
            except ValueError as e:
                st.error(str(e))
                st.stop()
        else:
            df_raw = load_or_generate_data()
            st.info(f"Using synthetic dataset — {len(df_raw):,} records.")

    # Clear cache when user explicitly re-runs so new settings take effect
    if run_btn:
        run_detection.clear()

    # ── Run pipeline ─────────────────────────────────────────────────────────
    with st.spinner("Running anomaly detection..."):
        try:
            df_processed, X, model, scaler, labels, scores = run_detection(
                df_raw, contamination=contamination
            )
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            st.stop()

    df_results = build_results_df(df_processed, labels, scores)

    # ── KPI cards ────────────────────────────────────────────────────────────
    total      = len(df_results)
    n_anomaly  = (labels == -1).sum()
    n_normal   = (labels == 1).sum()
    anomaly_pct = n_anomaly / total * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records",  f"{total:,}")
    col2.metric("Normal",         f"{n_normal:,}",  delta=f"{100 - anomaly_pct:.1f}%")
    col3.metric("Anomalies",      f"{n_anomaly:,}", delta=f"{anomaly_pct:.1f}%", delta_color="inverse")
    col4.metric("Detection Rate", f"{anomaly_pct:.2f}%")

    st.divider()

    # ── Filters ───────────────────────────────────────────────────────────────
    st.subheader("🔎 Filter Results")
    fcol1, fcol2, fcol3 = st.columns(3)

    with fcol1:
        status_filter = st.selectbox(
            "Anomaly Status",
            options=["All", "🔴 Anomaly", "🟢 Normal"],
        )
    with fcol2:
        all_ips = ["All"] + sorted(df_results["ip_address"].unique().tolist())
        ip_filter = st.selectbox("IP Address", options=all_ips)
    with fcol3:
        if "login_status" in df_results.columns:
            all_logins = ["All"] + sorted(df_results["login_status"].unique().tolist())
            login_filter = st.selectbox("Login Status", options=all_logins)
        else:
            login_filter = "All"

    # Apply filters
    filtered = df_results.copy()
    if status_filter != "All":
        filtered = filtered[filtered["status"] == status_filter]
    if ip_filter != "All":
        filtered = filtered[filtered["ip_address"] == ip_filter]
    if login_filter != "All" and "login_status" in filtered.columns:
        filtered = filtered[filtered["login_status"] == login_filter]

    # ── Data table ────────────────────────────────────────────────────────────
    st.subheader(f"📋 Log Records ({len(filtered):,} shown)")

    # Highlight anomaly rows red
    def highlight_anomaly(row):
        color = "background-color: #fde8e8" if row["anomaly_label"] == -1 else ""
        return [color] * len(row)

    st.dataframe(
        filtered.style.apply(highlight_anomaly, axis=1),
        use_container_width=True,
        height=400,
    )

    # Download button
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️  Download filtered results",
        data=csv_bytes,
        file_name="anomaly_results.csv",
        mime="text/csv",
    )

    st.divider()

    # ── Visualizations ────────────────────────────────────────────────────────
    st.subheader("📊 Visualizations")

    vcol1, vcol2 = st.columns(2)

    with vcol1:
        st.markdown("**Anomaly Score Distribution**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.hist(scores[labels == 1],  bins=40, color="#2ecc71", alpha=0.7, label="Normal")
        ax1.hist(scores[labels == -1], bins=40, color="#e74c3c", alpha=0.7, label="Anomaly")
        ax1.set_xlabel("Anomaly Score")
        ax1.set_ylabel("Count")
        ax1.legend()
        ax1.set_title("Score Distribution")
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with vcol2:
        st.markdown("**Hour of Day vs Failure Ratio**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        colors_plot = ["#e74c3c" if l == -1 else "#2ecc71" for l in labels]
        if "failure_ratio" in X.columns:
            ax2.scatter(X["hour_of_day"], X["failure_ratio"], c=colors_plot, alpha=0.5, s=20)
            ax2.set_xlabel("Hour of Day")
            ax2.set_ylabel("Failure Ratio")
        else:
            ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], c=colors_plot, alpha=0.5, s=20)
            ax2.set_xlabel(X.columns[0])
            ax2.set_ylabel(X.columns[1])
        patches = [mpatches.Patch(color="#e74c3c", label="Anomaly"),
                   mpatches.Patch(color="#2ecc71", label="Normal")]
        ax2.legend(handles=patches)
        ax2.set_title("Scatter Plot")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # ── Top anomalous IPs bar chart ───────────────────────────────────────────
    st.markdown("**Top Anomalous IP Addresses**")
    top_ips = (
        df_results[df_results["anomaly_label"] == -1]
        .groupby("ip_address")
        .size()
        .sort_values(ascending=False)
        .head(10)
    )
    if not top_ips.empty:
        fig3, ax3 = plt.subplots(figsize=(10, 3))
        ax3.barh(top_ips.index[::-1], top_ips.values[::-1], color="#e74c3c")
        ax3.set_xlabel("Number of Anomalous Events")
        ax3.set_title("Top 10 IPs by Anomaly Count")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    st.divider()

    # ── Real-time simulation ───────────────────────────────────────────────────
    if run_sim:
        st.subheader("⚡ Real-Time Log Simulation")
        st.markdown("Streaming synthetic log events with live anomaly detection:")

        log_placeholder = st.empty()
        alert_placeholder = st.empty()

        pool = generate_synthetic_logs(n_normal=200, n_anomalous=20)
        pool_processed = preprocess(pool)
        pool_features = build_feature_matrix(pool_processed)

        log_lines = []
        for _ in range(30):
            idx = random.randint(0, len(pool) - 1)
            row_raw  = pool.iloc[[idx]]
            row_feat = pool_features.iloc[[idx]]

            row_scaled = scaler.transform(row_feat)
            lbl   = model.predict(row_scaled)[0]
            score = -model.decision_function(row_scaled)[0]

            ip      = row_raw["ip_address"].values[0]
            ts      = str(row_raw["timestamp"].values[0])[:19]
            status  = row_raw["login_status"].values[0]
            reqtype = row_raw["request_type"].values[0]

            if lbl == -1:
                line = f"⚠️  **ANOMALY** | `{ip}` | {ts} | {status} | {reqtype} | score={score:.3f}"
            else:
                line = f"✅  Normal     | `{ip}` | {ts} | {status} | {reqtype} | score={score:.3f}"

            log_lines.append(line)
            log_placeholder.markdown("\n\n".join(log_lines[-15:]))
            time.sleep(0.3)

        st.success("Simulation complete.")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.caption("AI Log Anomaly Detection System · Built with Isolation Forest + Streamlit")


if __name__ == "__main__":
    main()
