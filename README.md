# AI-Powered Log Anomaly Detection System

An ML-based system that analyzes system logs and automatically flags suspicious activity using **Isolation Forest** — an unsupervised anomaly detection algorithm.

---

## Project Overview

This system simulates a real-world **Security Operations Center (SOC)** workflow:

1. Raw log data is ingested (or synthetically generated)
2. Timestamps are parsed and behavioral features are engineered per IP
3. An Isolation Forest model is trained on the feature matrix
4. Anomalous records are flagged, summarized, and visualized
5. A Streamlit dashboard provides an interactive UI

Typical anomalies detected:
- **Brute-force attacks** — many login failures from one IP
- **Off-hours access** — logins at 2–5 AM
- **Data exfiltration patterns** — mass GET requests at night
- **Port/endpoint scanning** — unusual DELETE/PUT bursts

---

## Project Structure

```
aplads-app/
│
├── data/                  # Auto-generated datasets and model artifacts
│   ├── logs.csv           # Raw log data (synthetic or uploaded)
│   ├── results.csv        # Prediction results
│   ├── anomaly_plot.png   # Saved visualization
│   └── model/             # Persisted Isolation Forest + scaler
│
├── src/
│   ├── utils.py           # Data generation, loading, saving helpers
│   ├── preprocessing.py   # Cleaning, timestamp parsing, encoding
│   ├── features.py        # IP-level behavioral feature engineering
│   └── model.py           # Isolation Forest training, prediction, plots
│
├── app.py                 # Streamlit dashboard
├── main.py                # CLI entry point
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd aplads-app

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### CLI (Command Line)

```bash
# Generate synthetic data, train model, detect anomalies, show plot
python main.py

# Use your own CSV dataset
python main.py --data path/to/your/logs.csv

# Add real-time log streaming simulation after training
python main.py --simulate
```

### Streamlit Dashboard

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## Dataset Formats

The system auto-detects one of two supported formats.

### Option 1 — Custom / Synthetic logs

| Column         | Type     | Example                   |
|----------------|----------|---------------------------|
| `ip_address`   | string   | `192.168.1.42`            |
| `timestamp`    | datetime | `2024-01-15 14:32:00`     |
| `login_status` | string   | `success` / `failure`     |
| `request_type` | string   | `GET` / `POST` / `DELETE` |

### Option 2 — Windows Event Log export

Export directly from **Windows Event Viewer → Save All Events As → CSV**.

| Column          | Type   | Example                    |
|-----------------|--------|----------------------------|
| `Level`         | string | `Information` / `Error`    |
| `Date and Time` | string | `4/19/2025 10:15:00 AM`    |
| `Source`        | string | `Service Control Manager`  |
| `Event ID`      | number | `4625`                     |
| `Task Category` | string | `(optional)`               |

**Encoding note:** Windows Event Viewer CSV exports are often UTF-16 or Windows-1252.
The dashboard handles this automatically. If upload fails, open the file in Notepad,
choose **File → Save As → Encoding: UTF-8**, then re-upload.

---

## Dashboard Features

- **Upload any CSV** — custom logs or Windows Event Viewer exports
- **Contamination slider** — tune the expected anomaly percentage (1–20%)
- **Run Detection button** — re-run the pipeline with current data and settings
- **Filters** — filter results by anomaly status, IP address, or login status
- **Download** — export filtered results as CSV
- **Visualizations** — score distribution histogram, scatter plot, top anomalous IPs
- **Real-time simulation** — stream synthetic events with live anomaly scoring

---

## Example Output

```
==================================================
  AI Log Anomaly Detection System
==================================================

[✓] Dataset saved to: data/logs.csv
[✓] Preprocessing complete. Shape: (1000, 14)
[✓] Feature matrix ready.
[✓] Model training complete.

==================================================
  Detection Results
==================================================
  Total records  : 1000
  Normal         : 949 (94.9%)
  Anomalies      : 51  (5.1%)
==================================================

  Sample anomalous records:
  ip_address       timestamp             login_status  request_type
  10.0.0.3         2024-02-14 02:14:00   failure       POST
  10.0.0.7         2024-03-01 03:45:00   failure       DELETE
  10.0.0.1         2024-01-22 01:30:00   success       GET
```

---

## Tech Stack

| Component         | Library              |
|-------------------|----------------------|
| Data manipulation | pandas, numpy        |
| ML model          | scikit-learn         |
| Visualization     | matplotlib           |
| Dashboard         | Streamlit            |
| Model persistence | joblib               |

---

## Future Enhancements

- Real-time log streaming via Kafka/Redis
- Integration with SIEM tools (Splunk, Elastic)
- Email/SMS alerting on anomaly detection
- Model explainability (SHAP values)
- Scheduled retraining pipeline
