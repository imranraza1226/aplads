# 🤖 AI Log Anomaly Detection System

## 📌 Project Overview

This project builds an AI-powered system to detect anomalous behavior in system logs using Machine Learning. It is designed as a beginner-to-intermediate level project with real-world cybersecurity relevance.

The system processes log data, extracts meaningful features, and identifies suspicious patterns such as unusual login behavior, abnormal access times, or irregular request patterns.

---

## 🎯 Objectives

* Detect anomalies in log data using unsupervised learning
* Simulate a real-world SOC (Security Operations Center) use case
* Provide a simple dashboard for visualization
* Maintain clean, modular, production-style code

---

## 🧠 Core Concepts

* Machine Learning (Unsupervised Learning)
* Anomaly Detection
* Feature Engineering
* Data Preprocessing
* Cybersecurity Log Analysis

---

## 🏗️ Architecture

### Data Flow

1. Raw logs (CSV or synthetic data)
2. Data preprocessing
3. Feature engineering
4. Model training (Isolation Forest)
5. Prediction (anomaly detection)
6. Visualization (CLI + dashboard)

---

## 📂 Project Structure

project/
│
├── data/                  # Raw and processed datasets
├── notebooks/             # Jupyter notebooks (EDA, experimentation)
├── src/
│   ├── preprocessing.py   # Data cleaning and transformation
│   ├── features.py        # Feature engineering logic
│   ├── model.py           # ML model training and prediction
│   ├── utils.py           # Helper functions
│
├── app.py                 # Streamlit dashboard
├── main.py                # CLI entry point
├── requirements.txt       # Dependencies
└── README.md              # Project documentation

---

## ⚙️ Tech Stack

* Python 3.x
* pandas
* scikit-learn
* matplotlib
* Streamlit (optional UI)

---

## 📊 Data Requirements

Expected dataset format:

* ip_address (string)
* timestamp (datetime)
* login_status (success/failure)
* request_type (GET/POST/etc.)

If dataset is not provided:

* Generate synthetic logs with realistic anomalies

---

## 🔧 Implementation Guidelines

### 1. Preprocessing

* Handle missing/null values
* Normalize timestamps
* Extract:

  * Hour of day
  * Day of week
* Encode categorical variables

---

### 2. Feature Engineering

Create features such as:

* Login frequency per IP
* Failed login ratio
* Access time deviation

---

### 3. Model

Use Isolation Forest:

* contamination = 0.05
* Output labels:

  * -1 → anomaly
  * 1 → normal

---

### 4. Evaluation

* Count anomalies
* Display sample anomalous records
* Provide basic visualization (scatter plot)

---

### 5. Dashboard (Streamlit)

* Display dataset
* Highlight anomalies
* Add filters (IP, anomaly status)

---

## 🧾 Coding Standards

* Use modular functions (no monolithic scripts)
* Add docstrings for all functions
* Keep code readable and maintainable
* Follow PEP8 guidelines
* Avoid hardcoding values

---

## 🚀 How to Run

### Install dependencies

pip install -r requirements.txt

### Run main script

python main.py

### Run dashboard

streamlit run app.py

---

## 📈 Expected Output

* Clean dataset
* Trained anomaly detection model
* List of flagged anomalies
* Optional dashboard visualization

---

## ⚠️ Constraints

* Keep implementation beginner-friendly
* Avoid unnecessary complexity
* Do not use deep learning frameworks
* Focus on functionality over perfection

---

## 🔥 Future Enhancements

* Real-time log streaming
* Integration with SIEM tools
* Alerting system (email/SMS)
* Model explainability (why anomaly detected)

---

## 🧠 Developer Notes

This project is intended to simulate a real-world AI + cybersecurity workflow. Prioritize clarity, correctness, and simplicity over advanced optimizations.

When modifying or extending:

* Preserve modular structure
* Maintain separation of concerns
* Document all changes

---