
# 🛡️ FraudGuard

### AI-Powered Credit Card Fraud Detection & Transaction Intelligence Platform

FraudGuard is an interactive machine-learning platform designed to detect potentially fraudulent credit card transactions, assess transaction risk, investigate suspicious activity, and explore transaction patterns through an analyst-style dashboard.

The platform combines a Random Forest fraud detection model with interactive analytics and investigation workflows to provide a practical fraud-monitoring experience.

---

## 🚀 Live Demo

🌐 **FraudGuard:**  
(https://fraudguard-dmtcrkknc8amwsmpxlkrh9.streamlit.app/)

---

## ✨ Key Features

### 🔴 Risk Overview
- Real-time fraud risk overview
- Transaction and fraud-signal KPIs
- Fraud-rate monitoring
- Risk distribution visualization
- Interactive transaction analytics

### 📡 Live Monitor
- Simulated transaction replay
- Transaction-by-transaction fraud scoring
- Fraud probability calculation
- Risk classification
- High-risk transaction alerts
- Transaction monitoring dashboard

### 🔎 Investigation Workspace
- Investigate suspicious transactions
- Transaction-level risk assessment
- Model input signals
- Case-oriented investigation workflow
- Review and clear investigation actions

### 📦 Bulk Scanner
- Upload transaction datasets
- Multi-stage scanning workflow
- Data ingestion
- Feature engineering
- Machine-learning prediction
- Risk filtering
- Fraud signal identification

### 🧠 Intelligence Workspace
- Interactive transaction filtering
- Location analysis
- Transaction-type analysis
- Monthly trends
- Merchant signals
- Day-of-week analysis
- Amount distribution
- Dynamic key observations

### 📊 Data Explorer
- Explore transaction-level data
- Filter and inspect transaction records
- Analyze transaction patterns

### 🧪 Model Lab
- Explore the fraud detection model
- Review model-related information
- Understand the features used for prediction

---

## 🤖 Machine Learning

FraudGuard uses a **Random Forest Classifier** to estimate the probability that a transaction is fraudulent.

The model evaluates transaction information including:

- Transaction amount
- Transaction month
- Transaction day
- Transaction day of week
- Transaction type
- Transaction location
- Merchant frequency

The model produces a fraud probability using:

```python
model.predict_proba(features)[:, 1]
