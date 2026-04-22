# 🛡️ Context-Aware Cyber Threat Forecasting

> **Predict the next attack before it happens** — a production-ready ML pipeline that classifies live network traffic *and* forecasts future threat states using a hybrid XGBoost → LSTM → Adaptive Markov architecture.

---

## 🔍 What This Project Does

Most intrusion detection systems react. This one **anticipates**.

Given a window of network traffic, the system:
1. **Classifies** each flow in real-time (Normal / DoS / DDoS / Reconnaissance)
2. **Forecasts** what threat class is likely to appear *next*
3. **Quantifies uncertainty** using Monte Carlo Dropout
4. **Weighs contextual signals** (time-of-day, device type, geolocation, threat history)
5. **Triggers tiered alerts** (🔴 HIGH / 🟠 MEDIUM / 🟢 LOW) with risk scores

---

## 🏗️ Architecture

```
Network Traffic
      │
      ▼
┌─────────────────────┐
│  XGBoost Classifier │  ← 32 flow features, calibrated probabilities
│  (Isotonic Calib.)  │    Accuracy: 99.89%  │  F1: 0.9993
└────────┬────────────┘
         │  Probability sequences (window = 10 steps)
         ▼
┌─────────────────────┐
│   LSTM Forecaster   │  ← Temporal sequence learning
│  (MC Dropout ×30)   │    Val Accuracy: 94.8%  │  Uncertainty-aware
└────────┬────────────┘
         │  Posterior distribution over next state
         ▼
┌──────────────────────────────┐
│  Adaptive Markov v3          │  ← Blends empirical transitions +
│  + Escalation Prior          │    cyber kill-chain domain priors
│  + Context Engine            │    + 5 real-time context signals
└────────┬─────────────────────┘
         │
         ▼
   🎯 Next-State Forecast + Risk Score + Alert Level
```

---

## 📊 Model Performance

| Model | Metric | Score |
|---|---|---|
| XGBoost (calibrated) | Test Accuracy | **99.89%** |
| XGBoost | Macro F1 (5-fold CV) | **0.9993 ± 0.0002** |
| LSTM | Validation Accuracy | **94.8%** |
| Full Pipeline | Forecast Accuracy (Normal scenario) | **100%** |

**Dataset:** Bot-IoT — 3,668,522 flows × 46 features (DDoS, DoS, Reconnaissance, Normal)  
**Class imbalance handled with:** SMOTE-based balanced sampling

---

## 🚀 Demo Scenarios

The live demo (`demo1.py`) ships with 5 pre-built scenarios:

| Scenario | Description |
|---|---|
| 🟢 A — All Normal | Baseline healthy traffic |
| 🟠 B — Slow Escalation | Recon gradually escalates to DoS |
| 🔴 C — Sudden DDoS | Abrupt volumetric attack |
| 🔵 D — Stealth Recon | Low-and-slow reconnaissance sweep |
| 🟣 E — APT Simulation | Multi-stage advanced persistent threat |

---

## 🗂️ Repo Structure

```
├── demo1.py                        # Interactive demo (4 modes)
├── CyberThreat_FYP_Final_Clean.ipynb  # Full training notebook
├── fyp_saved_models/
│   ├── xgb_calibrated.pkl          # Trained XGBoost + isotonic calibration
│   ├── lstm_model.keras            # Trained LSTM forecaster
│   ├── scaler.pkl                  # Feature scaler
│   ├── label_encoder.pkl           # Class label encoder
│   ├── class_names.json            # [DDoS, DoS, Normal, Reconnaissance]
│   └── feature_cols.json           # 32 selected flow features
├── viz_01_raw_distribution.png     # Class distribution (raw)
├── viz_05_correlation_heatmap.png  # Feature correlation heatmap
├── viz_09_xgb_confusion.png        # XGBoost confusion matrix
├── viz_12_lstm_training.png        # LSTM training curves
├── viz_14_threat_forecast.png      # Markov forecast output
├── dashboard_*.png                 # Live dashboard screenshots
└── classification_basis.svg        # Architecture diagrams
```

---

## ⚡ Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/cyber-threat-forecasting.git
cd cyber-threat-forecasting

# Install dependencies
pip install -r requirements.txt

# Run the interactive demo
python demo1.py
```

**Demo Modes:**
```
1 → Scenario Sweep     (all 5 pre-built scenarios)
2 → Real Samples       (draws from actual Bot-IoT data)
3 → Interactive        (enter your own feature values)
4 → Stress Test        (edge cases & uncertainty analysis)
```

---

## 🧰 Tech Stack

| Category | Libraries |
|---|---|
| ML / Classification | `XGBoost`, `scikit-learn` (isotonic calibration, SMOTE) |
| Deep Learning | `TensorFlow / Keras` (LSTM, MC Dropout) |
| Data Processing | `NumPy`, `Pandas` |
| Visualisation | `Matplotlib`, `Seaborn` |
| Serialisation | `joblib` |

---

## 🧠 Key Design Decisions

**Why XGBoost → LSTM (not end-to-end)?**  
XGBoost gives calibrated *probability vectors* per flow. The LSTM learns patterns over sequences of these probability vectors — a richer temporal signal than raw features.

**Why Adaptive Markov v3?**  
Pure neural forecasting ignores domain knowledge. The Markov layer blends three signals: empirical state transitions, cyber kill-chain escalation priors (Normal → Recon → DoS → DDoS), and real-time contextual signals.

**Why MC Dropout?**  
Uncertainty quantification matters in security. A high-confidence wrong prediction is more dangerous than an uncertain correct one. 30 stochastic forward passes give a calibrated uncertainty estimate alongside every forecast.

---

## 📸 Visualisations

<table>
  <tr>
    <td><img src="viz_09_xgb_confusion.png" width="300"/><br/><sub>XGBoost Confusion Matrix</sub></td>
    <td><img src="viz_12_lstm_training.png" width="300"/><br/><sub>LSTM Training Curves</sub></td>
    <td><img src="viz_14_threat_forecast.png" width="300"/><br/><sub>Threat Forecast Output</sub></td>
  </tr>
  <tr>
    <td><img src="dashboard_A_AllNormal.png" width="300"/><br/><sub>Dashboard — Normal Traffic</sub></td>
    <td><img src="dashboard_C_SuddenDDoS.png" width="300"/><br/><sub>Dashboard — DDoS Attack</sub></td>
    <td><img src="dashboard_E_APTSimulation.png" width="300"/><br/><sub>Dashboard — APT Simulation</sub></td>
  </tr>
</table>

---

## 📄 Requirements

```
tensorflow>=2.12
xgboost>=1.7
scikit-learn>=1.2
imbalanced-learn>=0.10
numpy>=1.23
pandas>=1.5
matplotlib>=3.6
seaborn>=0.12
joblib>=1.2
```

---

## 👤 Author

**[Your Name]**  
Final Year Project — B.Tech / BSc [Your Degree]  
[Your University], [Year]

[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/YOUR_USERNAME)

---

## ⭐ If this helped you

Give it a star — it helps others in cybersecurity & ML find this work!
