# Jet-Engine-Health-Monitoring-System-for-Predictive-Maintenance

An AI-driven predictive maintenance pipeline designed to predict the **Remaining Useful Life (RUL)** of turbofan jet engines. This project leverages deep temporal neural networks to analyze sequential sensor data from the benchmark **NASA C-MAPSS dataset**, enabling proactive maintenance scheduling before catastrophic component failure occurs.

---

## 📌 Project Overview
In safety-critical aviation environments, relying on fixed-interval maintenance schedules is inefficient and risky. This repository implements a data science workflow that transforms raw time-series sensor telemetry into actionable prognostic point-estimates and statistical safety windows.

### Key Milestones Completed:
* **Feature Engineering (Week 11):** Implemented rolling temporal window feature extraction across selected degradation sensors.
* **Model Optimization (Week 11):** Developed and tuned an optimal, sequence-length 50 LSTM network using a Piecewise Linear RUL target strategy (capped at 125 cycles).
* **Bias & Risk Audit (Week 13):** Stress-tested the system across all 4 NASA operational profiles, established prediction uncertainty boundaries ($\pm1.96\sigma$), and mapped critical failure modes.

---

## 📊 Dataset Structure
The system utilizes the **NASA C-MAPSS dataset** (nested inside `BD_Dataset.zip` under `/CMaps/`). The data consists of four distinct sub-datasets operating under varying environmental settings and fault modes:

| Sub-Dataset | Operating Conditions | Fault Modes |
| :--- | :--- | :--- |
| **FD001** | 1 (Sea Level) | 1 (HPC Degradation) |
| **FD002** | 6 (Varied Altitudes/Speeds) | 1 (HPC Degradation) |
| **FD003** | 1 (Sea Level) | 2 (HPC + Fan Degradation) |
| **FD004** | 6 (Varied Altitudes/Speeds) | 2 (HPC + Fan Degradation) |

---

## 📈 Evaluation & Key Findings

### 1. Cross-Dataset Bias Analysis (Week 13 Results)
The model was evaluated independently across all sub-datasets using a local MinMaxScaler profile to track structural domain generalization risks:

* **FD001 Evaluated:** RMSE = **41.1148** (93 Engines)
* **FD002 Evaluated:** RMSE = **52.4750** (235 Engines)
* **FD003 Evaluated:** RMSE = **40.5701** (97 Engines)
* **FD004 Evaluated:** RMSE = **52.3914** (228 Engines)

**Domain Bias Conclusion:** Performance is highly consistent on single-condition profiles (`FD001`, `FD003`) but degrades significantly when complex multi-condition operational settings (`FD002`, `FD004`) are introduced.

### 2. Quantification of Uncertainty
By evaluating the baseline testing distribution residuals, a **95% Confidence Interval** was established:
* **Residual Standard Deviation ($\sigma$):** 41.03 cycles
* **95% Safety Buffer Buffer Margin:** **$\pm80.42$ cycles**

*Operational Guideline:* Maintenance must be pulled according to the conservative lower bound ($RUL_{predicted} - 80.42$) to guarantee safety thresholds.

---

## 🗺️ Systemic Risk Profile
The implementation balances five deployment vulnerabilities mapped across a standard probability vs. severity framework:
1. **R1: Out-of-Distribution Data (Bias) [High Risk]:** Model performance drops in multi-condition operations.
2. **R2: Sensor Dropout Interruption [High Risk]:** LSTM requires a continuous, unbroken 50-cycle window; dropped telemetry frames crash inference.
3. **R3: Early-Life Point Overestimation [Medium Risk]:** Increased variance in the early healthy stages of the engine lifespan.
4. **R4: Automation Alert Fatigue [Medium Risk]:** Variance noise could trigger false alarms, leading to technician complacency.
5. **R5: Prognostic Explainability Deficit [Medium Risk]:** The LSTM acts as a "black box"—it predicts *when* an engine fails, but cannot isolate *which* internal component broke.

---

## 🛠️ Environment & Dependencies
To run this notebook, the following core Python libraries are required:
```python
pandas
numpy
scikit-learn
keras / tensorflow
matplotlib
