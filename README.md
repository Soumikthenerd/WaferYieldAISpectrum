# Wafer Yield AI Spectrum

> **3D Semiconductor Wafer Yield Prediction Engine**  
> Comparing a lightweight spatial ensemble (**Model A**) against deep sub-die signal telemetry (**Model B / C**).

---

## Executive Summary

**Wafer Yield AI Spectrum** is a two-tiered machine learning system designed to detect semiconductor die defects during wafer fabrication:

* **Model A (Tri-Model Ensemble):** Ingests spatial coordinates $(X, Y)$ and 503 engineered geometric/topological features. Blends LightGBM, XGBoost, and CatBoost using Out-of-Fold (OOF) Logistic Regression stacking.
* **Model B / C (PyTorch Hybrid 1D-CNN):** Processes 2,000-point sub-die laser sensor signal streams alongside tabular spatial features for deep anomaly detection.
* **Cost vs. Accuracy ROI:** Model A achieves **0.9557 AUC-PR** at 1% of the infrastructure cost. Model B adds a **+1.05% AUC-PR bump (0.9656)** for micro-defect verification, enabling a high-efficiency two-stage triage pipeline.

---

## Production Benchmark Comparison

Evaluated across **39,351 unseen test dies**:

| Metric | Model A (Spatial Ensemble) | Model B / C (1D-CNN Hybrid) | Production Impact |
| :--- | :---: | :---: | :--- |
| **AUC-PR Score** | **0.9557** | **0.9656** | High precision-recall trade-off |
| **Fail Precision** | **1.00** | **1.00** | Zero false scrap generated |
| **Pass Recall** | **1.00** | **1.00** | 100% nominal die retention |
| **Best F1-Score** | **0.9411** | **0.9421** | Robust defect classification |
| **Overall Accuracy** | **98.0%** | **98.0%** | Enterprise-grade reliability |

---

## Architecture Highlights

* **Leak-Free Validation:** Built with `StratifiedGroupKFold` grouped by physical wafer ID to ensure zero cross-die data leakage across training folds.
* **Feature Engineering:** Extracts radial distance, ring indices, local defect density, edge proximity, and coordinate symmetries (503 total spatial features).
* **Live API Backend:** FastAPI server deployed on Render (`https://waferyieldaispectrum.onrender.com`) serving sub-50ms inference.
* **3D Interactive UI:** Three.js WebGL viewport rendering SEMI-M1 300mm wafer topology with real-time laser sweep controls and 2,000-point sensor sparklines.

---

## Local Setup & Development

### 1. Backend (FastAPI Engine)

```bash
# Clone the repository
git clone [https://github.com/Soumikthenerd/WaferYieldAISpectrum.git](https://github.com/Soumikthenerd/WaferYieldAISpectrum.git)
cd WaferYieldAISpectrum

# Install python dependencies
pip install -r requirements.txt

# Start local server
uvicorn main:app --reload
