# Symptom-Prediction: Advanced Clinical AI Diagnostics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ML Engine: XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Deep Learning: PyTorch](https://img.shields.io/badge/DL-PyTorch-red.svg)](https://pytorch.org/)

A comprehensive pipeline for predictive symptom analysis using hybrid machine learning architectures. This project integrates traditional gradient boosting (XGBoost) with deep sequential modeling (LSTM) to provide accurate diagnostic insights from clinical data.

## 🚀 Key Features

- **Hybrid Architecture**: Combines XGBoost for tabular feature extraction and LSTM for temporal sequence analysis.
- **Explainable AI (XAI)**: Integrated SHAP (SHLineary Additive Explanations) for model transparency and feature importance visualization.
- **Synthetic Data Generation**: Robust preprocessing pipeline with built-in fake data generation for testing and validation.
- **Clinical Interpretability**: Designed to handle multivariate symptom data with a focus on diagnostic accuracy.

## 🏗️ Project Structure

```text
symptom_prediction/
├── data/               # Clinical datasets and generated samples
├── models/             # Serialized model checkpoints (.json, .pth)
├── notebooks/          # Exploratory Data Analysis (EDA) and prototyping
├── src/                # Core engine modules
│   ├── preprocess.py   # Data cleaning and synthetic generation
│   ├── train_xgb.py    # XGBoost training logic
│   ├── train_lstm.py   # PyTorch LSTM implementation
│   └── evaluate.py     # SHAP and metric evaluation
├── main.py             # Orchestration script
└── requirements.txt    # Dependency manifest
```

## 🛠️ Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Ananthapadmanabhan333/Symptom-Prediction.git
   cd Symptom-Prediction
   ```

2. **Set up Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚦 Usage

Execute the entire pipeline (data generation → training → evaluation) with a single command:

```bash
python main.py
```

## 📊 Methodology

### XGBoost Baseline
The system utilizes XGBoost to capture non-linear relationships in symptom-disease mapping. This serves as a high-performance baseline for tabular data.

### LSTM Sequence Modeling
For symptoms that evolve over time, a PyTorch-based LSTM network is employed to capture temporal dependencies and sequential patterns in patient history.

### Interpretability with SHAP
We use SHAP kernels to explain why the model made a specific prediction, highlighting which symptoms were the strongest drivers for the diagnostic output.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Built with ❤️ for Clinical Informatics by [Ananthapadmanabhan](https://github.com/Ananthapadmanabhan333)*
