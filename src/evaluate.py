import shap
import matplotlib.pyplot as plt
from train_xgb import train_xgb
from preprocess import load_data

def evaluate_xgb():
    model = train_xgb()
    df = load_data()
    X = df.drop("mood_score", axis=1)
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X)

if __name__ == "__main__":
    evaluate_xgb()
