from src.preprocess import generate_fake_data
from src.train_xgb import train_xgb
from src.train_lstm import train_lstm
from src.evaluate import evaluate_xgb


def main():
    print("Generating fake data...")
    generate_fake_data()
    
    print("\nTraining baseline XGBoost model...")
    xgb_model = train_xgb()
    
    print("\nTraining LSTM sequence model...")
    lstm_model = train_lstm()
    
    print("\nEvaluating XGBoost with SHAP...")
    evaluate_xgb()
    
    print("\nPipeline complete!")

if __name__ == "__main__":
    main()
