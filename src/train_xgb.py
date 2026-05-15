import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from preprocess import load_data

def train_xgb():
    df = load_data()
    X = df.drop("mood_score", axis=1)
    y = df["mood_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, preds)
    print(f"XGBoost ROC AUC: {auc:.3f}")
    return model
