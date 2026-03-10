import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "ml_dataset.csv"
MODEL_PATH = BASE_DIR / "data" / "rf_model.pkl"


def main():

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # Graph-aware features
    X = df[[
        "traffic",
        "length",
        "rain",
        "flood_risk",
        "degree",
        "nbr_risk",
        "nbr_traffic",
        "scale"
    ]]

    y = df["failed"]

    print("Train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)

    print("Evaluating...")
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    auc = roc_auc_score(y_test, probs)
    print("\nROC AUC:", auc)

    print("\nSaving model...")
    joblib.dump(model, MODEL_PATH)

    print("Saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()