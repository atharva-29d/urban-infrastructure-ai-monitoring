import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "ml_dataset.csv"
MODEL_PATH = BASE_DIR / "data" / "best_failure_model.pkl"
METRICS_PATH = BASE_DIR / "data" / "best_failure_model_metrics.json"
FEATURE_IMPORTANCE_PATH = BASE_DIR / "data" / "best_failure_model_feature_importance.csv"
COMPARISON_PATH = BASE_DIR / "data" / "model_comparison.json"
N_JOBS = 1


def build_feature_matrix(df):
    exclude = {"failed", "simulation_id", "run_index"}
    feature_columns = [column for column in df.columns if column not in exclude]
    return df[feature_columns], feature_columns


def grouped_train_val_test_split(df):
    groups = df["simulation_id"] if "simulation_id" in df.columns else pd.Series(range(len(df)))

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_idx, test_idx = next(splitter.split(df, df["failed"], groups=groups))

    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_groups = train_val_df["simulation_id"] if "simulation_id" in train_val_df.columns else pd.Series(range(len(train_val_df)))
    val_splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, val_idx = next(val_splitter.split(train_val_df, train_val_df["failed"], groups=train_groups))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def best_threshold(y_true, probabilities):
    precisions, recalls, thresholds = precision_recall_curve(y_true, probabilities)
    best_score = -1.0
    best = 0.5

    for idx, threshold in enumerate(thresholds):
        precision = precisions[idx]
        recall = recalls[idx]
        score = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        if score > best_score:
            best_score = score
            best = float(threshold)

    return best


def evaluate(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "f1": float(f1_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "threshold": float(threshold),
        "classification_report": classification_report(y_true, predictions, output_dict=True),
    }


def candidate_models():
    return [
        (
            "Random Forest",
            [
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=N_JOBS,
                    random_state=42,
                ),
            ],
        ),
        (
            "Extra Trees",
            [
                ExtraTreesClassifier(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced",
                    n_jobs=N_JOBS,
                    random_state=42,
                ),
            ],
        ),
        (
            "Logistic Regression",
            [
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
                    ]
                ),
            ],
        ),
    ]


def feature_importance_frame(model_name, model, feature_columns):
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame(
            {"feature": feature_columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

    if hasattr(model, "named_steps") and "model" in model.named_steps:
        inner = model.named_steps["model"]
        if hasattr(inner, "coef_"):
            return pd.DataFrame(
                {"feature": feature_columns, "importance": abs(inner.coef_[0])}
            ).sort_values("importance", ascending=False)

    return pd.DataFrame({"feature": feature_columns, "importance": [0.0] * len(feature_columns)})


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df.replace([float("inf"), -float("inf")], 0).fillna(0)

    if "initial_failed" in df.columns:
        df = df[df["initial_failed"] == 0].reset_index(drop=True)

    train_df, val_df, test_df = grouped_train_val_test_split(df)

    X_train, feature_columns = build_feature_matrix(train_df)
    X_val, _ = build_feature_matrix(val_df)
    X_test, _ = build_feature_matrix(test_df)

    y_train = train_df["failed"]
    y_val = val_df["failed"]
    y_test = test_df["failed"]

    comparison_rows = []
    best_model = None
    best_name = None
    best_threshold_value = 0.5
    best_score = -1.0

    print("Evaluating model candidates...")
    for model_name, variants in candidate_models():
        for index, model in enumerate(variants, start=1):
            print(f"\nTraining {model_name} variant {index}...")
            model.fit(X_train, y_train)
            val_probabilities = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_val)

            if not hasattr(model, "predict_proba"):
                val_probabilities = 1 / (1 + pd.Series(-val_probabilities).apply(lambda x: pow(2.718281828, x))).to_numpy()

            threshold = best_threshold(y_val, val_probabilities)
            validation_metrics = evaluate(y_val, val_probabilities, threshold)

            comparison_rows.append(
                {
                    "model_name": model_name,
                    "variant": index,
                    "validation_roc_auc": validation_metrics["roc_auc"],
                    "validation_pr_auc": validation_metrics["pr_auc"],
                    "validation_f1": validation_metrics["f1"],
                    "threshold": threshold,
                }
            )

            print(
                f"{model_name} variant {index} -> "
                f"ROC-AUC {validation_metrics['roc_auc']:.4f}, "
                f"PR-AUC {validation_metrics['pr_auc']:.4f}, "
                f"F1 {validation_metrics['f1']:.4f}"
            )

            if validation_metrics["pr_auc"] > best_score:
                best_score = validation_metrics["pr_auc"]
                best_model = model
                best_name = model_name
                best_threshold_value = threshold

    print(f"\nBest validation model: {best_name} with PR-AUC {best_score:.4f}")

    combined_train_df = pd.concat([train_df, val_df], ignore_index=True)
    X_combined, _ = build_feature_matrix(combined_train_df)
    y_combined = combined_train_df["failed"]

    # Refit the best model class on train+val by cloning from its fitted params.
    if isinstance(best_model, Pipeline):
        final_model = Pipeline(best_model.steps)
    else:
        final_model = best_model.__class__(**best_model.get_params())

    final_model.fit(X_combined, y_combined)

    test_probabilities = final_model.predict_proba(X_test)[:, 1] if hasattr(final_model, "predict_proba") else final_model.decision_function(X_test)
    if not hasattr(final_model, "predict_proba"):
        test_probabilities = 1 / (1 + pd.Series(-test_probabilities).apply(lambda x: pow(2.718281828, x))).to_numpy()

    metrics = evaluate(y_test, test_probabilities, best_threshold_value)
    metrics.update(
        {
            "selected_model": best_name,
            "best_validation_pr_auc": float(best_score),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "feature_columns": feature_columns,
        }
    )

    print("\nFinal test metrics:")
    print("ROC-AUC:", round(metrics["roc_auc"], 4))
    print("PR-AUC:", round(metrics["pr_auc"], 4))
    print("F1:", round(metrics["f1"], 4))
    print("Balanced Accuracy:", round(metrics["balanced_accuracy"], 4))

    joblib.dump(final_model, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison_rows, f, indent=2)

    importance_df = feature_importance_frame(best_name, final_model, feature_columns)
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    print("\nSaved model to:", MODEL_PATH)
    print("Saved metrics to:", METRICS_PATH)
    print("Saved comparison to:", COMPARISON_PATH)
    print("Saved feature importances to:", FEATURE_IMPORTANCE_PATH)


if __name__ == "__main__":
    main()
