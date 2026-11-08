"""
End-to-End Model Training & Evaluation Script
Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset (UCI / scikit-learn)
Features: 30 numerical cytological attributes | Instances: 569 | Classes: 2 (Malignant vs Benign)
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# Paths configuration
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 2026
TEST_RATIO = 0.25


def prepare_dataset():
    """Load the Wisconsin Breast Cancer Diagnostic dataset and split into train/test sets."""
    raw_bundle = load_breast_cancer()
    feature_names = [
        col.replace(" ", "_").replace("(", "").replace(")", "")
        for col in raw_bundle.feature_names
    ]
    
    features_df = pd.DataFrame(raw_bundle.data, columns=feature_names)
    # In sklearn breast_cancer: 0 = malignant, 1 = benign
    # We keep target column named 'diagnosis_target' (0: Malignant, 1: Benign)
    target_series = pd.Series(raw_bundle.target, name="diagnosis_target")
    
    full_dataset = pd.concat([features_df, target_series], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        features_df,
        target_series,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=target_series,
    )
    
    # Save test_data.csv (unscaled raw features + target column) at project root
    test_export_df = pd.concat([X_test, y_test], axis=1)
    test_csv_path = BASE_DIR / "test_data.csv"
    test_export_df.to_csv(test_csv_path, index=False)
    
    # Also save full dataset for reference in model directory
    full_dataset.to_csv(MODEL_DIR / "full_breast_cancer_dataset.csv", index=False)
    
    return X_train, X_test, y_train, y_test, feature_names


def build_classifiers():
    """Initialize all 5 required classification models."""
    return {
        "Logistic Regression": {
            "filename": "logistic_regression.joblib",
            "estimator": LogisticRegression(
                C=0.85, max_iter=1000, solver="lbfgs", random_state=RANDOM_SEED
            ),
        },
        "Decision Tree": {
            "filename": "decision_tree.joblib",
            "estimator": DecisionTreeClassifier(
                criterion="gini", max_depth=5, min_samples_split=6, random_state=RANDOM_SEED
            ),
        },
        "KNN": {
            "filename": "knn_classifier.joblib",
            "estimator": KNeighborsClassifier(
                n_neighbors=7, weights="distance", metric="minkowski", p=2
            ),
        },
        "Naive Bayes": {
            "filename": "naive_bayes.joblib",
            "estimator": GaussianNB(var_smoothing=1e-8),
        },
        "Support Vector Machine": {
            "filename": "svm_classifier.joblib",
            "estimator": SVC(
                C=1.2, kernel="rbf", gamma="scale", probability=True, random_state=RANDOM_SEED
            ),
        },
    }


def compute_evaluation_metrics(y_true, y_pred, y_prob):
    """Calculate all 6 required evaluation metrics."""
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "AUC": round(float(roc_auc_score(y_true, y_prob)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, average="binary")), 4),
        "Recall": round(float(recall_score(y_true, y_pred, average="binary")), 4),
        "F1": round(float(f1_score(y_true, y_pred, average="binary")), 4),
        "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 4),
    }


def main():
    print("=" * 72)
    print("Breast Cancer Diagnostic Classification - Training & Evaluation Pipeline")
    print("=" * 72)

    X_train, X_test, y_train, y_test, feature_cols = prepare_dataset()
    print(f"Total Instances: {len(X_train) + len(X_test)} | Features: {len(feature_cols)}")
    print(f"Training Instances: {len(X_train)} | Test Instances: {len(X_test)}")

    # Fit StandardScaler on training features and save preprocessing artifact
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    scaler_path = MODEL_DIR / "feature_scaler.joblib"
    joblib.dump(feature_scaler, scaler_path)
    print(f"Saved preprocessing scaler to: {scaler_path}")

    classifiers_registry = build_classifiers()
    comparison_records = []
    detailed_diagnostics = {}

    for model_label, spec in classifiers_registry.items():
        clf = spec["estimator"]
        clf.fit(X_train_scaled, y_train)

        # Save trained model artifact
        model_save_path = MODEL_DIR / spec["filename"]
        joblib.dump(clf, model_save_path)

        # Predict on test set
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]

        metrics = compute_evaluation_metrics(y_test, y_pred, y_prob)
        row = {"ML Model Name": model_label, **metrics}
        comparison_records.append(row)

        detailed_diagnostics[model_label] = {
            "filename": spec["filename"],
            "metrics": metrics,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, target_names=["Malignant (0)", "Benign (1)"], output_dict=True
            ),
        }

    comparison_df = pd.DataFrame(comparison_records)
    comparison_df.to_csv(MODEL_DIR / "metrics_comparison.csv", index=False)

    metadata_payload = {
        "dataset_name": "Breast Cancer Wisconsin (Diagnostic) Dataset",
        "num_instances": int(len(X_train) + len(X_test)),
        "num_train": int(len(X_train)),
        "num_test": int(len(X_test)),
        "num_features": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "target_column": "diagnosis_target",
        "class_mapping": {"0": "Malignant", "1": "Benign"},
        "models": detailed_diagnostics,
    }

    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as fp:
        json.dump(metadata_payload, fp, indent=2)

    print("\nModel Evaluation Summary on Test Set (test_data.csv):")
    print("-" * 72)
    print(comparison_df.to_string(index=False))
    print("-" * 72)
    print("All models, preprocessing artifacts, and test_data.csv saved successfully!")


if __name__ == "__main__":
    main()
