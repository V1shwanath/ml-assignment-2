"""
Interactive Streamlit Web Application for Breast Cancer Diagnostic Classification
Demonstrates 5 pre-trained classification models evaluated on uploaded test CSV data.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve,
)

# Page configuration
st.set_page_config(
    page_title="OncoMetric AI | Breast Cancer Classification Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DEFAULT_TEST_CSV = BASE_DIR / "test_data.csv"

MODEL_FILE_MAP = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn_classifier.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm_classifier.joblib",
}


@st.cache_resource
def load_artifacts():
    """Load pre-trained models, feature scaler, and metadata from model/ directory."""
    scaler = joblib.load(MODEL_DIR / "feature_scaler.joblib")
    with open(MODEL_DIR / "model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    models = {}
    for name, fname in MODEL_FILE_MAP.items():
        models[name] = joblib.load(MODEL_DIR / fname)
    return scaler, metadata, models


def evaluate_model_on_data(clf, X_scaled, y_true):
    """Run inference using a saved model without retraining and compute all 6 metrics."""
    y_pred = clf.predict(X_scaled)
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_scaled)[:, 1]
    else:
        y_prob = clf.decision_function(X_scaled)

    metrics = {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "AUC": round(float(roc_auc_score(y_true, y_prob)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 4),
    }
    cm = confusion_matrix(y_true, y_pred)
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=["Malignant (0)", "Benign (1)"],
        output_dict=True,
        zero_division=0,
    )
    return metrics, cm, report_dict, y_pred, y_prob


def main():
    # Custom header styling
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #0f172a 0%, #1e3a8a 100%);
                    padding: 1.4rem 1.8rem; border-radius: 12px; color: white; margin-bottom: 1.2rem;">
            <h2 style="margin: 0; color: #f8fafc;">🔬 OncoMetric AI — Breast Cancer Diagnostic Evaluator</h2>
            <p style="margin: 0.4rem 0 0 0; color: #cbd5e1; font-size: 0.98rem;">
                End-to-End Multi-Model Classification Benchmark | M.Tech (AIML) Machine Learning Assignment 2
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        scaler, metadata, trained_models = load_artifacts()
    except Exception as exc:
        st.error(f"Error loading saved models from `model/`: {exc}")
        st.stop()

    expected_features = metadata["feature_columns"]
    target_col = metadata["target_column"]

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Evaluation Controls")

        st.subheader("1. Upload Test Dataset (CSV)")
        uploaded_file = st.file_uploader(
            "Upload test CSV containing 30 features + target column",
            type=["csv"],
            help=f"Must include the 30 cytology features and `{target_col}` column.",
        )

        # Provide sample test_data.csv download for easy grading
        if DEFAULT_TEST_CSV.exists():
            with open(DEFAULT_TEST_CSV, "rb") as fp:
                st.download_button(
                    label="📥 Download Sample test_data.csv",
                    data=fp,
                    file_name="test_data.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        use_bundled_test = st.checkbox(
            "Use bundled test_data.csv if no file is uploaded",
            value=True,
        )

        st.divider()
        st.subheader("2. Select Classification Model")
        selected_model_name = st.selectbox(
            "Choose a pre-trained classifier:",
            options=list(MODEL_FILE_MAP.keys()),
            index=4,  # Default to Support Vector Machine or user choice
        )

        st.divider()
        st.caption(
            f"**Dataset**: {metadata['dataset_name']}\n\n"
            f"**Total Instances**: {metadata['num_instances']} "
            f"(Train: {metadata['num_train']}, Test: {metadata['num_test']})\n\n"
            f"**Features**: {metadata['num_features']} numeric attributes\n\n"
            f"**Target**: `{target_col}` (`0` = Malignant, `1` = Benign)"
        )

    # Load test dataframe
    active_df = None
    data_source_label = ""
    if uploaded_file is not None:
        try:
            active_df = pd.read_csv(uploaded_file)
            data_source_label = f"Uploaded File (`{uploaded_file.name}`)"
        except Exception as err:
            st.error(f"Could not parse uploaded CSV file: {err}")
            st.stop()
    elif use_bundled_test and DEFAULT_TEST_CSV.exists():
        active_df = pd.read_csv(DEFAULT_TEST_CSV)
        data_source_label = "Bundled Test Set (`test_data.csv`)"
    else:
        st.info("👈 Please upload `test_data.csv` from the sidebar (or enable the bundled test dataset checkbox) to view model evaluations.")
        st.stop()

    # Validate columns
    missing_features = [c for c in expected_features if c not in active_df.columns]
    if missing_features:
        st.error(
            f"The uploaded CSV is missing {len(missing_features)} required feature column(s): "
            f"`{', '.join(missing_features[:5])}...`"
        )
        st.stop()

    # Detect target column (either 'diagnosis_target' or 'target')
    actual_target_col = None
    for candidate in [target_col, "target", "diagnosis", "label", "class"]:
        if candidate in active_df.columns:
            actual_target_col = candidate
            break

    if actual_target_col is None:
        st.error(
            f"Target column `{target_col}` not found in uploaded CSV. "
            "Please ensure the test CSV includes both input features and the ground-truth target column."
        )
        st.stop()

    X_raw = active_df[expected_features]
    y_true = active_df[actual_target_col].astype(int)

    # Apply saved StandardScaler (no retraining)
    X_scaled = scaler.transform(X_raw)

    # Dataset Overview Banner
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    col_info1.metric("Active Data Source", data_source_label.split("(")[0].strip())
    col_info2.metric("Test Instances Evaluated", f"{len(active_df)}")
    col_info3.metric("Input Features", f"{len(expected_features)}")
    malignant_cnt = int((y_true == 0).sum())
    benign_cnt = int((y_true == 1).sum())
    col_info4.metric("Class Balance (M / B)", f"{malignant_cnt} / {benign_cnt}")

    with st.expander("🔍 Preview Uploaded Test Data (First 10 Rows)", expanded=False):
        st.dataframe(active_df.head(10), use_container_width=True)

    st.divider()

    # Evaluate selected model
    selected_clf = trained_models[selected_model_name]
    sel_metrics, sel_cm, sel_report, sel_pred, sel_prob = evaluate_model_on_data(
        selected_clf, X_scaled, y_true
    )

    st.subheader(f"📊 Evaluation Metrics — {selected_model_name}")
    st.caption(
        f"Loaded saved artifact `model/{MODEL_FILE_MAP[selected_model_name]}` "
        f"and evaluated on {len(active_df)} test instances without retraining."
    )

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"{sel_metrics['Accuracy']:.4f}")
    m2.metric("AUC Score", f"{sel_metrics['AUC']:.4f}")
    m3.metric("Precision", f"{sel_metrics['Precision']:.4f}")
    m4.metric("Recall", f"{sel_metrics['Recall']:.4f}")
    m5.metric("F1 Score", f"{sel_metrics['F1']:.4f}")
    m6.metric("MCC Score", f"{sel_metrics['MCC']:.4f}")

    st.write("")

    # Confusion Matrix & Classification Report side-by-side
    left_col, right_col = st.columns([1, 1.15])

    with left_col:
        st.markdown(f"#### 🧩 Confusion Matrix ({selected_model_name})")
        fig_cm, ax_cm = plt.subplots(figsize=(5.2, 4.2))
        sns.heatmap(
            sel_cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Predicted Malignant (0)", "Predicted Benign (1)"],
            yticklabels=["Actual Malignant (0)", "Actual Benign (1)"],
            annot_kws={"size": 13, "weight": "bold"},
            ax=ax_cm,
        )
        ax_cm.set_xlabel("Predicted Diagnosis", fontsize=10, fontweight="bold")
        ax_cm.set_ylabel("Actual Ground Truth", fontsize=10, fontweight="bold")
        ax_cm.set_title(f"{selected_model_name} — Confusion Matrix", fontsize=11, pad=10)
        fig_cm.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with right_col:
        st.markdown(f"#### 📋 Classification Report ({selected_model_name})")
        report_df = pd.DataFrame(sel_report).transpose()
        # Keep key rows and format nicely
        display_rows = ["Malignant (0)", "Benign (1)", "macro avg", "weighted avg"]
        filtered_report = report_df.loc[[r for r in display_rows if r in report_df.index]].copy()
        filtered_report = filtered_report[["precision", "recall", "f1-score", "support"]]
        filtered_report.columns = ["Precision", "Recall", "F1-Score", "Support"]
        filtered_report["Support"] = filtered_report["Support"].astype(int)
        st.dataframe(
            filtered_report.style.format(
                {"Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}", "Support": "{:d}"}
            ),
            use_container_width=True,
        )

        # ROC Curve plot
        fpr, tpr, _ = roc_curve(y_true, sel_prob)
        fig_roc, ax_roc = plt.subplots(figsize=(5.2, 2.5))
        ax_roc.plot(
            fpr,
            tpr,
            color="#1d4ed8",
            lw=2.2,
            label=f"{selected_model_name} (AUC = {sel_metrics['AUC']:.4f})",
        )
        ax_roc.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", lw=1.2, label="Random Baseline")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=10)
        ax_roc.legend(loc="lower right", fontsize=8.5)
        ax_roc.grid(alpha=0.25)
        fig_roc.tight_layout()
        st.pyplot(fig_roc)
        plt.close(fig_roc)

    st.divider()

    # All 5 Models Benchmark Comparison Table on Uploaded Test Data
    st.subheader("🏆 All 5 Models Comparison on Test Data")
    st.caption(
        "Live side-by-side comparison across all five saved classifiers on the active test dataset."
    )

    all_rows = []
    for m_name, m_clf in trained_models.items():
        m_scores, _, _, _, _ = evaluate_model_on_data(m_clf, X_scaled, y_true)
        all_rows.append({"ML Model Name": m_name, **m_scores})

    all_comparison_df = pd.DataFrame(all_rows)

    st.dataframe(
        all_comparison_df.style.format(
            {
                "Accuracy": "{:.4f}",
                "AUC": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
                "MCC": "{:.4f}",
            }
        ).highlight_max(
            subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
            color="#dcfce7",
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Bar chart visual comparison
    fig_bar, ax_bar = plt.subplots(figsize=(9, 3.6))
    melted = all_comparison_df.melt(
        id_vars="ML Model Name",
        value_vars=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
        var_name="Metric",
        value_name="Score",
    )
    sns.barplot(
        data=melted,
        x="ML Model Name",
        y="Score",
        hue="Metric",
        palette="viridis",
        ax=ax_bar,
    )
    ax_bar.set_ylim(0.70, 1.03)
    ax_bar.set_ylabel("Metric Score")
    ax_bar.set_xlabel("")
    ax_bar.set_title("Multi-Metric Comparison Across All 5 Classifiers", fontsize=11, fontweight="bold")
    ax_bar.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
    ax_bar.grid(axis="y", alpha=0.3)
    fig_bar.tight_layout()
    st.pyplot(fig_bar)
    plt.close(fig_bar)


if __name__ == "__main__":
    main()
