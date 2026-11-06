
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🩺",
    layout="wide"
)

st.title("Breast Cancer Classification")
st.write(
    "Compare five machine learning classifiers "
    "using the saved Wisconsin Breast Cancer models."
)

# --------------------------------------------------
# 2. Load saved models and preprocessing artifacts
# --------------------------------------------------

MODEL_DIR = Path(__file__).parent / "saved_models"


@st.cache_resource
def load_artifacts():

    models = {
        "Logistic Regression": joblib.load(
            MODEL_DIR / "logistic_regression.pkl"
        ),
        "Decision Tree": joblib.load(
            MODEL_DIR / "decision_tree.pkl"
        ),
        "K-Nearest Neighbors": joblib.load(
            MODEL_DIR / "knn.pkl"
        ),
        "Gaussian Naive Bayes": joblib.load(
            MODEL_DIR / "naive_bayes.pkl"
        ),
        "Support Vector Machine": joblib.load(
            MODEL_DIR / "svm.pkl"
        )
    }

    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_names = joblib.load(
        MODEL_DIR / "feature_names.pkl"
    )

    return models, scaler, feature_names


try:
    models, scaler, feature_names = load_artifacts()
    st.success("All five trained models loaded successfully.")

except Exception as e:
    st.error(f"Unable to load saved models: {e}")
    st.stop()


# --------------------------------------------------
# 3. Upload test dataset
# --------------------------------------------------

st.header("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset")
        st.write("Dataset shape:", data.shape)
        st.dataframe(data.head())

        # --------------------------------------------------
        # 4. Validate dataset
        # --------------------------------------------------

        if "diagnosis" not in data.columns:
            st.error(
                "The uploaded CSV must contain a diagnosis column."
            )
            st.stop()

        # Remove optional ID column
        if "id" in data.columns:
            data = data.drop(columns=["id"])

        # Separate actual target and input features
        y_actual = data["diagnosis"].copy()
        X_input = data.drop(columns=["diagnosis"])

        # Validate feature columns
        missing_features = [
            col for col in feature_names
            if col not in X_input.columns
        ]

        extra_features = [
            col for col in X_input.columns
            if col not in feature_names
        ]

        if missing_features or extra_features:
            st.error(
                f"Missing features: {missing_features}\n\n"
                f"Unexpected features: {extra_features}"
            )
            st.stop()

        # Enforce the same feature order used during training
        X_input = X_input[feature_names]

        # Convert target labels to numeric values
        y_actual = y_actual.astype(str).str.strip().str.upper()

        target_mapping = {
            "B": 0,
            "M": 1,
            "0": 0,
            "1": 1
        }

        y_actual = y_actual.map(target_mapping)

        if y_actual.isna().any():
            st.error(
                "Invalid diagnosis values. Use B/M or 0/1."
            )
            st.stop()

        if X_input.isna().any().any():
            st.error("The dataset contains missing feature values.")
            st.stop()

        X_input = X_input.astype(float)
        y_actual = y_actual.astype(int)

        if y_actual.nunique() < 2:
            st.error(
                "AUC cannot be calculated because the test data "
                "must contain both diagnosis classes."
            )
            st.stop()

        # --------------------------------------------------
        # 5. Select model
        # --------------------------------------------------

        st.header("Select Classification Model")

        selected_model_name = st.selectbox(
            "Choose a classifier",
            list(models.keys())
        )

        # --------------------------------------------------
        # 6. Prepare input and make predictions
        # --------------------------------------------------

        scaled_models = [
            "Logistic Regression",
            "K-Nearest Neighbors",
            "Gaussian Naive Bayes",
            "Support Vector Machine"
        ]

        model = models[selected_model_name]

        if selected_model_name in scaled_models:
            X_processed = scaler.transform(X_input)
        else:
            X_processed = X_input

        y_pred = model.predict(X_processed)

        # Obtain scores for AUC calculation
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_processed)[:, 1]
        else:
            y_score = model.decision_function(X_processed)

        # --------------------------------------------------
        # 7. Calculate evaluation metrics
        # --------------------------------------------------

        accuracy = accuracy_score(y_actual, y_pred)
        auc = roc_auc_score(y_actual, y_score)
        precision = precision_score(
            y_actual, y_pred, zero_division=0
        )
        recall = recall_score(
            y_actual, y_pred, zero_division=0
        )
        f1 = f1_score(
            y_actual, y_pred, zero_division=0
        )
        mcc = matthews_corrcoef(y_actual, y_pred)

        st.header("Model Evaluation Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        # --------------------------------------------------
        # 8. Confusion Matrix
        # --------------------------------------------------

        st.header("Confusion Matrix")

        cm = confusion_matrix(
            y_actual,
            y_pred,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Benign", "Malignant"]
        )

        display.plot(
            ax=ax,
            cmap="Blues",
            values_format="d"
        )

        ax.set_title(selected_model_name)
        st.pyplot(fig)
        plt.close(fig)

        # --------------------------------------------------
        # 9. Prediction results
        # --------------------------------------------------

        st.header("Prediction Results")

        prediction_results = X_input.copy()

        prediction_results["Actual Diagnosis"] = (
            y_actual.map({0: "Benign", 1: "Malignant"})
        )

        prediction_results["Predicted Diagnosis"] = (
            pd.Series(y_pred, index=X_input.index)
            .map({0: "Benign", 1: "Malignant"})
        )

        st.dataframe(prediction_results)

        csv_output = prediction_results.to_csv(index=False)

        st.download_button(
            label="Download Prediction Results",
            data=csv_output,
            file_name="prediction_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing the uploaded dataset: {e}")

else:
    st.info("Please upload a test CSV file to begin.")
