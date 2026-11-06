import streamlit as st
import pandas as pd
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
    ConfusionMatrixDisplay,
)

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🩺",
    layout="wide",
)

st.title("Breast Cancer Classification")
st.write(
    "Compare five machine learning classifiers "
    "using the saved Wisconsin Breast Cancer models."
)

# --------------------------------------------------
# 2. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"
DEFAULT_TEST_FILE = BASE_DIR / "test_data.csv"

# --------------------------------------------------
# 3. Load saved models and preprocessing artifacts
# --------------------------------------------------

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
        ),
    }

    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")

    return models, scaler, feature_names


try:
    models, scaler, feature_names = load_artifacts()
    st.success("All five trained models loaded successfully.")
except Exception as error:
    st.error(f"Unable to load saved models: {error}")
    st.stop()

# --------------------------------------------------
# 4. Load default test data or accept an optional upload
# --------------------------------------------------

st.header("Test Dataset")

uploaded_file = st.file_uploader(
    "Optionally upload another test CSV file to replace the default dataset",
    type=["csv"],
    help=(
        "If no file is uploaded, the app automatically uses "
        "test_data.csv included in the GitHub repository."
    ),
)

try:
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.caption(f"Using uploaded dataset: {uploaded_file.name}")
    else:
        if not DEFAULT_TEST_FILE.exists():
            st.error(
                "The default test_data.csv file was not found. "
                "Make sure it is present in the repository root."
            )
            st.stop()

        data = pd.read_csv(DEFAULT_TEST_FILE)
        st.caption("Automatically loaded the default test_data.csv from the repository.")

    st.subheader("Dataset Preview")
    st.write(f"Dataset shape: {data.shape[0]} rows × {data.shape[1]} columns")
    st.dataframe(data.head(), use_container_width=True)

except Exception as error:
    st.error(f"Unable to read the test dataset: {error}")
    st.stop()

# --------------------------------------------------
# 5. Validate dataset
# --------------------------------------------------

if "diagnosis" not in data.columns:
    st.error("The test CSV must contain a 'diagnosis' column.")
    st.stop()

# Remove optional ID column
if "id" in data.columns:
    data = data.drop(columns=["id"])

# Separate actual target and input features
y_actual = data["diagnosis"].copy()
X_input = data.drop(columns=["diagnosis"])

# Validate feature columns
missing_features = [
    column for column in feature_names if column not in X_input.columns
]
extra_features = [
    column for column in X_input.columns if column not in feature_names
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
    "1": 1,
}
y_actual = y_actual.map(target_mapping)

if y_actual.isna().any():
    st.error("Invalid diagnosis values. Use B/M or 0/1.")
    st.stop()

if X_input.isna().any().any():
    st.error("The dataset contains missing feature values.")
    st.stop()

try:
    X_input = X_input.astype(float)
except (TypeError, ValueError):
    st.error("All input feature columns must contain numeric values.")
    st.stop()

y_actual = y_actual.astype(int)

if y_actual.nunique() < 2:
    st.error(
        "AUC cannot be calculated because the test data must contain "
        "both diagnosis classes."
    )
    st.stop()

# --------------------------------------------------
# 6. Select model
# --------------------------------------------------

st.header("Select Classification Model")

selected_model_name = st.selectbox(
    "Choose a classifier",
    list(models.keys()),
)

# --------------------------------------------------
# 7. Prepare input and make predictions
# --------------------------------------------------

scaled_models = [
    "Logistic Regression",
    "K-Nearest Neighbors",
    "Gaussian Naive Bayes",
    "Support Vector Machine",
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
# 8. Calculate and display evaluation metrics
# --------------------------------------------------

accuracy = accuracy_score(y_actual, y_pred)
auc = roc_auc_score(y_actual, y_score)
precision = precision_score(y_actual, y_pred, zero_division=0)
recall = recall_score(y_actual, y_pred, zero_division=0)
f1 = f1_score(y_actual, y_pred, zero_division=0)
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
# 9. Confusion Matrix
# --------------------------------------------------

st.header("Confusion Matrix")

cm = confusion_matrix(y_actual, y_pred, labels=[0, 1])

fig, ax = plt.subplots(figsize=(6, 4))
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign", "Malignant"],
)
display.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title(selected_model_name)
st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# 10. Prediction results and download
# --------------------------------------------------

st.header("Prediction Results")

prediction_results = X_input.copy()
prediction_results["Actual Diagnosis"] = y_actual.map(
    {0: "Benign", 1: "Malignant"}
)
prediction_results["Predicted Diagnosis"] = pd.Series(
    y_pred, index=X_input.index
).map({0: "Benign", 1: "Malignant"})

st.dataframe(prediction_results, use_container_width=True)

csv_output = prediction_results.to_csv(index=False)

st.download_button(
    label="Download Prediction Results",
    data=csv_output,
    file_name="prediction_results.csv",
    mime="text/csv",
)
