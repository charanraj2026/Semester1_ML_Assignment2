# Breast Cancer Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to develop and evaluate five supervised machine learning classification algorithms to predict whether a breast tumour is benign or malignant.

The project compares the performance of the following classifiers:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

All five classifiers are evaluated using the same training and testing split and six performance metrics.

## 2. Dataset Description

**Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)

**Source:** [UCI Machine Learning Repository – Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

Dataset characteristics:

- Total instances: 569
- Number of numerical input features: 30
- Target variable: Diagnosis
- Classification type: Binary classification
- Classes:
  - B – Benign
  - M – Malignant

The dataset contains measurements computed from digitized images of fine needle aspirates of breast masses.

The ID column was excluded from model training. The diagnosis column was encoded as:
- Benign = 0
- Malignant = 1

## 3. Data Preprocessing

The following preprocessing steps were performed:

1. Loaded and inspected the dataset.
2. Checked for missing values and duplicate records.
3. Removed the ID column because it is not a predictive feature.
4. Encoded the diagnosis target variable into binary numerical labels.
5. Split the dataset into training and testing sets using an 80:20 ratio.
6. Used stratified sampling to preserve the class distribution.
7. Applied StandardScaler to the models that require feature scaling.

The training set contains 455 instances, and the test set contains 114 instances.

The scaler was fitted only on the training data and then applied to the test data to prevent data leakage.

## 4. Machine Learning Models

### 4.1 Logistic Regression

Logistic Regression is a linear classification algorithm that estimates the probability of a sample belonging to a particular class.

### 4.2 Decision Tree Classifier

Decision Tree is a supervised learning algorithm that makes predictions using a sequence of feature-based decision rules.

### 4.3 K-Nearest Neighbors

KNN classifies a sample based on the majority class among its nearest neighbours. The implementation uses five neighbours.

### 4.4 Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem, assuming conditional independence among features and Gaussian feature distributions.

### 4.5 Support Vector Machine

SVM identifies a decision boundary that separates classes while maximizing the margin between them. The implementation uses an RBF kernel.

## 5. Model Evaluation Metrics

The five classifiers were evaluated using the following six metrics:

- **Accuracy:** Measures the proportion of correctly classified instances.
- **AUC Score:** Measures the model's ability to distinguish between the two classes across classification thresholds.
- **Precision:** Measures the proportion of predicted malignant cases that are actually malignant.
- **Recall:** Measures the proportion of actual malignant cases correctly identified.
- **F1 Score:** Harmonic mean of precision and recall.
- **Matthews Correlation Coefficient (MCC):** Measures the quality of binary classification using all four confusion matrix outcomes.

## 6. Model Performance Comparison

The following results were obtained on the held-out test dataset containing 114 instances.

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| K-Nearest Neighbors | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

## 7. Observations

### Logistic Regression
- Achieved 96.49% accuracy.
- Recorded the highest AUC score of 0.9960.
- Achieved strong precision and recall.
- Demonstrated excellent class discrimination on the test dataset.

### Decision Tree
- Achieved 92.98% accuracy.
- Produced lower AUC and MCC compared with the other models.
- Offers an interpretable decision-making structure.
- Its performance was lower than the other evaluated classifiers on this test split.

### K-Nearest Neighbors
- Achieved 95.61% accuracy.
- Recorded an AUC score of 0.9823.
- Achieved strong precision but comparatively lower recall.
- Feature scaling is important for distance-based classification.

### Gaussian Naive Bayes
- Achieved 92.11% accuracy.
- Recorded an AUC score of 0.9891.
- Its recall was 0.8571, the lowest among the five models.
- The results indicate that its probability-based ranking performance was strong, although its classification performance at the selected threshold was comparatively lower.

### Support Vector Machine
- Achieved the highest accuracy of 97.37%.
- Recorded the highest precision of 1.0000.
- Achieved the highest F1 score of 0.9630 and MCC of 0.9442.
- Demonstrated strong overall classification performance using the RBF kernel.

## 8. Overall Best-Performing Model

Based on the evaluation results, **Support Vector Machine (SVM)** is the overall best-performing classifier for this experiment.

It achieved:

- Accuracy: 97.37%
- Precision: 100%
- Recall: 92.86%
- F1 Score: 96.30%
- MCC: 0.9442
- AUC Score: 0.9947

Although Logistic Regression achieved a slightly higher AUC score of 0.9960, SVM achieved better accuracy, precision, F1 score and MCC.

Therefore, SVM demonstrated the strongest overall classification performance on the held-out test dataset.

These results are specific to the selected dataset split and model configurations.

## 9. Streamlit Application

A Streamlit application was developed to evaluate the five saved machine learning classifiers.

Application features:

- Upload a test dataset in CSV format.
- Select one of the five trained classifiers.
- Load the saved model without retraining.
- Apply the saved scaler where required.
- Calculate all six evaluation metrics.
- Display a confusion matrix.
- Display actual and predicted diagnosis values.
- Download prediction results as a CSV file.

### Run the application locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

Open the local application:

```text
http://localhost:8501
```

## 10. Project Structure

```text
Python/
│
├── saved_models/
│   ├── decision_tree.pkl
│   ├── feature_names.pkl
│   ├── knn.pkl
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── scaler.pkl
│   └── svm.pkl
│
├── app.py
├── Assignment_2_ML.ipynb
├── requirements.txt
├── test_data.csv
├── wdbc.csv
└── README.md
```

## 11. Repository and Application Links

**GitHub Repository:** To be added after publishing the project to GitHub.

**Streamlit Community Cloud Application:** To be added after deployment.

---

**Project:** Breast Cancer Classification Using Machine Learning  
**Dataset:** Wisconsin Diagnostic Breast Cancer  
**Classification:** Binary Classification  
**Algorithms:** Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes and SVM