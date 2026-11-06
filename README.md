# BITS Pilani – M.Tech AI & ML
## Assignment 2: Breast Cancer Classification Using Machine Learning
### ID - 2026AK05267

## 1. Project Overview

This project implements and compares five supervised machine learning classification algorithms to predict whether a breast tumour is Benign (B) or Malignant (M) using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

The objective is to train, evaluate, and compare five classification models using multiple performance metrics and identify the best-performing classifier on the selected test dataset.

The five machine learning algorithms implemented are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

A Streamlit web application is also developed to allow users to upload test data, select a trained classifier, and view its evaluation metrics and confusion matrix.

## 2. Dataset Description

**Dataset Name:** Wisconsin Diagnostic Breast Cancer (WDBC)

**Dataset Source:** UCI Machine Learning Repository

**Dataset Link:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

Dataset characteristics:

- Total instances: 569
- Number of input features: 30
- Target variable: Diagnosis
- Classification type: Binary classification
- Missing values: None
- Duplicate records: None

The target variable contains two classes:

- B – Benign (encoded as 0)
- M – Malignant (encoded as 1)

The 30 numerical features describe characteristics of cell nuclei obtained from digitized images of fine needle aspirates of breast masses.

## 3. GitHub Repository

**GitHub Repository:**  
https://github.com/charanraj2026/Semester1_ML_Assignment2

The repository contains the source code, Jupyter Notebook, trained models, preprocessing artifacts, test dataset, and application dependencies.

## 4. Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the WDBC dataset.
2. Removed the ID column because it does not contribute to classification.
3. Checked the dataset for missing values and duplicate records.
4. Converted the diagnosis column into numerical labels:
   - Benign (B) = 0
   - Malignant (M) = 1
5. Separated the input features and target variable.
6. Split the dataset into training and testing sets using an 80:20 ratio.
7. Used stratified sampling to maintain the class distribution in both datasets.
8. Applied StandardScaler to the training features and used the fitted scaler to transform the test features.

**Train-Test Split:**

| Dataset | Number of Records |
|---|---:|
| Training dataset | 455 |
| Testing dataset | 114 |
| Total | 569 |

The random state was set to 42 to ensure reproducibility.

Feature scaling was applied to Logistic Regression, KNN, Gaussian Naive Bayes, and SVM. The Decision Tree model was trained using the original unscaled features.

## 5. Machine Learning Models

### 5.1 Logistic Regression

Logistic Regression is a supervised classification algorithm that estimates the probability of a sample belonging to a particular class using the logistic function.

Configuration:
- Maximum iterations: 1000
- Random state: 42
- Input: Standardized features

### 5.2 Decision Tree Classifier

Decision Tree is a supervised learning algorithm that makes predictions using a tree-like structure of decision rules based on feature values.

Configuration:
- Criterion: Gini
- Random state: 42
- Input: Original unscaled features

### 5.3 K-Nearest Neighbors (KNN)

KNN classifies a sample based on the majority class among its nearest neighbours. It uses distance measurements to identify similar data points.

Configuration:
- Number of neighbours (K): 5
- Distance metric: Minkowski
- Power parameter (p): 2 (Euclidean distance)
- Input: Standardized features

### 5.4 Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem. It assumes that the features follow a Gaussian distribution within each class and are conditionally independent given the class.

Configuration:
- Algorithm: Gaussian Naive Bayes
- Input: Standardized features

### 5.5 Support Vector Machine (SVM)

SVM is a supervised learning algorithm that identifies an optimal decision boundary to separate different classes. The RBF kernel is used to handle nonlinear relationships between features.

Configuration:
- Kernel: RBF
- C: 1
- Gamma: Scale
- Probability estimation: Disabled
- AUC calculation: Decision function
- Input: Standardized features

## 6. Evaluation Metrics

The five classifiers were evaluated using the following six performance metrics:

1. **Accuracy:** Measures the proportion of correctly classified instances out of the total instances.
2. **AUC Score:** Measures the model's ability to distinguish between the positive and negative classes across classification thresholds.
3. **Precision:** Measures the proportion of predicted positive cases that are actually positive.
4. **Recall:** Measures the proportion of actual positive cases correctly identified by the model.
5. **F1 Score:** Represents the harmonic mean of precision and recall.
6. **Matthews Correlation Coefficient (MCC):** Measures the quality of binary classifications by considering true positives, true negatives, false positives, and false negatives.

## 7. Model Performance Comparison

All five classifiers were evaluated using the same held-out test dataset containing 114 records.

| Model | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| SVM | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

## 8. Observations and Analysis

### Logistic Regression
- Achieved an accuracy of 96.49%.
- Obtained the highest AUC score of 0.9960 among the five models.
- Achieved a precision of 0.9750 and recall of 0.9286.
- Demonstrated strong classification performance and good class separation.

### Decision Tree
- Achieved an accuracy of 92.98%.
- Obtained an AUC score of 0.9246.
- Produced balanced precision and recall values of 0.9048.
- Its performance was lower than the other models on this test split.

### K-Nearest Neighbors
- Achieved an accuracy of 95.61%.
- Obtained an AUC score of 0.9823.
- Achieved a precision of 0.9744.
- Demonstrated strong performance, although its recall was slightly lower than Logistic Regression and SVM.

### Gaussian Naive Bayes
- Achieved an accuracy of 92.11%.
- Obtained an AUC score of 0.9891.
- Achieved a recall of 0.8571.
- Although it demonstrated good class discrimination based on AUC, its classification performance at the selected threshold was comparatively lower.

### Support Vector Machine
- Achieved the highest accuracy of 97.37%.
- Obtained a precision of 1.0000.
- Achieved an F1 score of 0.9630 and MCC of 0.9442.
- Demonstrated the strongest overall classification performance among the five models on this test split.

## 9. Overall Best Model

Based on the evaluation results, **Support Vector Machine (SVM)** is the best-performing classifier for this experiment.

Reasons:
- Highest accuracy: 97.37%.
- Highest precision: 1.0000.
- Highest F1 score: 0.9630.
- Highest MCC: 0.9442.
- Strong AUC score: 0.9947.

Logistic Regression achieved the highest AUC score of 0.9960, indicating excellent ability to distinguish between the two classes across different thresholds.

Therefore, SVM is selected as the overall winner based on its performance across the evaluated classification metrics, while Logistic Regression achieved the best AUC score.

These conclusions apply to the selected dataset split and model configurations. They do not establish that one algorithm will always outperform the others on different datasets.

## 10. Streamlit Web Application

**Live Application:**  
https://breast-cancer-ml-classifier-assignment2.streamlit.app/

The Streamlit application provides an interactive interface to evaluate the five trained classification models.

### Application Features

- Upload a test dataset in CSV format.
- Preview the uploaded dataset.
- Select one of the five trained classification models using a dropdown.
- Display Accuracy, AUC Score, Precision, Recall, F1 Score, and MCC.
- Display the confusion matrix.
- View classification results.
- Download prediction results as a CSV file.

### Application Workflow

1. The user uploads a test CSV file containing the 30 input features and the diagnosis column.
2. The application validates the uploaded dataset.
3. The saved model is loaded from the repository.
4. The saved StandardScaler is applied where required.
5. The selected classifier predicts the target labels.
6. The application calculates and displays the evaluation metrics and confusion matrix.

**Important:** The Streamlit application loads previously trained models. It does not retrain the models when a user uploads a test dataset.

## 11. Project Structure

```text
Semester1_ML_Assignment2/
│
├── Assignment_2_ML.ipynb
├── app.py
├── README.md
├── requirements.txt
├── wdbc.csv
├── test_data.csv
│
└── saved_models/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── svm.pkl
    ├── scaler.pkl
    └── feature_names.pkl
```

### File Description

| File | Description |
|---|---|
| Assignment_2_ML.ipynb | Data preprocessing, model training, evaluation and model saving |
| app.py | Streamlit application source code |
| README.md | Project documentation and evaluation results |
| requirements.txt | Python package dependencies |
| wdbc.csv | Original dataset used for model development |
| test_data.csv | Held-out test dataset used for evaluation |
| logistic_regression.pkl | Trained Logistic Regression model |
| decision_tree.pkl | Trained Decision Tree model |
| knn.pkl | Trained KNN model |
| naive_bayes.pkl | Trained Gaussian Naive Bayes model |
| svm.pkl | Trained SVM model |
| scaler.pkl | Fitted StandardScaler |
| feature_names.pkl | Expected input feature names and order |

## 12. Installation and Execution

### Prerequisites

- Python
- pip
- Git

### Step 1: Clone the repository

```bash
git clone https://github.com/charanraj2026/Semester1_ML_Assignment2.git
```

### Step 2: Navigate to the project directory

```bash
cd Semester1_ML_Assignment2
```

### Step 3: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## 13. Conclusion

This project demonstrates the implementation and comparison of five supervised machine learning classification algorithms using the Wisconsin Diagnostic Breast Cancer dataset.

All five models were trained and evaluated using a consistent training and testing methodology. Their performance was compared using six evaluation metrics.

Among the evaluated models, SVM achieved the highest accuracy, precision, F1 score, and MCC, making it the strongest overall model for this experiment. Logistic Regression achieved the highest AUC score.

The deployed Streamlit application provides an interactive way to evaluate the saved models using uploaded test datasets without requiring model retraining.

---

**Project:** ML Assignment 2  
**Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)  
**Classification:** Benign vs Malignant  
**GitHub:** https://github.com/charanraj2026/Semester1_ML_Assignment2  
**Live Application:** https://breast-cancer-ml-classifier-assignment2.streamlit.app/