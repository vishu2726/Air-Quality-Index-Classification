# Air Quality Index Classification using Machine Learning

## Overview

This project focuses on predicting and classifying Air Quality Index (AQI) categories using multiple machine learning algorithms on real-world environmental datasets. The project includes complete implementations, preprocessing pipelines, visualization, and comparative performance analysis of different classification models.

The objective is to evaluate how effectively machine learning techniques can classify AQI levels based on pollutant concentrations such as PM2.5, PM10, CO, NO2, SO2, and O3.

---

# Implemented Machine Learning Models

- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Naive Bayes

---

# Key Features

- From-scratch implementation of machine learning algorithms
- Data preprocessing and feature engineering
- Missing value handling and feature scaling
- SMOTE-based class imbalance handling
- Model evaluation using Accuracy, Precision, Recall, and F1-Score
- Confusion matrices and visualization plots
- Comparative analysis across multiple datasets

---

# Datasets Used

## 1. City Day AQI Dataset

- Multi-city air quality dataset
- 29,531 samples
- 16 original features
- 6 AQI classification categories

## 2. Delhi Air Quality Dataset

- Delhi-specific air quality dataset
- 1,461 samples
- Temporal and pollutant-based features
- 6 AQI classification categories

---

# Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Imbalanced-learn (SMOTE)
- CVXOPT

---

# Project Structure

```text
Air-Quality-Index-Classification/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── datasets/
│
├── notebooks/
│   ├── logistic_regression.ipynb
│   ├── svm.ipynb
│   ├── decision_tree.ipynb
│   └── naive_bayes.ipynb
│
├── images/
│
└── report/
    └── Air_Quality_Index_Classification_Report.pdf
```

---

# Model Performance Summary

| Model | Dataset | Accuracy | F1-Score |
|------|------|------|------|
| Logistic Regression | City Day | 71.16% | 72.17% |
| SVM | City Day | 70.77% | 70.63% |
| Decision Tree | City Day | **77.64%** | **77.84%** |
| Naive Bayes | City Day | 68.00% | 68.00% |
| Logistic Regression | Delhi AQI | 64.69% | 65.51% |
| SVM | Delhi AQI | 56.26% | 57.57% |
| Decision Tree | Delhi AQI | **70.16%** | **70.39%** |
| Naive Bayes | Delhi AQI | 68.94% | 69.27% |

---

# Key Findings

- Decision Tree achieved the best overall performance across both datasets.
- PM2.5 was identified as the most influential feature for AQI prediction.
- Larger datasets significantly improved model performance.
- SMOTE improved minority class prediction performance.
- Non-linear models performed better than linear models for AQI classification.

---

# Visualizations Included

- Confusion Matrices
- Metrics Comparison Charts
- Feature Importance Analysis
- Prediction Confidence Distribution
- SMOTE Class Distribution Comparison
- Performance Comparison Across Models

---

# Installation

Clone the repository:

```bash
git clone https://github.com/thota-vivek05/Air-Quality-Index-Classification.git
```

Move into the project directory:

```bash
cd Air-Quality-Index-Classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open any notebook from the `notebooks/` directory and run all cells.

---

# Future Improvements

- Random Forest and XGBoost implementation
- Hyperparameter optimization
- SHAP/LIME explainability
- Real-time AQI prediction system
- Deployment using Flask or Streamlit

---

# Results

The project demonstrates the effectiveness of machine learning techniques for environmental data analysis and AQI prediction. Among all models, the Decision Tree classifier produced the highest classification accuracy and overall performance.

---

# Author

## Thota Vivek

GitHub: https://github.com/thota-vivek05

---