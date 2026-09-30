# Air Quality Index Classification Using Machine Learning Algorithms

**Author:** Thota Vivek  
**Department of Computer Science and Engineering, IIIT Sricity**

---

## Abstract

This paper focuses on comparing the performance of selected classification algorithms for analyzing urban air quality. The primary goal is to explore how different machine learning approaches can be utilized to predict and group cities based on key air quality indicators such as PM2.5, PM10, CO, NO, SO, and O levels. Publicly available datasets from Kaggle are used to ensure the availability of diverse and reliable environmental data. The project implements four distinct algorithms — Logistic Regression, Support Vector Machine (SVM), Decision Tree, and Naïve Bayes — to evaluate their effectiveness in modeling air quality patterns. Their performance is evaluated using accuracy and macro/weighted F1 on a held-out evaluation set. Through this comparative approach, the study aims to determine the most suitable algorithm for understanding and predicting air quality trends across urban regions.

**Keywords:** Logistic Regression, Support Vector Machine, Decision Tree, Naive Bayes, Gradient Descent, Quadratic Programming, Air Quality Index, AQI Classification, From-Scratch Implementation, Class Imbalance, Feature Importance, SMOTE

---

## 1. Introduction

Air quality has emerged as one of the most pressing environmental challenges of the 21st century, with significant implications for public health, climate change, and sustainable urban development. The Air Quality Index (AQI) serves as a standardized metric to communicate air quality levels to the public, categorizing air quality into six distinct classes: Good, Satisfactory, Moderate, Poor, Very Poor, and Severe.

The accurate prediction and classification of AQI categories is essential for:

- Public health protection through early warning systems
- Environmental policy formulation and regulation
- Urban planning and infrastructure development
- Research on air pollution patterns and trends

Traditional statistical methods for air quality prediction often struggle with the non-linear relationships, high-dimensional feature spaces, and class imbalance inherent in environmental datasets. Machine learning approaches offer promising alternatives by automatically learning complex patterns from data without explicit programming of rules.

This study presents a comprehensive evaluation of four fundamental machine learning classifiers for AQI category prediction:

1. **Logistic Regression** — A linear classifier with probabilistic outputs
2. **Support Vector Machine (SVM)** — A maximum-margin classifier with kernel methods
3. **Decision Tree** — A non-parametric tree-based classifier
4. **Naive Bayes** — A probabilistic classifier based on Bayes' theorem

Each classifier was implemented from scratch with state-of-the-art enhancements including regularization, optimization techniques, and memory-efficient processing. The models were evaluated on two distinct datasets to assess generalizability and performance consistency.

The primary objectives of this research are:

- To implement and compare four machine learning classifiers for AQI classification
- To evaluate model performance using comprehensive metrics (accuracy, precision, recall, F1-score)
- To analyze feature importance and model interpretability
- To assess scalability and computational efficiency
- To identify the most suitable classifier for air quality prediction tasks

---

## 2. Datasets Description

This study leverages two complementary air quality datasets to evaluate the robustness of machine learning models across diverse AQI prediction scenarios: a multi-city dataset (`city_day_dataset.csv`) focused on various pollutants across India, and a city-specific dataset (`delhi_air_quality_dataset.csv`) centered on Delhi's air quality profiles.

### 2.1 city_day_dataset.csv: Multi-City Air Quality Dataset

The City Day Dataset is a comprehensive air quality dataset containing measurements from multiple cities over time. The dataset consists of 29,531 samples with 16 columns representing various air quality parameters and metadata.

**Features:**

| Feature | Description |
|---------|-------------|
| City | Categorical variable representing 26 different cities |
| Date | Temporal information in datetime format |
| PM2.5 | Particulate matter < 2.5 μm diameter (μg/m³) |
| PM10 | Particulate matter < 10 μm diameter (μg/m³) |
| NO | Nitrogen monoxide concentration (μg/m³) |
| NO2 | Nitrogen dioxide concentration (μg/m³) |
| NOx | Nitrogen oxides concentration (μg/m³) |
| NH3 | Ammonia concentration (μg/m³) |
| CO | Carbon monoxide concentration (μg/m³) |
| SO2 | Sulfur dioxide concentration (μg/m³) |
| O3 | Ozone concentration (μg/m³) |
| Benzene | Benzene concentration (μg/m³) |
| Toluene | Toluene concentration (μg/m³) |
| Xylene | Xylene concentration (μg/m³) |
| AQI | Numeric Air Quality Index value |
| AQI_Bucket | Categorical target variable (6 classes) |

**Dataset Characteristics:**

- Total samples: 29,531
- Number of cities: 26
- Number of features used: 13 (excluding AQI to prevent data leakage)
- Target classes: 6 AQI categories
- Class distribution (before SMOTE): Moderate (13,510), Poor (8,224), Satisfactory (2,781), Very Poor (2,337), Good (1,341), Severe (1,338)
- Missing values: 88,488 total (handled through median imputation)
- Train-test split: 70% training (20,671 samples), 30% testing (8,860 samples)
- After SMOTE: 57,030 training samples (balanced across all classes)

### 2.2 delhi_air_quality_dataset.csv: Delhi-Specific Air Quality Dataset

The Delhi Air Quality Dataset is a focused dataset containing air quality measurements specifically for Delhi, India. It includes 1,461 samples with 12 columns.

**Features:**

| Feature | Description |
|---------|-------------|
| Date | Temporal information |
| Month | Month of the year (1–12) |
| Year | Year of measurement |
| Holidays_Count | Number of holidays in the period |
| Days | Day count |
| PM2.5 | Particulate matter 2.5 (μg/m³) |
| PM10 | Particulate matter 10 (μg/m³) |
| NO2 | Nitrogen dioxide concentration (μg/m³) |
| SO2 | Sulfur dioxide concentration (μg/m³) |
| CO | Carbon monoxide concentration (μg/m³) |
| Ozone | Ozone concentration (μg/m³) |
| AQI | Numeric Air Quality Index value |

**Dataset Characteristics:**

- Total samples: 1,461
- Number of features used: 8 (PM2.5, PM10, NO2, SO2, CO, Ozone, Month, Holidays_Count)
- Target variable: AQI_Bucket (derived from AQI values using standard thresholds)
- Target classes: 6 AQI categories
- Class distribution (before SMOTE): Moderate (463), Poor (384), Satisfactory (267), Very Poor (231), Severe (65), Good (51)
- Train-test split: 70% training (1,022 samples), 30% testing (439 samples)
- After SMOTE: ~1,800–2,200 training samples

### 2.3 Dataset Summary

| Characteristic | city_day.csv | delhi_air_quality.csv |
|---------------|--------------|----------------------|
| Dataset Type | Multi-City | City-Specific |
| Number of Records | 29,531 | 1,461 |
| Original Features | 16 | 12 |
| Final Features | 13 | 8 |
| Numerical Features | 13 | 9 |
| Categorical Features | 3 | 3 |
| Missing Values | Yes (88,488) | Yes |
| Class Distribution | Imbalanced | Highly Imbalanced |
| Feature Scaling | Standardization | Standardization |
| Train/Test Split | 70/30 | 70/30 |
| SMOTE Applied | Yes | Yes |

Both datasets share common air quality parameters including PM2.5, PM10, NO2, SO2, CO, and Ozone. The City Day Dataset provides a broader geographical scope, while the Delhi dataset offers focused temporal analysis for a single location.

---

## 3. Methodology

This section details the comprehensive preprocessing pipeline and the from-scratch implementation of four machine learning classifiers. All models were optimized for memory efficiency and performance on air quality datasets.

### 3.1 Data Preprocessing

The preprocessing pipeline was designed to handle the unique characteristics of air quality datasets while maintaining data integrity and maximizing model performance.

**Missing Value Handling:**
- Numeric columns: Missing values filled with the median of respective columns
- Categorical columns: `AQI_Bucket` missing values filled with 'Moderate' (most common class)
- Rationale: Median imputation preserves distribution better than mean for skewed data and avoids information loss from row deletion

**Feature Engineering:**
- Temporal features extracted from Date column: Year, Month, Day, DayOfWeek, DayOfYear, Quarter
- These features capture seasonal patterns and temporal trends in air quality

**Label Encoding:**
- City column encoded to numeric values (26 unique cities)
- AQI_Bucket encoded to numeric labels (0–5 for 6 classes)

**Feature Selection:**
- AQI column explicitly excluded to prevent data leakage
- Feature columns (City Day): City, PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene
- Feature columns (Delhi): PM2.5, PM10, NO2, SO2, CO, Ozone, Month, Holidays_Count

**Feature Scaling:**
- StandardScaler applied: `(X − μ) / σ`
- Essential for distance-based algorithms (SVM, Logistic Regression)
- Fit on training data only; applied to both training and test sets

**Class Imbalance Handling:**
- SMOTE (Synthetic Minority Oversampling Technique) applied to training data
- Generates synthetic samples for minority classes using k-nearest neighbors (k = 5)
- Applied only to training data to prevent data leakage

**Memory Optimizations:**
- `float32` data types used instead of `float64` (50% memory reduction)
- Sample limits: `MAX_TRAIN_SAMPLES` = 10,000–15,000 depending on model

---

### 3.2 Classifier 1: Logistic Regression

The Logistic Regression model uses Cross-Entropy Loss with L2 regularization, optimized using Mini-Batch Gradient Descent with Momentum and Adaptive Learning Rate.

**Cost Function:**

$$L = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] + \frac{\lambda}{2n} \|W\|^2$$

For multiclass classification, the softmax function is used:

$$\sigma(z)_j = \frac{\exp(z_j)}{\sum_k \exp(z_k)}$$

**Optimization Details:**

- Mini-batch size: 128 samples per batch
- Momentum coefficient: 0.9 — velocity update: `v_t = β·v_{t-1} − α·∇L`
- Adaptive learning rate schedule: starts at 0.01, decayed in stages down to 0.0005
- Early stopping with patience of 100 epochs
- Xavier/Glorot weight initialization

**Results:**

| Dataset | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| City Day (Train) | 0.7548 | 0.7554 | 0.7548 | 0.7536 |
| City Day (Test) | 0.7116 | 0.7533 | 0.7116 | 0.7217 |
| Delhi (Train) | 0.7796 | 0.7841 | 0.7796 | 0.7763 |
| Delhi (Test) | 0.6469 | 0.6800 | 0.6469 | 0.6551 |

**Per-Class Performance — City Day Dataset (Test):**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Good | 0.3559 | 0.8677 | 0.5047 | 431 |
| Moderate | 0.8499 | 0.7506 | 0.7971 | 4,005 |
| Poor | 0.5154 | 0.7048 | 0.5954 | 830 |
| Satisfactory | 0.7604 | 0.5993 | 0.6703 | 2,463 |
| Severe | 0.7430 | 0.8171 | 0.7783 | 421 |
| Very Poor | 0.7094 | 0.7324 | 0.7207 | 710 |

**Per-Class Performance — Delhi Dataset (Test):**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Good | 0.3250 | 0.8667 | 0.4727 | 15 |
| Moderate | 0.8346 | 0.6795 | 0.7491 | 156 |
| Poor | 0.6475 | 0.6752 | 0.6611 | 117 |
| Satisfactory | 0.6515 | 0.5972 | 0.6232 | 72 |
| Severe | 0.4815 | 0.7222 | 0.5778 | 18 |
| Very Poor | 0.5263 | 0.4918 | 0.5085 | 61 |

---

### 3.3 Classifier 2: Support Vector Machine (SVM)

The SVM implementation uses a Soft-Margin formulation with the CVXOPT quadratic programming solver and a One-vs-One multiclass strategy.

**Mathematical Formulation (Primal):**

$$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^{m} \xi_i \quad \text{s.t.} \quad y_i(w^T x_i + b) \geq 1 - \xi_i, \; \xi_i \geq 0$$

**Dual Formulation:**

$$\max_{\alpha} \sum \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j) \quad \text{s.t.} \quad 0 \leq \alpha_i \leq C, \; \sum \alpha_i y_i = 0$$

**Implementation Details:**
- Regularization parameter: C = 0.1
- Kernel: Linear (for large datasets), RBF (γ = 1/n_features), Polynomial
- One-vs-One multiclass: 15 binary classifiers for 6 classes
- QP solver tolerances: `feastol = abstol = reltol = 1e-6`

**Results:**

| Dataset | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| City Day | 0.7077 | 0.7394 | 0.7077 | 0.7063 |
| Delhi | 0.5626 | 0.6223 | 0.5626 | 0.5757 |

---

### 3.4 Classifier 3: Decision Tree

The Decision Tree classifier builds a tree-like model of decisions using Gini impurity as the splitting criterion.

**Gini Impurity:**

$$\text{Gini}(D) = 1 - \sum_{i=1}^{C} p_i^2$$

**Information Gain:**

$$\text{Gain}(D, A) = \text{Gini}(D) - \sum_v \frac{|D_v|}{|D|} \text{Gini}(D_v)$$

**Hyperparameters:**
- `max_depth`: 10
- `min_samples_split`: 5
- `min_gain`: 1e-7

**Results:**

| Dataset | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| City Day | 0.7764 | 0.7821 | 0.7764 | 0.7784 |
| Delhi | 0.7016 | 0.7178 | 0.7016 | 0.7039 |

**Per-Class Performance — City Day Dataset:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Good | 0.5680 | 0.7328 | 0.6400 |
| Moderate | 0.8550 | 0.8176 | 0.8359 |
| Poor | 0.6050 | 0.6576 | 0.6302 |
| Satisfactory | 0.7761 | 0.7761 | 0.7761 |
| Severe | 0.7232 | 0.7953 | 0.7575 |
| Very Poor | 0.7401 | 0.6927 | 0.7157 |

**Feature Importance (City Day Dataset):**

| Feature | Importance |
|---------|-----------|
| PM2.5 | 0.5053 (50.53%) |
| CO | 0.2018 (20.18%) |
| PM10 | 0.1505 (15.05%) |
| O3, NO2, etc. | Remaining |

---

### 3.5 Classifier 4: Naive Bayes

The Naive Bayes implementation uses Gaussian Kernel Density Estimation (KDE) with bandwidth h = 0.1.

**Probability Density Estimation:**

$$P(x_f \mid \text{class} = c) = \frac{1}{n} \sum_{i=1}^{n} K_h(x_f - x_i)$$

where the Gaussian kernel is:

$$K(u) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{u^2}{2}\right)$$

**Naive Independence Assumption:**

$$P(x \mid \text{class} = c) = \prod_f P(x_f \mid \text{class} = c)$$

**Posterior Probability:**

$$P(\text{class} = c \mid x) = \frac{P(x \mid \text{class} = c) \cdot P(\text{class} = c)}{P(x)}$$

**Results:**

| Dataset | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| City Day | 0.6800 | ~0.68 | ~0.68 | ~0.68 |
| Delhi | 0.6894 | 0.7013 | 0.6894 | 0.6927 |

---

### 3.6 Evaluation Metrics

The following metrics were used to evaluate all classifiers:

1. **Accuracy:** (TP + TN) / (TP + TN + FP + FN)
2. **Precision (Weighted):** TP / (TP + FP)
3. **Recall (Weighted):** TP / (TP + FN)
4. **F1-Score (Weighted):** 2 × (P × R) / (P + R)
5. **Per-Class Metrics:** Precision, Recall, F1-Score for each AQI category
6. **Confusion Matrix:** Actual vs. predicted class distribution
7. **Classification Report:** Comprehensive per-class metrics with support

---

## 4. Results and Analysis

### 4.1 Comparative Summary

| Model | Dataset | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| Logistic Regression | City Day | 0.7116 | 0.7533 | 0.7116 | 0.7217 |
| Logistic Regression | Delhi | 0.6469 | 0.6800 | 0.6469 | 0.6551 |
| SVM | City Day | 0.7077 | 0.7394 | 0.7077 | 0.7063 |
| SVM | Delhi | 0.5626 | 0.6223 | 0.5626 | 0.5757 |
| Decision Tree | City Day | **0.7764** | **0.7821** | **0.7764** | **0.7784** |
| Decision Tree | Delhi | **0.7016** | **0.7178** | **0.7016** | **0.7039** |
| Naive Bayes | City Day | 0.6800 | ~0.68 | ~0.68 | ~0.68 |
| Naive Bayes | Delhi | 0.6894 | 0.7013 | 0.6894 | 0.6927 |

### 4.2 Key Observations

- **Decision Tree is the top-performing model** on both datasets, achieving 77.64% accuracy on City Day and 70.16% on Delhi.
- **PM2.5 is the most predictive feature** across models, accounting for 50.53% of feature importance in the Decision Tree.
- **All models struggle with minority classes** (Good, Severe) despite SMOTE augmentation.
- **The Delhi dataset is more challenging** due to its smaller size (1,461 vs. 29,531 samples) and reduced feature set.
- **Logistic Regression and SVM** offer competitive performance to Decision Tree on the larger City Day dataset, but fall behind significantly on the smaller Delhi dataset.
- **Naive Bayes** shows consistent but lower performance, likely due to the violated feature independence assumption (pollutants such as PM2.5 and PM10 are strongly correlated).
- **From-scratch implementations** achieve performance comparable to standard libraries.

---

## 5. Visualization and Interpretability

All plots were generated using Matplotlib and Seaborn with IEEE-compliant styling (clear labels, grid lines, colorblind-safe palettes, and high-resolution output).

### 5.1 Logistic Regression

- **Confusion Matrices:** Separate heatmaps for train (Blues) and test (Reds) sets, revealing that the Good class is frequently misclassified.
- **Metrics Comparison Charts:** Bar charts comparing train vs. test accuracy, precision, recall, and F1-score side by side.

### 5.2 Decision Tree

- **Feature Importance Plot:** Bar chart showing PM2.5, CO, and PM10 as dominant predictors.
- **Per-class F1 charts** highlight performance variation across AQI categories.

### 5.3 General Observations

- The Moderate class consistently achieves the highest F1-score across all models due to its large representation in both datasets.
- Good and Severe classes consistently show low precision but relatively high recall, indicating a tendency toward over-prediction for these minority classes even after SMOTE.
- The train-test gap in Logistic Regression (~4.3% on City Day vs. ~13.3% on Delhi) suggests the smaller Delhi dataset introduces higher variance.

---

## 6. Conclusion

This study demonstrates that machine learning classifiers can effectively predict AQI categories from pollutant concentration data. The Decision Tree classifier emerged as the best-performing model across both datasets, benefiting from its ability to model non-linear feature interactions without requiring feature scaling. PM2.5 was identified as the single most important predictor of air quality, contributing over 50% of the Decision Tree's feature importance.

Key takeaways:

- Tree-based methods are well-suited for environmental data with complex, non-linear relationships.
- Class imbalance remains a challenge for all models, even with SMOTE; further work on cost-sensitive learning or ensemble methods could improve minority class performance.
- From-scratch implementations of all four classifiers achieved results competitive with library implementations, validating the correctness of the approaches.
- Future work could explore ensemble methods (Random Forest, Gradient Boosting), deep learning architectures, and richer temporal features to further improve AQI classification accuracy.