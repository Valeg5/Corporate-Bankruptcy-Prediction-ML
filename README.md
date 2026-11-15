# Corporate Bankruptcy Prediction: Comparative ML Classification Benchmark

## Authors
* **Valentinos Goudas**
* **Olga Saridou**
* **Eleni Georgoudi**
* **Thrasyvoulos Vasdavanos**

Department of Applied Informatics, University of Macedonia (Academic Year 2025–2026)

---

A comparative machine learning research project evaluating eight classification algorithms for predicting corporate financial distress and bankruptcy, developed as part of the *Machine Learning* curriculum at the **University of Macedonia**.

The study addresses the critical challenge of **severe class imbalance** in historical corporate financial ratios, benchmarking models across predictive precision, recall, F1-score, ROC-AUC, and computational training latency.

---

## Dataset & Preprocessing Pipeline

* **Data Characteristics:** Multi-year financial performance indicators and activity ratios comparing healthy vs. bankrupt enterprises.
* **Class Imbalance:** Significant majority representation of solvent firms over bankrupt entities.
* **Feature Scaling:** Applied `MinMaxScaler` normalization $[0, 1]$ across continuous financial ratios to prevent scale-dominant bias.
* **Validation Strategy:** Evaluated using **Stratified K-Fold Cross-Validation** to preserve native class ratios across training and evaluation splits, paired with undersampling strategies.

---

## Evaluated Machine Learning Models

The study conducted an exhaustive benchmark across eight distinct classification paradigms:

1. **Logistic Regression (LR):** Parametric baseline evaluating linear probability boundaries.
2. **Linear Discriminant Analysis (LDA):** Bayesian classifier assuming normal feature distributions and shared covariance.
3. **Gaussian Naive Bayes (NB):** Fast probabilistic classifier based on feature conditional independence.
4. **k-Nearest Neighbors (k-NN):** Non-parametric instance-based classification.
5. **Decision Trees (CART):** Interpretable recursive partitioning tree baseline.
6. **Random Forest (RF):** Bagging ensemble of de-correlated decision trees mitigating variance.
7. **Support Vector Machines (SVM):** Maximum-margin hyperplane classification optimized for high-dimensional spaces.
8. **XGBoost (Extreme Gradient Boosting):** Scalable sequential boosting ensemble.

---

## Benchmark Results & Comparative Analysis

Models were evaluated across all test folds with particular emphasis on **Recall** (minimizing False Negatives / missed bankruptcies) and **F1-Score**:

| Classifier | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | Avg Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM (Selected Best)** | **0.927** | **0.985** | **0.940** | **0.962** | **0.836** | 0.156s |
| **Logistic Regression** | 0.923 | 0.984 | 0.936 | 0.960 | 0.834 | 0.020s |
| **Random Forest** | 0.915 | 0.987 | 0.925 | 0.955 | **0.855** | 0.415s |
| **LDA** | 0.914 | 0.985 | 0.926 | 0.955 | 0.836 | **0.006s** |
| **k-NN** | 0.885 | 0.986 | 0.895 | 0.938 | 0.800 | 0.066s |
| **Naive Bayes** | 0.868 | 0.987 | 0.876 | 0.928 | 0.819 | **0.005s** |
| **Decision Tree** | 0.824 | 0.985 | 0.833 | 0.902 | 0.650 | 0.016s |
| **XGBoost** | 0.891 | 0.120 | 0.585 | 0.199 | 0.834 | 0.124s |

---

## Key Findings & Critical Insights

* **Top Performing Model (SVM):** Support Vector Machines emerged as the optimal architecture, achieving the best balance between risk detection (**94.0% Recall**) and accuracy (**F1: 0.962**), recording the fewest False Negatives (only 630 missed cases across test sets).
* **Ensemble Resilience (Random Forest):** Exhibited superior class discrimination (**0.855 ROC-AUC**), effectively cutting single-tree False Negatives by over 50%.
* **The Imbalance & Hyperparameter Challenge (XGBoost):** While achieving 0.834 ROC-AUC, default gradient boosting suffered from heavy False Positives (Precision 0.120), highlighting the sensitivity of boosting methods on uncalibrated, highly skewed datasets without extensive Bayesian hyperparameter tuning.
* **Efficiency vs. Performance:** Linear models (LDA, Logistic Regression) offered remarkable computational efficiency (~0.006s to 0.02s) with near-top classification performance, making them strong candidates for real-time risk screening.

---

## Tech Stack

* **Language:** Python 3.x
* **Core Libraries:** `scikit-learn`, `pandas`, `numpy`, `xgboost`, `matplotlib`, `seaborn`
