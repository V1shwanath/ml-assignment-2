# M.Tech (AIML) — Machine Learning Assignment 2
## End-to-End Breast Cancer Diagnostic Classification & Streamlit Deployment

---

### a. Problem Statement
Early and accurate differentiation between malignant and benign breast masses is a critical clinical classification problem in oncology. Given quantitative cytological measurements extracted from digitized images of fine needle aspirate (FNA) biopsies of breast masses, the objective of this project is to design, train, evaluate, and deploy five distinct supervised machine learning classification models (**Logistic Regression**, **Decision Tree Classifier**, **K-Nearest Neighbor Classifier**, **Gaussian Naive Bayes Classifier**, and **Support Vector Machine**) to predict whether a breast tumor sample is **Malignant (`0`)** or **Benign (`1`)**. Furthermore, an interactive **Streamlit** web application (`app.py`) is built and deployed on Streamlit Community Cloud to allow real-time evaluation of saved models on uploaded test datasets (`test_data.csv`) without retraining.

---

### b. Dataset Description
- **Dataset Name**: Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source Repository**: UCI Machine Learning Repository / `sklearn.datasets.load_breast_cancer`
- **Problem Type**: Binary Classification (`0` = Malignant, `1` = Benign)
- **Total Instances**: **569** samples (exceeds minimum requirement of 500 instances)
  - **Training Set (75%)**: 426 instances (stratified split, `random_state=2026`)
  - **Test Set (25% — `test_data.csv`)**: 143 instances (53 Malignant, 90 Benign)
- **Total Features**: **30** continuous numerical input features (exceeds minimum requirement of 12 features)
- **Feature Categories**:
  Ten real-valued cell nucleus properties are computed for each biopsy sample:
  1. `radius` (mean of distances from center to points on the perimeter)
  2. `texture` (standard deviation of gray-scale values)
  3. `perimeter`
  4. `area`
  5. `smoothness` (local variation in radius lengths)
  6. `compactness` ($\text{perimeter}^2 / \text{area} - 1.0$)
  7. `concavity` (severity of concave portions of the contour)
  8. `concave_points` (number of concave portions of the contour)
  9. `symmetry`
  10. `fractal_dimension` ("coastline approximation" $- 1$)
  
  For each of these 10 core properties, the **mean**, **standard error (`error`)**, and **worst (largest mean of top 3 values)** are recorded, yielding $10 \times 3 = 30$ features.
- **Preprocessing Applied**:
  Z-score feature standardization (`StandardScaler`) fitted strictly on the 426 training instances and persisted to `model/feature_scaler.joblib` to prevent data leakage during test evaluation.

---

### c. Links (GitHub Repository & Live Streamlit App)
- **GitHub Repository Link**: https://github.com/V1shwanath/ml-assignment-2
- **Live Streamlit App Link**: *(Deployed via Streamlit Community Cloud from `app.py`)*

#### Repository Structure
```text
project-folder/
│-- app.py                      # Interactive Streamlit web application
│-- requirements.txt            # Python package dependencies for cloud deployment
│-- README.md                   # Complete project documentation & evaluation tables
│-- test_data.csv               # Held-out test dataset (143 rows, 30 features + target)
│-- model/
    │-- train_models.py         # Complete model training & evaluation source script
    │-- model_training.ipynb    # Jupyter Notebook version of training & evaluation
    │-- feature_scaler.joblib   # Saved StandardScaler preprocessing artifact
    │-- logistic_regression.joblib
    │-- decision_tree.joblib
    │-- knn_classifier.joblib
    │-- naive_bayes.joblib
    │-- svm_classifier.joblib
    │-- metrics_comparison.csv  # Exported evaluation metrics table
    │-- model_metadata.json     # Feature names & diagnostic metadata
```

---

### d. Models Used & Evaluation Metrics Comparison Table

All five classification models were trained on the same standardized training split ($N_{\text{train}} = 426$) and evaluated on the held-out test set (`test_data.csv`, $N_{\text{test}} = 143$):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9790 | 0.9958 | 0.9780 | 0.9889 | 0.9834 | 0.9550 |
| **Decision Tree** | 0.9091 | 0.8991 | 0.9326 | 0.9222 | 0.9274 | 0.8060 |
| **KNN** | 0.9650 | 0.9828 | 0.9570 | 0.9889 | 0.9727 | 0.9251 |
| **Naive Bayes** | 0.9021 | 0.9799 | 0.9419 | 0.9000 | 0.9205 | 0.7947 |
| **Support Vector Machine** | 0.9790 | 0.9962 | 0.9888 | 0.9778 | 0.9832 | 0.9553 |

---

### e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved tied-highest Accuracy (**0.9790**), top Recall (**0.9889**), and top F1 Score (**0.9834**) with an outstanding AUC of **0.9958** and MCC of **0.9550**. Because standardized cytological features exhibit strong linear separability in log-odds space and L2 regularization ($C=0.85$) mitigates multicollinearity among correlated radius/perimeter/area features, Logistic Regression generalizes exceptionally well. |
| **Decision Tree** | Recorded an Accuracy of **0.9091**, AUC of **0.8991**, F1 of **0.9274**, and MCC of **0.8060**. Axis-aligned orthogonal splits (`max_depth=5`) struggle to capture smooth diagonal decision boundaries across 30 correlated continuous features, leading to higher variance and lower ranking discrimination compared to margin-based classifiers. |
| **KNN** | Delivered strong performance with **0.9650** Accuracy, **0.9828** AUC, **0.9889** Recall, **0.9727** F1, and **0.9251** MCC using distance-weighted $k=7$ neighbors. Feature standardization enables Euclidean distance to group benign and malignant profiles accurately, though minor sensitivity to local noise in 30-dimensional space keeps its precision (**0.9570**) slightly below SVM and Logistic Regression. |
| **Naive Bayes** | Achieved **0.9021** Accuracy, **0.9205** F1, and **0.7947** MCC, despite a high ranking AUC of **0.9799**. Gaussian Naive Bayes assumes conditional independence across features; however, features like `mean_radius`, `mean_perimeter`, `mean_area`, and their `worst_*` counterparts are heavily correlated, causing overconfident probability estimates and a drop in thresholded accuracy and recall (**0.9000**). |
| **Support Vector Machine** | Achieved tied-highest Accuracy (**0.9790**), the highest **AUC (0.9962)**, highest **Precision (0.9888)**, and highest **MCC (0.9553)**. The RBF kernel ($C=1.2$) effectively maps the 30 standardized cytological attributes into a higher-dimensional space where maximum-margin separation handles subtle non-linear boundary cases between borderline malignant and benign tumors. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (SVM)** (with **Logistic Regression** as a virtual tie/close runner-up). SVM edges ahead with the highest **MCC (0.9553)**, highest **AUC (0.9962)**, and highest **Precision (0.9888)** while matching the top **Accuracy (0.9790)**, making it the most reliable overall classifier on the Wisconsin Breast Cancer Diagnostic dataset. |
