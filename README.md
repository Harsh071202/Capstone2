# 🔍 Insurance Fraud Detection

This project aims to detect fraudulent insurance claims using machine learning techniques. With a dataset of real-world insurance claims, the model identifies patterns and anomalies that help in flagging potentially fraudulent activities.

## 📌 Problem Statement

Insurance fraud leads to significant financial losses each year. Identifying fraud early can save costs and ensure better claim verification processes. This project builds a model to distinguish between genuine and fraudulent claims using classification algorithms.

## 📊 Dataset

The dataset used contains various features related to insurance claims including:

- Demographics (e.g., Age, Gender)
- Claim history (e.g., Number of past claims)
- Policy details (e.g., Policy type, Deductible)
- Incident details (e.g., Accident Area, Fault)
- Target: `FraudFound` (1: Fraudulent, 0: Not Fraudulent)

## 🛠️ Technologies Used

- **Python**
- **Pandas, NumPy** – Data manipulation
- **Seaborn, Matplotlib** – Data visualization
- **LabelEncoder** – Categorical encoding
- **XGBoost** – Feature importance and modeling
- **SMOTE** – Class imbalance handling
- **Scikit-learn** – Model training and evaluation

## 🧠 Workflow

1. **Data Cleaning**: Removed irrelevant columns and handled missing values.
2. **Exploratory Data Analysis (EDA)**: Visualized fraud trends across various categories.
3. **Encoding**: Converted categorical variables using Label Encoding.
4. **Feature Selection**: Used XGBoost to identify top features contributing to fraud detection.
5. **SMOTE**: Applied Synthetic Minority Oversampling Technique to balance classes.
6. **Model Training**: Trained an XGBoost Classifier on the balanced dataset.
7. **Evaluation**: Compared model performance before and after SMOTE using accuracy and class distribution.

## 📈 Results

- Successfully balanced the dataset using SMOTE.
- Top contributing features included: `Fault`, `PolicyType`, `AddressChange-Claim`, `AccidentArea`, etc.
- XGBoost model provided interpretability and effective feature ranking.

## ✅ Conclusion

This project shows how machine learning can be leveraged to flag potentially fraudulent claims and assist insurance companies in early detection, reducing financial losses.

## 📂 Folder Structure

insurance-fraud-detection/
│
├── insurance_fraud_detection.ipynb # Jupyter Notebook with full project
├── carclaims.csv # Dataset (if shared)
├── README.md # Project documentation

## 🙌 Acknowledgments

- Dataset: Kaggle
- Libraries: Scikit-learn, XGBoost, imbalanced-learn

---
