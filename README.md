# 👨‍💼 Employee Attrition Prediction using XGBoost (Regularized)

## 📌 Overview

This project builds a **machine learning model** to predict whether an employee is likely to:

* ❌ Leave the company (Attrition = 1)
* ✅ Stay with the company (Attrition = 0)

The solution uses **XGBoost with L1 and L2 regularization** to ensure strong performance and prevent overfitting.

---

## 🎯 Business Objective

Employee attrition is a critical challenge for organizations.
This project helps:

* Identify employees at **high risk of leaving**
* Enable **proactive retention strategies**
* Improve **workplace satisfaction and productivity**
* Reduce **hiring and training costs**

---

## 📂 Dataset Description

The dataset contains **900 records** and **15 features**, representing employee-related metrics.

### 🔢 Numerical Features

* JobSatisfaction
* PerformanceRating
* YearsAtCompany
* WorkLifeBalance
* DistanceFromHome
* MonthlyIncome
* EducationLevel
* Age
* NumCompaniesWorked
* AnnualBonus
* TrainingHours
* AnnualBonus_Squared
* AnnualBonus_TrainingHours_Interaction

### 🔤 Categorical Features

* EmployeeRole
* Department

### 🎯 Target Variable

* **Attrition**

  * `0` → Employee stays
  * `1` → Employee leaves

---

## ⚙️ Data Preprocessing

* Missing values handled:

  * Numeric → Median
  * Categorical → Mode
* Feature scaling using **StandardScaler**
* Categorical encoding using **OneHotEncoder**
* Pipeline used for reproducibility

---

## 🧠 Model Used

### 🚀 XGBoost Classifier

XGBoost is chosen because it:

* Handles **non-linear relationships**
* Performs well on **tabular data**
* Supports **built-in regularization**

---

## 🔒 Regularization Strategy

To prevent overfitting:

* **L1 Regularization (`reg_alpha`)**

  * Encourages sparsity (feature selection)

* **L2 Regularization (`reg_lambda`)**

  * Penalizes large weights

---

## 🏗️ Pipeline Architecture

```id="arch2"
Raw Data → Cleaning → Encoding → Scaling → XGBoost → Predictions
```

---

## 🚀 Installation

Clone the repository:

```bash id="cmd4"
git clone https://github.com/your-username/employee-attrition-xgboost.git
cd employee-attrition-xgboost
```

Install dependencies:

```bash id="cmd5"
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the training script:

```bash id="cmd6"
python main.py
```

---

## 📊 Evaluation Metrics

* **Accuracy**
* **F1 Score**
* **Confusion Matrix**
* **Classification Report**

---

## 📈 Sample Output

```id="out2"
===== XGBOOST MODEL PERFORMANCE =====
Accuracy: 0.89
F1 Score: 0.88
```

---

## 📁 Project Structure

```id="struct2"
employee-attrition-xgboost/
│
├── data/
│   └── modified_employee_turnover.csv
│
├── models/
│   └── employee_attrition_xgboost.pkl
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 🔍 Key Features

* End-to-end ML pipeline using `Pipeline`
* Regularized XGBoost model
* Handles missing and mixed data
* Supports scalable deployment
* Feature importance extraction

---

## 💡 Future Improvements

* 🔍 SHAP explainability (understand why employees leave)
* ⚖️ Handle class imbalance (`scale_pos_weight`)
* 🎯 Hyperparameter tuning (RandomizedSearchCV / Optuna)
* 🌐 Deployment using FastAPI / Streamlit
* 📊 Dashboard for HR insights

---

## 🧪 Business Impact

* Predict employee churn early
* Improve retention strategies
* Optimize HR decision-making
* Reduce operational costs

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork the repository and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

* XGBoost open-source community
* Scikit-learn contributors

---

## 📬 Contact

For questions or collaboration, feel free to reach out.

---
