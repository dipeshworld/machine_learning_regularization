# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from xgboost import XGBClassifier

# 2. LOAD DATA
df = pd.read_csv("data/modified_employee_turnover.csv")

# 3. DATA PREPROCESSING

# Target column
target = "Employee_Turnover"

# Separate features
X = df.drop(columns=[target])
y = df[target]

# Identify column types
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

# Fill missing values
X[num_cols] = X[num_cols].fillna(X[num_cols].median())

for col in cat_cols:
    X[col].fillna(X[col].mode()[0], inplace=True)

# 4. PREPROCESSING PIPELINE
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

# 5. XGBOOST MODEL (WITH REGULARIZATION)
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,

    # 🔥 Regularization (VERY IMPORTANT)
    reg_alpha=1.0,     # L1 regularization
    reg_lambda=1.0,    # L2 regularization

    subsample=0.8,
    colsample_bytree=0.8,

    random_state=42,
    eval_metric="logloss"
)

# 6. PIPELINE
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", xgb_model)
])

# 7. TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 8. TRAIN MODEL
model.fit(X_train, y_train)

# 9. PREDICTIONS
y_pred = model.predict(X_test)

# 10. EVALUATION
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== XGBOOST MODEL PERFORMANCE =====")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 11. FEATURE IMPORTANCE

xgb_clf = model.named_steps["classifier"]
importance = xgb_clf.feature_importances_

print("\nTop Feature Importance (sample):")
print(importance[:10])
