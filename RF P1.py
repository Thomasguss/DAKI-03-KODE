import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# Load data
# =========================================================
df = pd.read_csv("/Users/thomas/Desktop/AI Programmering/P1 Prosjekt/Covid Data 2.csv")

# Replace missing codes
df.replace({97: np.nan, 98: np.nan, 99: np.nan}, inplace=True)

# =========================================================
# Define death variable
# =========================================================
df["DATE_DIED"] = df["DATE_DIED"].astype(str)
df["DIED"] = (df["DATE_DIED"] != "9999-99-99").astype(int)

# =========================================================
# FILTER: Only COVID positive (1, 2, 3)
# =========================================================
df_pos = df[df["CLASIFFICATION_FINAL"].isin([1,2,3])].copy()

# =========================================================
# Features (same as LR)
# =========================================================
feature_cols = [
    "AGE",
    "SEX",
    "DIABETES",
    "HIPERTENSION",
    "OBESITY",
    "COPD",
    "ASTHMA",
    "CARDIOVASCULAR",
    "RENAL_CHRONIC",
    "INMSUPR",
    "TOBACCO",
    "OTHER_DISEASE"
]

binary_columns = [c for c in feature_cols if c != "AGE"]

# Convert 1/2 → 1/0
for col in binary_columns:
    df_pos[col] = df_pos[col].replace({1: 1, 2: 0})

# Ensure AGE is numeric
df_pos["AGE"] = pd.to_numeric(df_pos["AGE"], errors="coerce")

# =========================================================
# Prepare X and y
# =========================================================
X = df_pos[feature_cols].copy()
y = df_pos["DIED"].copy()

# Remove missing
mask = X.notnull().all(axis=1) & y.notnull()
X = X[mask]
y = y[mask]

# As floats
X = X.astype(float)


# =========================================================
# 80 / 10 / 10 SPLIT
# =========================================================

# Først: 80% train, 20% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Deretter: del temp i 10% validation og 10% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)



# =========================================================
# Random Forest
# =========================================================
rf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=10,
    min_samples_split=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

# =========================================================
# Evaluation
# =========================================================
print("AUC-ROC:", roc_auc_score(y_test, y_prob))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# =========================================================
# Feature importance
# =========================================================
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nRandom Forest Feature Importance:\n")
print(importance_df)
