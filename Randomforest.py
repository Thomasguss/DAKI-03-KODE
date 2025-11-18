# =========================================================
# IMPORTS
# =========================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    recall_score, precision_score, f1_score
)

# =========================================================
# 0. LOAD DATA
# =========================================================
df = pd.read_csv("CovidData.csv")

# =========================================================
# 1. SELECT FEATURES
# =========================================================
features = [
    "AGE", "DIABETES", "COPD", "ASTHMA", "INMSUPR", "HIPERTENSION",
    "CARDIOVASCULAR", "RENAL_CHRONIC", "OTHER_DISEASE", "OBESITY",
    "TOBACCO", "SEX", "PREGNANT", "DATE_DIED", "CLASIFFICATION_FINAL"
]

df = df[features].copy()

# =========================================================
# 2. CLEANING
# =========================================================
df.replace({97: np.nan, 98: np.nan, 99: np.nan}, inplace=True)
df["DIED"] = (df["DATE_DIED"] != "9999-99-99").astype(int)
df.drop(columns=["DATE_DIED"], inplace=True)

binary_cols = [
    "DIABETES", "COPD", "ASTHMA", "INMSUPR", "HIPERTENSION",
    "CARDIOVASCULAR", "RENAL_CHRONIC", "OTHER_DISEASE",
    "OBESITY", "TOBACCO", "SEX", "PREGNANT"
]
for col in binary_cols:
    df[col] = df[col].replace({1: 1, 2: 0})

df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")
df["CLASIFFICATION_FINAL"] = pd.to_numeric(df["CLASIFFICATION_FINAL"], errors="coerce")

df = df[df["CLASIFFICATION_FINAL"].isin([1, 2, 3])].copy()
df.drop(columns=["CLASIFFICATION_FINAL"], inplace=True)

print("Antal COVID-smittede:", len(df))
print("Dødsrate:", df["DIED"].mean())

# =========================================================
# 3. TRAIN/VAL/TEST SPLIT
# =========================================================
X = df.drop(columns=["DIED"])
y = df["DIED"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# =========================================================
# 4. RANDOM FOREST MODEL
# =========================================================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    n_jobs=-1,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# =========================================================
# 5. BASELINE @ threshold = 0.5
# =========================================================
y_proba = model.predict_proba(X_test)[:, 1]
y_pred_05 = (y_proba >= 0.5).astype(int)

print("\n=== BASELINE RESULTS (threshold 0.5) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_05))
print("Recall:", recall_score(y_test, y_pred_05))
print("AUC:", roc_auc_score(y_test, y_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_05))

baseline_acc = accuracy_score(y_test, y_pred_05)
baseline_rec = recall_score(y_test, y_pred_05)
baseline_combined = (baseline_acc + baseline_rec) / 2

print("\n=== BASELINE ACC + RECALL ===")
print(f"Accuracy: {baseline_acc:.4f}")
print(f"Recall:   {baseline_rec:.4f}")
print(f"Combined: {baseline_combined:.4f}")

# =========================================================
# 6. BEST ACCURACY + RECALL THRESHOLD
# =========================================================
def find_best_accuracy_recall_threshold(y_true, y_proba):
    thresholds = np.linspace(0, 0.99, 1000)
    best_t = 0
    best_score = -1
    best_acc = 0
    best_rec = 0

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        score = (acc + rec) / 2

        if score > best_score:
            best_score = score
            best_t = t
            best_acc = acc
            best_rec = rec

    return best_t, best_acc, best_rec, best_score

best_t, best_acc, best_rec, best_score = find_best_accuracy_recall_threshold(y_test, y_proba)

print("\n=== BEST ACC+RECALL THRESHOLD ===")
print(f"Best threshold: {best_t:.4f}")
print(f"Accuracy:       {best_acc:.4f}")
print(f"Recall:         {best_rec:.4f}")
print(f"Combined score: {best_score:.4f}")

# =========================================================
# 7. CONFUSION MATRIX — OPTIMAL THRESHOLD
# =========================================================
y_pred_best = (y_proba >= best_t).astype(int)
cm_best = confusion_matrix(y_test, y_pred_best)

print("\n=== CONFUSION MATRIX — BEST ACC+RECALL MODEL ===")
print(cm_best)
print("\nTN, FP, FN, TP =", cm_best.ravel())

# =========================================================
# 8. RISK GROUPS
# =========================================================
def find_threshold_for_recall(y_true, y_proba, target=0.95):
    thresholds = np.linspace(0.0, 0.99, 1000)
    best_t, best_diff, best_rec = 0, 999, 0
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rec = recall_score(y_true, y_pred)
        diff = abs(rec - target)
        if diff < best_diff:
            best_diff = diff
            best_t = t
            best_rec = rec
    return best_t, best_rec


def find_f1_threshold(y_true, y_proba):
    thresholds = np.linspace(0.0, 0.99, 1000)
    best_t, best_f1 = 0, -1
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f = f1_score(y_true, y_pred)
        if f > best_f1:
            best_f1 = f
            best_t = t
    return best_t, best_f1


t95, rec95 = find_threshold_for_recall(y_test, y_proba, target=0.95)
t_f1, f1_val = find_f1_threshold(y_test, y_proba)

low_thr  = min(t95, t_f1)
high_thr = max(t95, t_f1)

print("\n=== RISK MODEL THRESHOLDS ===")
print(f"95% recall threshold: {t95:.4f}")
print(f"F1 threshold:         {t_f1:.4f}")
print(f"Low < {low_thr:.4f} < Medium < {high_thr:.4f} < High")

df_risk = X_test.copy()
df_risk["proba"] = y_proba
df_risk["DIED"] = y_test.values

df_risk["risk_group"] = np.select(
    [
        df_risk["proba"] >= high_thr,
        (df_risk["proba"] >= low_thr) & (df_risk["proba"] < high_thr),
        df_risk["proba"] < low_thr
    ],
    ["High", "Medium", "Low"],
    default="Low"
).astype(str)

print("\n=== RISK GROUP DISTRIBUTION ===")
print(df_risk["risk_group"].value_counts())

total_dead = df_risk["DIED"].sum()

dead_in_risk = df_risk[
    (df_risk["DIED"] == 1) & 
    (df_risk["risk_group"].isin(["Medium", "High"]))
].shape[0]

dead_in_high = df_risk[
    (df_risk["DIED"] == 1) & 
    (df_risk["risk_group"] == "High")
].shape[0]

print("\n=== DEATH CAPTURE ===")
print(f"Total deaths:               {total_dead}")
print(f"Deaths in MED+HIGH:         {dead_in_risk}")
print(f"Capture rate MED+HIGH:      {dead_in_risk / total_dead:.3f}")
print(f"Deaths in HIGH:             {dead_in_high}")
print(f"Capture rate HIGH:          {dead_in_high / total_dead:.3f}")

cm_risk = pd.crosstab(
    df_risk["risk_group"],
    df_risk["DIED"],
    rownames=["Predicted risk group"],
    colnames=["Actual (DIED)"]
)

print("\n=== CONFUSION MATRIX — RISK GROUPS ===")
print(cm_risk)
