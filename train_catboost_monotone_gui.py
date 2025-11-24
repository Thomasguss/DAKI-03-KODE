import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score
)
from catboost import CatBoostClassifier

# ============================================================
# 1. Load og forbered data
# ============================================================
df = pd.read_csv("CovidData.csv")
df.replace({97: np.nan, 98: np.nan, 99: np.nan}, inplace=True)

# Døds-variabel
df["DATE_DIED"] = df["DATE_DIED"].astype(str)
df["DIED"] = (df["DATE_DIED"] != "9999-99-99").astype(int)

# Binære kolonner
binary_cols = [
    "SEX", "DIABETES", "HIPERTENSION", "OBESITY", "COPD", "ASTHMA",
    "CARDIOVASCULAR", "RENAL_CHRONIC", "INMSUPR", "TOBACCO",
    "OTHER_DISEASE", "PREGNANT"
]
for col in binary_cols:
    df[col] = df[col].replace({1: 1, 2: 0})

df["PREGNANT"] = df["PREGNANT"].fillna(0)
df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")

# ============================================================
# 2. Features til GUI-modellen
# ============================================================
# Enkel, gennemskuelig model:
# - AGE (kontinuert)
# - SEX (0/1)
# - 11 sygdomme (0/1)
features = [
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
    "OTHER_DISEASE",
    "PREGNANT",
]

mask = df[features + ["DIED"]].notna().all(axis=1)
clean = df[mask]

X = clean[features]
y = clean["DIED"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# ============================================================
# 3. CatBoost med monotone constraints
# ============================================================
# Feature-orden:
# [AGE, SEX, DIABETES, HIPERTENSION, OBESITY, COPD, ASTHMA,
#  CARDIOVASCULAR, RENAL_CHRONIC, INMSUPR, TOBACCO, OTHER_DISEASE, PREGNANT]
#
# Vi vil have:
# - AGE:          monotont stigende (1)
# - SEX:          fri (0)
# - alle sygdomme: monotont stigende (1)
monotone_constraints = [1, 0] + [1] * 11  # total length = 13

n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
scale_pos_weight = n_neg / n_pos

cat = CatBoostClassifier(
    iterations=800,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    scale_pos_weight=scale_pos_weight,
    random_seed=42,
    verbose=False,
    monotone_constraints=monotone_constraints
)

cat.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_val),
    verbose=False
)

# ============================================================
# 4. Evaluer på test
# ============================================================
y_prob = cat.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)

def find_optimal_threshold(y_true, y_prob):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_youden = 0.5, -1
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens = recall_score(y_true, y_pred, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        youden = sens + spec - 1
        if youden > best_youden:
            best_youden = youden
            best_t = t
    return best_t, best_youden

best_t, best_youden = find_optimal_threshold(y_test, y_prob)

print("\n===== Monoton GUI-model – CatBoost =====")
print(f"AUC (test): {auc:.3f}")
print(f"Optimal threshold (Youden, test): {best_t:.3f} (score={best_youden:.3f})")

# ============================================================
# 5. Gem modellen til GUI'en
# ============================================================
cat.save_model("catboost_covid_gui_monotone.cbm")
print("\n[INFO] GUI-model gemt som 'catboost_covid_gui_monotone.cbm'")
