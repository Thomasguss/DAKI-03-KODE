# =========================================================
# IMPORTS
# =========================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score
)
import matplotlib.pyplot as plt
import shap
import optuna

# =========================================================
# 0. LOAD DATA
# =========================================================
df = pd.read_csv("CovidData.csv")

# =========================================================
# 1. FEATURES (blue boxes)
# =========================================================
blue_features = [
    "AGE", "DIABETES", "COPD", "ASTHMA", "INMSUPR", "HIPERTENSION",
    "CARDIOVASCULAR", "RENAL_CHRONIC", "OTHER_DISEASE", "OBESITY",
    "TOBACCO", "SEX", "PREGNANT", "DATE_DIED", "CLASIFFICATION_FINAL"
]

df_blue = df[blue_features].copy()

# =========================================================
# 2. Missing values: convert 97/98/99 → NaN
# =========================================================
df_blue.replace({97: np.nan, 98: np.nan, 99: np.nan}, inplace=True)

# =========================================================
# 3. DATE_DIED → DIED (1=død, 0=i live)
# =========================================================
df_blue["DIED"] = (df_blue["DATE_DIED"] != "9999-99-99").astype(int)
df_blue.drop(columns=["DATE_DIED"], inplace=True)

# =========================================================
# 4. Convert 1/2 → 1/0
# =========================================================
binary_columns = [
    "DIABETES", "COPD", "ASTHMA", "INMSUPR", "HIPERTENSION",
    "CARDIOVASCULAR", "RENAL_CHRONIC", "OTHER_DISEASE",
    "OBESITY", "TOBACCO", "SEX", "PREGNANT"
]

for col in binary_columns:
    df_blue[col] = df_blue[col].replace({1: 1, 2: 0})

# =========================================================
# 5. Ensure numeric types
# =========================================================
df_blue["AGE"] = pd.to_numeric(df_blue["AGE"], errors="coerce")
df_blue["CLASIFFICATION_FINAL"] = pd.to_numeric(df_blue["CLASIFFICATION_FINAL"], errors="coerce")

# =========================================================
# 6. Filter: only COVID positive (1,2,3)
# =========================================================
df_covid = df_blue[df_blue["CLASIFFICATION_FINAL"].isin([1, 2, 3])].copy()
df_covid.drop(columns=["CLASIFFICATION_FINAL"], inplace=True)

print("Antal COVID-smittede:", len(df_covid))
print("Dødsrate:", df_covid["DIED"].mean())

# =========================================================
# 7. Stratified 80/10/10 split
# =========================================================
X = df_covid.drop(columns=["DIED"])
y = df_covid["DIED"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print("\nTrain:", len(X_train), "Val:", len(X_val), "Test:", len(X_test))
# =========================================================
# A) INTERAKTIONS-FEATURES
# =========================================================
def add_interactions(df):
    df = df.copy()

    # Age × comorbidities
    comorb_cols = [
        "DIABETES", "COPD", "ASTHMA", "INMSUPR", "HIPERTENSION",
        "CARDIOVASCULAR", "RENAL_CHRONIC", "OTHER_DISEASE", "OBESITY"
    ]
    
    df["N_COMORB"] = df[comorb_cols].sum(axis=1)
    df["AGE_x_N_COMORB"] = df["AGE"] * df["N_COMORB"]

    # Pairwise clinically-relevant interactions
    df["AGE_x_DIABETES"] = df["AGE"] * df["DIABETES"]
    df["AGE_x_OBESITY"] = df["AGE"] * df["OBESITY"]
    df["AGE_x_HYPERT"] = df["AGE"] * df["HIPERTENSION"]
    df["AGE_x_RENAL"] = df["AGE"] * df["RENAL_CHRONIC"]

    df["DIAB_x_RENAL"] = df["DIABETES"] * df["RENAL_CHRONIC"]
    df["OBESITY_x_HYPERT"] = df["OBESITY"] * df["HIPERTENSION"]

    return df

# Tilføj interaktioner til alle splits
X_train = add_interactions(X_train)
X_val = add_interactions(X_val)
X_test = add_interactions(X_test)


# =========================================================
# D1) SCALE_POS_WEIGHT (class imbalance fix)
# =========================================================
pos = y_train.sum()
neg = len(y_train) - pos
scale_weight = neg / pos      # typisk ca. 6

print("scale_pos_weight =", scale_weight)


# =========================================================
# D2) FOCAL LOSS (kan bruges som objective)
# =========================================================
# gamma = fokus på svære samples
# alpha = vægtning af positive samples
def focal_loss(preds, dtrain, alpha=0.25, gamma=2.0):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))  # sigmoid

    grad = alpha * ((1 - preds)**gamma) * (preds - labels) \
         + gamma * alpha * ((1 - preds)**(gamma - 1)) * (-np.log(preds)) * preds * (1 - preds)

    hess = alpha * ((1 - preds)**gamma) * (preds * (1 - preds)) \
         + gamma * alpha * ((1 - preds)**(gamma - 1)) * (
             -np.log(preds) * preds * (1 - preds) +
             (preds - labels) * (1 - 2 * preds)
         )

    return grad, hess


# =========================================================
# VALG AF OBJECTIVE
# =========================================================
USE_FOCAL = False   # ← du kan sætte denne til True hvis du vil tænde focal loss

if USE_FOCAL:
    obj = focal_loss
else:
    obj = "binary:logistic"   # standard XGBoost loss

# =========================================================
# 8. Baseline XGBoost model
# =========================================================
model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_weight,
    objective=obj,
    eval_metric="auc",
    tree_method="hist",
    n_jobs=-1
)

model.set_params(early_stopping_rounds=50)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

# =========================================================
# 9. Baseline test-evaluering
# =========================================================
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

print("\n=== BASELINE TEST RESULTS ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_pred_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# =========================================================
# 10. SHAP (C)
# =========================================================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

shap.summary_plot(shap_values, X_val, plot_type="dot")
shap.summary_plot(shap_values, X_val, plot_type="bar")

i = 0
shap.force_plot(
    explainer.expected_value,
    shap_values[i, :],
    X_val.iloc[i, :],
    matplotlib=True
)

# =========================================================
# 11. OPTUNA HYPERPARAMETER TUNING (B)
# =========================================================
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),

        # VIGTIGT: brug samme settings som i din "rigtige" model
        "scale_pos_weight": scale_weight,
        "objective": obj,       # binary:logistic ELLER focal_loss afhængig af USE_FOCAL
        "tree_method": "hist",
        "eval_metric": "auc"
    }

    mdl = XGBClassifier(**params)
    mdl.set_params(early_stopping_rounds=50)

    mdl.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    preds = mdl.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)

print("\n=== RUNNING OPTUNA (30 trials) ===")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("\nBest AUC:", study.best_trial.value)
print("Params:", study.best_trial.params)

# =========================================================
# 12. Train final tuned model
# =========================================================
model_opt = XGBClassifier(
    **study.best_trial.params,
    scale_pos_weight=scale_weight,
    objective=obj,
    tree_method="hist",
    eval_metric="auc"
)

model_opt.set_params(early_stopping_rounds=50)

model_opt.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

# =========================================================
# 13. Threshold tuning functions
# =========================================================
def find_threshold_for_recall(y_true, y_proba, target=0.80):
    thresholds = np.linspace(0.0, 0.7, 700)
    best_t, best_diff, best_recall, best_precision = 0, 999, 0, 0

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rec = recall_score(y_true, y_pred)
        diff = abs(rec - target)

        if diff < best_diff:
            best_t = t
            best_diff = diff
            best_recall = rec
            best_precision = precision_score(y_true, y_pred, zero_division=0)

    return best_t, best_recall, best_precision


def find_f1_threshold(y_true, y_proba):
    thresholds = np.linspace(0.0, 0.7, 700)
    best_t, best_f1 = 0, -1

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f = f1_score(y_true, y_pred)
        if f > best_f1:
            best_f1 = f
            best_t = t

    return best_t, best_f1


# =========================================================
# 14. Evaluate tuned model
# =========================================================
y_pred_proba = model_opt.predict_proba(X_test)[:, 1]

# --- A: 80% Recall threshold ---
t80, rec80, prec80 = find_threshold_for_recall(y_test, y_pred_proba, target=0.80)

print("\n=== Optimal Threshold for 80% Recall ===")
print("Threshold:", t80)
print("Recall:", rec80)
print("Precision:", prec80)
print("Confusion Matrix:\n", confusion_matrix(y_test, (y_pred_proba >= t80).astype(int)))

# --- B: F1 optimal threshold ---
t_f1, f1_val = find_f1_threshold(y_test, y_pred_proba)

print("\n=== F1-optimal threshold ===")
print("Threshold:", t_f1)
print("F1-score:", f1_val)

# =========================================================
# 15. Risk groups (D)
# =========================================================
df_risk = X_test.copy()
df_risk["proba"] = y_pred_proba
df_risk["DIED"] = y_test.values


high_thr = t80  # high-risk = threshold for 80% recall
low_thr = t_f1  # medium boundary can be F1 or custom

high_risk = df_risk["proba"] >= high_thr
medium_risk = (df_risk["proba"] >= low_thr) & (df_risk["proba"] < high_thr)
low_risk = df_risk["proba"] < low_thr

df_risk["risk_group"] = np.select(
    [high_risk, medium_risk, low_risk],
    ["High", "Medium", "Low"],
    default="Low"         # FIXED: must be string!
)

print("\n=== Risk group distribution ===")
print(df_risk["risk_group"].value_counts())

# =========================================================
# 16. Hvor mange døde er fanget i HIGH-gruppen?
# =========================================================
total_dead = df_risk["DIED"].sum()
dead_in_high = df_risk[(df_risk["DIED"] == 1) & (df_risk["risk_group"] == "High")].shape[0]

print("\n=== DEAD CAPTURE IN HIGH GROUP ===")
print(f"Totalt døde i test-sæt: {total_dead}")
print(f"Døde i HIGH-gruppen: {dead_in_high}")
print(f"Andel fanget af HIGH-gruppen: {dead_in_high/total_dead:.3f}")
