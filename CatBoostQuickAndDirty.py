# catboost_80_10_10_holdout.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from catboost import CatBoostClassifier, Pool

# ---------------------------------------------------------
# 1) Indlæs data og standardiser kolonnenavne
# ---------------------------------------------------------
df = pd.read_csv(r"C:\Users\Chris\Desktop\P1\CovidData.csv")
df.columns = df.columns.str.lower()

# Kun bekræftede covid-tilfælde (1–3)
df = df[df["clasiffication_final"].isin([1, 2, 3])].copy()

# Target: death = 1 hvis patienten er død
df["death"] = (df["date_died"] != "9999-99-99").astype(int)

# ---------------------------------------------------------
# 2) Missing values (97, 98, 99 -> np.nan)
# ---------------------------------------------------------
missing_codes = [97, 98, 99]

cols_with_missing_codes = [
    "diabetes", "copd", "asthma", "inmsupr", "hipertension",
    "cardiovascular", "renal_chronic", "other_disease",
    "obesity", "tobacco", "pneumonia",
    "sex", "pregnant",
    "clasiffication_final", "usmer", "medical_unit",
    "patient_type", "intubed", "icu"
]

for col in cols_with_missing_codes:
    if col in df.columns:
        df[col] = df[col].replace(missing_codes, np.nan)

df = df.replace({pd.NA: np.nan})

# ---------------------------------------------------------
# 3) Features og mål
# ---------------------------------------------------------
feature_cols = [
    "age",
    "diabetes", "copd", "asthma", "inmsupr", "hipertension",
    "cardiovascular", "renal_chronic", "other_disease",
    "obesity", "tobacco", "pneumonia",
    "sex", "pregnant",
    "clasiffication_final", "usmer", "medical_unit",
    "patient_type", "intubed", "icu"
]

feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols]
y = df["death"]

# Kategoriske features = objekt-/kategori-typer (ofte tomt her, og det er fint)
cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

# ---------------------------------------------------------
# 4) 80 / 10 / 10 SPLIT
# ---------------------------------------------------------
# Først: 80% train, 20% temp (val+test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Så: de 20% deles til 10% val, 10% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=42
)

print(f"Train size: {len(X_train)}")
print(f"Val size:   {len(X_val)}")
print(f"Test size:  {len(X_test)}")
print("-" * 60)

# 👉 Herfra bruger vi KUN train + val. X_test / y_test gemmes til senere.

train_pool = Pool(X_train, y_train, cat_features=cat_features)
val_pool   = Pool(X_val,   y_val,   cat_features=cat_features)

# ---------------------------------------------------------
# 5) CatBoost-model (tunes mod val, ikke test)
# ---------------------------------------------------------
model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="Accuracy",
    iterations=1000,
    depth=6,
    learning_rate=0.1,
    random_seed=42,
    class_weights=[1.0, 5.0],   # ekstra vægt på død=1
    verbose=100
)

model.fit(
    train_pool,
    eval_set=val_pool,          # early stopping bruger KUN validation
    early_stopping_rounds=100
)

# ---------------------------------------------------------
# 6) Evaluering på VALIDERINGSSÆT (test røres ikke)
# ---------------------------------------------------------
y_val_pred = model.predict(X_val).astype(int)
acc_val = accuracy_score(y_val, y_val_pred)

print("\nConfusion matrix (validering):")
print(confusion_matrix(y_val, y_val_pred))

print("\nClassification report (validering):")
print(classification_report(y_val, y_val_pred))

print(f"\nAccuracy på valideringssættet: {acc_val:.3f}")

# ---------------------------------------------------------
# 7) Risiko-grupper på valideringssæt
# ---------------------------------------------------------
proba_val_death = model.predict_proba(X_val)[:, 1]

risk_group_val = pd.cut(
    proba_val_death,
    bins=[0.0, 0.10, 0.30, 1.0],
    labels=["Lav risiko", "Mellem risiko", "Høj risiko"]
)

results_val = pd.DataFrame({
    "true_death": y_val.values,
    "pred_label": y_val_pred,
    "proba_death": proba_val_death,
    "risk_group": risk_group_val
})

print("\nEksempel på risikoklassificering (validering, første 20):")
print(results_val.head(20))

# ---------------------------------------------------------
# 8) Test-sættet er stadig gemt til senere
# ---------------------------------------------------------
# X_test, y_test ligger klar til endelig, unbiased evaluering
# ---------------------------------------------------------
# 9) Opsummering af risikogrupper (valideringssættet)
# ---------------------------------------------------------
risk_summary = results_val["risk_group"].value_counts().sort_index()

print("\nFordeling af risikogrupper (validering):")
print(risk_summary)
