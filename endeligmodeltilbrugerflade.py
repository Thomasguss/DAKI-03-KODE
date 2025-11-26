import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve
)
import matplotlib.pyplot as plt
import joblib


# =====================================================================
# DATAFORBEREDELSE
# =====================================================================
def prepare_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df.replace({97: np.nan, 98: np.nan, 99: np.nan}, inplace=True)
    # KUN COVID-SMITTEDE (1, 2, 3)
    df = df[df["CLASIFFICATION_FINAL"].isin([1, 2, 3])]

    df["DATE_DIED"] = df["DATE_DIED"].astype(str)
    df["DIED"] = (df["DATE_DIED"] != "9999-99-99").astype(int)

    binary_cols = [
        "SEX","DIABETES","HIPERTENSION","OBESITY","COPD","ASTHMA",
        "CARDIOVASCULAR","RENAL_CHRONIC","INMSUPR","TOBACCO",
        "OTHER_DISEASE","PREGNANT"
    ]
    for col in binary_cols:
        df[col] = df[col].replace({1:1, 2:0})

    df["PREGNANT"] = df["PREGNANT"].fillna(0)
    df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")

    # Age bins
    df["AGE_CAT"] = pd.cut(
        df["AGE"],
        bins=[0,50,60,70,80,120],
        labels=["<50","50-59","60-69","70-79","80+"],
        right=False
    )
    df = pd.get_dummies(df, columns=["AGE_CAT"], drop_first=True)
    age_dummies = [c for c in df.columns if c.startswith("AGE_CAT_")]

    # Comorbidity count
    comorb_cols = [
        "DIABETES","HIPERTENSION","OBESITY","COPD","ASTHMA",
        "CARDIOVASCULAR","RENAL_CHRONIC","INMSUPR","TOBACCO","OTHER_DISEASE"
    ]
    df["COMORB_COUNT"] = df[comorb_cols].sum(axis=1)
    df["COMORB_CAT"] = df["COMORB_COUNT"].clip(upper=6).astype("category")
    df = pd.get_dummies(df, columns=["COMORB_CAT"], drop_first=True)
    comorb_dummies = [c for c in df.columns if c.startswith("COMORB_CAT_")]

    categorical = binary_cols + age_dummies + comorb_dummies
    numeric = ["AGE"]

    feature_names = numeric + categorical

    mask = df[categorical+numeric+["DIED"]].notna().all(axis=1)
    clean = df[mask]
    # ============================================
# 🔵  KØNSBALANCERING (indsæt dette)
# ============================================
    df_male_dead     = clean[(clean.SEX==1) & (clean.DIED==1)]
    df_male_alive    = clean[(clean.SEX==1) & (clean.DIED==0)]
    df_female_dead   = clean[(clean.SEX==0) & (clean.DIED==1)]
    df_female_alive  = clean[(clean.SEX==0) & (clean.DIED==0)]

    n = min(len(df_male_dead), len(df_male_alive),
            len(df_female_dead), len(df_female_alive))

    balanced_df = pd.concat([
        df_male_dead.sample(n, random_state=42),
        df_male_alive.sample(n, random_state=42),
        df_female_dead.sample(n, random_state=42),
        df_female_alive.sample(n, random_state=42)
    ])

    clean = balanced_df.sample(frac=1, random_state=42)
# ============================================
    X = clean[categorical+numeric]
    y = clean["DIED"]

    # Splits
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, numeric


def build_preprocessor(numeric_cols):
    return ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_cols)],
        remainder="passthrough"
    )


# =====================================================================
# BASE MODEL
# =====================================================================
class BaseModel:
    def __init__(self, name):
        self.name = name
        self.model = None
        self.best_threshold = 0.5
        self.metrics_ = None
        self.y_val_prob_ = None
        self.y_test_prob_ = None

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:,1]

    # ---- Optimal threshold via Youden ----
    def find_optimal_threshold(self, y_true, y_prob):
        thresholds = np.linspace(0.01, 0.99, 99)
        best_t = 0.5
        best_score = -1

        for t in thresholds:
            pred = (y_prob >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()

            sens = recall_score(y_true, pred)
            spec = tn/(tn+fp) if (tn+fp)>0 else 0

            youden = sens + spec - 1

            if youden > best_score:
                best_score = youden
                best_t = t

        self.best_threshold = best_t
        return best_t, best_score

    def evaluate_on_test(self, y_true, y_prob):
        t = self.best_threshold
        pred = (y_prob >= t).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()

        sens = recall_score(y_true, pred)
        spec = tn/(tn+fp) if (tn+fp)>0 else 0
        prec = precision_score(y_true, pred, zero_division=0)
        acc = accuracy_score(y_true, pred)
        f1 = 2*prec*sens/(prec+sens+1e-9)
        auc = roc_auc_score(y_true, y_prob)

        metrics = {
            "Model": self.name,
            "AUC": auc,
            "Threshold": t,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": sens,
            "Specificity": spec,
            "F1": f1,
            "Youden": sens+spec-1,
            "ConfusionMatrix": np.array([[tn,fp],[fn,tp]]),
            "ClassificationReport": classification_report(y_true, pred, digits=3)
        }

        self.metrics_ = metrics
        return metrics

    def get_feature_importances(self, feature_names):
        return None


# =====================================================================
# SPECIFIC MODELS
# =====================================================================
class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__("Logistic Regression")
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")


class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__("Random Forest")
        self.model = RandomForestClassifier(
            n_estimators=300,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

    def get_feature_importances(self, feature_names):
        imp = self.model.feature_importances_
        return pd.DataFrame({"Feature": feature_names, "Importance": imp}).sort_values("Importance", ascending=False)


class CatBoostModel(BaseModel):
    def __init__(self, scale_pos_weight):
        super().__init__("CatBoost")
        self.model = CatBoostClassifier(
            iterations=800,
            learning_rate=0.03,
            depth=6,
            loss_function="Logloss",
            scale_pos_weight=scale_pos_weight,
            eval_metric="AUC",
            random_seed=42,
            verbose=False
        )

    def get_feature_importances(self, feature_names):
        imp = self.model.get_feature_importance()
        return pd.DataFrame({"Feature": feature_names, "Importance": imp}).sort_values("Importance", ascending=False)


# =====================================================================
# MODEL MANAGER
# =====================================================================
class ModelManager:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test,
                 feature_names, scale_pos_weight_cat):

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.feature_names = feature_names

        self.models = [
            LogisticRegressionModel(),
            RandomForestModel(),
            CatBoostModel(scale_pos_weight_cat),
        ]

        self.results = []
        self.y_test_probas = {}
        self.conf_matrices = {}
        self.feature_importances = {}

    def run_all(self):
        for model in self.models:
            print("\n" + "="*60)
            print(model.name.upper())
            print("="*60)

            # ---- Train model ----
            model.fit(self.X_train, self.y_train)

            # ---- Gem CatBoost-model ----
            if model.name == "CatBoost":
                model.model.save_model("catboost_covid_model.cbm")
                print("\n[INFO] CatBoost-model gemt som 'catboost_covid_model.cbm'")

            # ---- Optimize threshold on val ----
            y_val_prob = model.predict_proba(self.X_val)
            best_t, best_youden = model.find_optimal_threshold(self.y_val, y_val_prob)
            print(f"\nOptimal threshold (Youden): {best_t:.3f}")

            # ---- Evaluate on test ----
            y_test_prob = model.predict_proba(self.X_test)
            metrics = model.evaluate_on_test(self.y_test, y_test_prob)

            # Save
            self.results.append({k: metrics[k] for k in metrics if k!="ClassificationReport"})
            self.y_test_probas[model.name] = y_test_prob
            self.conf_matrices[model.name] = metrics["ConfusionMatrix"]

            # Feature importances
            fi = model.get_feature_importances(self.feature_names)
            if fi is not None:
                self.feature_importances[model.name] = fi

            # ---- Print metrics ----
            print("\n--- Testresultater ---")
            for k in ["AUC","Threshold","Accuracy","Precision","Recall","Specificity","F1","Youden"]:
                print(f"{k:12s}: {metrics[k]:.3f}")

            print("\nConfusion matrix:")
            print(metrics["ConfusionMatrix"])

            print("\nClassification Report:")
            print(metrics["ClassificationReport"])

    def create_summary_table(self):
        df = pd.DataFrame(self.results)
        return df[[
            "Model","AUC","Threshold","Accuracy","Precision",
            "Recall","Specificity","F1","Youden"
        ]]

    def plot_roc_curves(self):
        plt.figure(figsize=(8,6))

        for model in self.models:
            y_prob = self.y_test_probas[model.name]
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            auc = roc_auc_score(self.y_test, y_prob)
            plt.plot(fpr, tpr, label=f"{model.name} (AUC={auc:.3f})")

        plt.plot([0,1],[0,1],"--")
        plt.title("ROC-kurver – test-sæt")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.show()

    def plot_confusion_matrices(self):
        n = len(self.models)
        plt.figure(figsize=(5*n,4))

        for i, model in enumerate(self.models,1):
            cm = self.conf_matrices[model.name]

            plt.subplot(1,n,i)
            plt.imshow(cm, cmap="Blues")
            plt.title(model.name)
            plt.colorbar()
            plt.xticks([0,1], ["Pred 0","Pred 1"])
            plt.yticks([0,1], ["True 0","True 1"])

            for r in range(2):
                for c in range(2):
                    plt.text(c,r,cm[r,c],ha="center",va="center")

        plt.tight_layout()
        plt.show()

    def plot_feature_importances(self, top_n=20):
        for name, fi in self.feature_importances.items():
            plt.figure(figsize=(8,6))
            top = fi.head(top_n)
            plt.barh(top["Feature"], top["Importance"])
            plt.gca().invert_yaxis()
            plt.title(f"{name} – Top {top_n} features")
            plt.show()


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, numeric = prepare_data("CovidData.csv")

    pre = build_preprocessor(numeric)
    X_train_p = pre.fit_transform(X_train)
    X_val_p = pre.transform(X_val)
    X_test_p = pre.transform(X_test)

    # Gem scaleren (til PyQt)
    joblib.dump(pre.named_transformers_["num"], "scaler.pkl")

    scale_pos_weight = (len(y_train)-y_train.sum()) / y_train.sum()

    manager = ModelManager(
        X_train_p, y_train.values,
        X_val_p, y_val.values,
        X_test_p, y_test.values,
        feature_names,
        scale_pos_weight
    )

    manager.run_all()

    summary = manager.create_summary_table()
    print("\n\n=== SAMLET TABEL ===")
    print(summary.to_string(index=False))

    manager.plot_roc_curves()
    manager.plot_confusion_matrices()
    manager.plot_feature_importances(top_n=20)
