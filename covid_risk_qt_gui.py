import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QGridLayout,
    QMessageBox, QCheckBox
)
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtCore import Qt

from catboost import CatBoostClassifier

# -------------------------------------------------------------
# Risikoniveau-funktion
# -------------------------------------------------------------
def risk_level(prob):
    """
    Definer grænserne for Lav / Middel / Høj risiko.
    Du kan justere disse hvis du vil.
    """
    if prob < 0.10:
        return "LAV risiko", QColor(0, 180, 0)
    elif prob < 0.25:
        return "MIDDEL risiko", QColor(230, 180, 0)
    else:
        return "HØJ risiko", QColor(220, 0, 0)


class CovidRiskApp(QWidget):
    def __init__(self):
        super().__init__()

        # Load GUI-modellen (monoton CatBoost)
        self.model = CatBoostClassifier()
        self.model.load_model("catboost_covid_gui_monotone.cbm")

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("COVID-19 Dødelighedsrisiko")
        self.setGeometry(200, 200, 800, 550)

        layout = QGridLayout()

        title = QLabel("COVID-19 Dødelighedsrisiko")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title, 0, 0, 1, 2)

        # Alder
        layout.addWidget(QLabel("Alder (år):"), 1, 0)
        self.age_input = QLineEdit()
        self.age_input.setPlaceholderText("Indtast alder i år (fx 65)")
        layout.addWidget(self.age_input, 1, 1)

        # Køn
        self.sex_checkbox = QCheckBox("Køn: Mand (Ja = mand, Nej = kvinde)")
        layout.addWidget(self.sex_checkbox, 2, 0, 1, 2)

        # Komorbiditeter
        layout.addWidget(QLabel("Komorbiditeter (Ja = til stede):"), 3, 0, 1, 2)

        self.disease_checkboxes = {}

        disease_labels = {
            "DIABETES": "Diabetes",
            "HIPERTENSION": "Hypertension",
            "OBESITY": "Overvægt",
            "COPD": "KOL",
            "ASTHMA": "Astma",
            "CARDIOVASCULAR": "Hjerte-kar sygdom",
            "RENAL_CHRONIC": "Nyresygdom",
            "INMSUPR": "Immunosupprimeret",
            "TOBACCO": "Ryger",
            "OTHER_DISEASE": "Anden sygdom",
            "PREGNANT": "Gravid",
        }

        row = 4
        for key, label in disease_labels.items():
            cb = QCheckBox(label)
            self.disease_checkboxes[key] = cb
            layout.addWidget(cb, row, 0, 1, 2)
            row += 1

        # Beregn-knap
        self.button = QPushButton("Beregn risiko")
        self.button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.button.clicked.connect(self.compute_risk)
        layout.addWidget(self.button, row, 0, 1, 2)

        # Resultat-label
        self.result_label = QLabel("")
        self.result_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result_label, row + 1, 0, 1, 2)

        self.setLayout(layout)

    def compute_risk(self):
        # Alder
        try:
            age = float(self.age_input.text())
        except ValueError:
            QMessageBox.critical(self, "Fejl", "Alder skal være et gyldigt tal.")
            return

        if not (0 <= age <= 120):
            QMessageBox.critical(self, "Fejl", "Alder skal være mellem 0 og 120.")
            return

        # Køn: 1 = mand, 0 = kvinde
        sex = 1 if self.sex_checkbox.isChecked() else 0

        # Sygdomme (0/1)
        disease_values = {
            key: int(cb.isChecked())
            for key, cb in self.disease_checkboxes.items()
        }

        # Feature-orden SKAL matche træningen:
        # [AGE, SEX, DIABETES, HIPERTENSION, OBESITY, COPD, ASTHMA,
        #  CARDIOVASCULAR, RENAL_CHRONIC, INMSUPR, TOBACCO, OTHER_DISEASE, PREGNANT]
        features_order = [
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

        row = {
            "AGE": age,
            "SEX": sex,
            **disease_values
        }

        X = np.array([[row[col] for col in features_order]], dtype=float)

        # Forudsig sandsynlighed
        prob = float(self.model.predict_proba(X)[0, 1])

        # Risikoniveau
        level, color = risk_level(prob)

        palette = self.result_label.palette()
        palette.setColor(QPalette.ColorRole.WindowText, color)
        self.result_label.setPalette(palette)

        self.result_label.setText(
            f"Risiko: {level}\nSandsynlighed for død: {prob*100:.2f}%"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CovidRiskApp()
    window.show()
    sys.exit(app.exec())
