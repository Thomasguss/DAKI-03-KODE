# ---------------------------------------
# DataVisualisering8.py
# 8-søjle-graf: køn × alder × COVID-status
# ---------------------------------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------
# 1. Indlæs og klargør data
# ---------------------------------------

df = pd.read_csv("CovidData.csv")

# Rens kolonnenavne
df.columns = df.columns.str.strip().str.upper()

# Kendte missing values
df = df.replace([97, 98, 99, '9999-99-99'], pd.NA)

# DIED-kolonne
df['DIED'] = df['DATE_DIED'].notna().astype(int)

# COVID-status
df['COVID_CONFIRMED'] = df['CLASIFFICATION_FINAL'].apply(
    lambda x: 1 if x in [1, 2, 3] else (
        0 if x in [4, 5, 6, 7] else pd.NA
    )
)
df = df.dropna(subset=['COVID_CONFIRMED'])

# Aldersgrupper
df['AGE_GROUP'] = np.where(df['AGE'] < 60, '<60', '≥60')

# ---------------------------------------
# 2. BEREGN DØDELIGHED FOR 8 KATEGORIER
# ---------------------------------------

overall = df.groupby(['SEX', 'AGE_GROUP', 'COVID_CONFIRMED'])['DIED'].mean().reset_index()
overall['Dødelighed (%)'] = overall['DIED'] * 100

# Labels (samme logik som sektion 2 → virker!)
sex_labels = {1: "Kvinder", 2: "Mænd"}
covid_labels = {0: "Ikke smittet", 1: "Smittet"}

overall['Køn'] = overall['SEX'].map(sex_labels)
overall['COVID-status'] = overall['COVID_CONFIRMED'].map(covid_labels)

# Lav én samlet kategori til x-aksen
overall['Kategori'] = (
    overall['Køn'] + " – " + overall['AGE_GROUP'] + " – " + overall['COVID-status']
)

# Sørg for fast rækkefølge (8 søjler)
kategori_order = [
    "Kvinder – <60 – Ikke smittet",
    "Kvinder – ≥60 – Ikke smittet",
    "Mænd – <60 – Ikke smittet",
    "Mænd – ≥60 – Ikke smittet",
    "Kvinder – <60 – Smittet",
    "Kvinder – ≥60 – Smittet",
    "Mænd – <60 – Smittet",
    "Mænd – ≥60 – Smittet",
]

overall['Kategori'] = pd.Categorical(
    overall['Kategori'],
    categories=kategori_order,
    ordered=True
)

# ---------------------------------------
# 3. GRAF MED 8 SØJLER
# ---------------------------------------

plt.figure(figsize=(16, 6))
ax = sns.barplot(
    data=overall,
    x='Kategori',
    y='Dødelighed (%)',
    palette='Set2',
    errorbar=None
)

plt.title("Dødelighed (%) fordelt på køn, alder og COVID-status", fontsize=14)
plt.xlabel("Gruppe")
plt.ylabel("Dødelighed (%)")
plt.xticks(rotation=45, ha='right')

# Tallene ovenpå søjlerne
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f")

plt.tight_layout()
plt.show()

# ---------------------------------------
# SLUT PÅ FIL
# ---------------------------------------
