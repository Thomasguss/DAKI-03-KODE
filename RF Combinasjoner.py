# =========================================================
# IMPORTS
# =========================================================
import pandas as pd
import numpy as np
import itertools

# =========================================================
# 1. LOAD OG GRUNNLEGGENDE DATABEHANDLING
# =========================================================

path = "/Users/thomas/Desktop/AI Programmering/P1 Prosjekt/Covid Data 2.csv"

df = pd.read_csv(path)

# Manglende-koder → NaN
df.replace({97: np.nan, 98: np.nan, 99: np.nan}, inplace=True)

# Lag DIED-variabel (1 = død, 0 = i live)
df["DATE_DIED"] = df["DATE_DIED"].astype(str)
df["DIED"] = (df["DATE_DIED"] != "9999-99-99").astype(int)

# =========================================================
# 2. FILTRER TIL COVID-POSITIVE (CLASIFFICATION_FINAL ∈ {1,2,3})
# =========================================================

df_pos = df[df["CLASIFFICATION_FINAL"].isin([1, 2, 3])].copy()

print("Antall COVID-positive:", len(df_pos))
print("Overordnet dødsrate blant COVID-positive:", df_pos["DIED"].mean())

# =========================================================
# 3. DEFINER KOMORBIDITETER / FEATURES
# =========================================================
# Dette er de samme som i LR/RF-modellene, minus AGE (ikke binær)

combo_features = [
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

# Sørg for at alle disse faktisk finnes i datasettet
missing_cols = [c for c in combo_features if c not in df_pos.columns]
if missing_cols:
    raise KeyError(f"Mangler kolonner i datasettet: {missing_cols}")

# =========================================================
# 4. KODING AV BINÆRE FEATURES (1/2 → 1/0)
# =========================================================

for col in combo_features:
    df_pos[col] = df_pos[col].replace({1: 1, 2: 0})

# (Valgfritt) drop rader med NaN i disse kolonnene + DIED
mask = df_pos[combo_features + ["DIED"]].notna().all(axis=1)
df_pos = df_pos[mask].copy()

print("Antall rader etter rensing:", len(df_pos))
print("Ny dødsrate:", df_pos["DIED"].mean())

# =========================================================
# 5. 2-VEIS KOMBINASJONER: DØDELIGHET
# =========================================================

two_way_rows = []

for f1, f2 in itertools.combinations(combo_features, 2):
    # Gruppér på (f1, f2) og beregn dødsrate og antall
    grouped = df_pos.groupby([f1, f2])["DIED"].agg(["mean", "count"]).reset_index()
    grouped.rename(columns={"mean": "death_rate", "count": "n"}, inplace=True)
    
    # Legg på metadata om hvilke features som er i kombinasjonen
    grouped["features"] = f"{f1} + {f2}"
    # Legg til en logisk tekst for kombinasjonen (0/1-verdier)
    grouped["combo"] = (
        grouped[f1].astype(int).astype(str) + "_" +
        grouped[f2].astype(int).astype(str)
    )
    
    two_way_rows.append(grouped)

# Slå sammen alle 2-veis-kombinasjoner til én stor tabell
two_way_df = pd.concat(two_way_rows, ignore_index=True)

# Sortér etter høyest dødsrate, men filtrer bort veldig små grupper (f.eks. n < 50)
two_way_df_filtered = two_way_df[two_way_df["n"] >= 50].copy()
two_way_df_filtered = two_way_df_filtered.sort_values(by="death_rate", ascending=False)

print("\nTOPP 20 2-VEIS KOMBINASJONER (med minst 50 observasjoner):\n")
print(two_way_df_filtered.head(20))


# =========================================================
# 6. 3-VEIS KOMBINASJONER: DØDELIGHET
# =========================================================

three_way_rows = []

for f1, f2, f3 in itertools.combinations(combo_features, 3):
    grouped = df_pos.groupby([f1, f2, f3])["DIED"].agg(["mean", "count"]).reset_index()
    grouped.rename(columns={"mean": "death_rate", "count": "n"}, inplace=True)
    
    grouped["features"] = f"{f1} + {f2} + {f3}"
    grouped["combo"] = (
        grouped[f1].astype(int).astype(str) + "_" +
        grouped[f2].astype(int).astype(str) + "_" +
        grouped[f3].astype(int).astype(str)
    )
    
    three_way_rows.append(grouped)

three_way_df = pd.concat(three_way_rows, ignore_index=True)

# Filtrer bort små grupper (f.eks. n < 50) og sorter
three_way_df_filtered = three_way_df[three_way_df["n"] >= 50].copy()
three_way_df_filtered = three_way_df_filtered.sort_values(by="death_rate", ascending=False)

print("\nTOPP 20 3-VEIS KOMBINASJONER (med minst 50 observasjoner):\n")
print(three_way_df_filtered.head(20))

# =========================================================
# 7. (VALGFRITT) LAGRE RESULTATENE TIL CSV
# =========================================================

two_way_df_filtered.to_csv("two_way_combinations_death_rate.csv", index=False)
three_way_df_filtered.to_csv("three_way_combinations_death_rate.csv", index=False)

print("\nLagret two_way_combinations_death_rate.csv og three_way_combinations_death_rate.csv")
