# --- Importer nødvendige biblioteker ---
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- Indlæs data ---
df = pd.read_csv("CovidData.csv")

# --- Rens kolonnenavne ---
df.columns = df.columns.str.strip().str.upper()

# --- Erstat kendte 'missing values' med NaN ---
df = df.replace([97, 98, 99, '9999-99-99'], pd.NA)

# --- Opret kolonne for dødelighed ---
df['DIED'] = df['DATE_DIED'].notna().astype(int)

# --- Opret kolonne for COVID-status ---
# 1–3 = bekræftet COVID, 4–7 = ikke smittet
df['COVID_CONFIRMED'] = df['CLASIFFICATION_FINAL'].apply(
    lambda x: 1 if x in [1, 2, 3] else (0 if x in [4, 5, 6, 7] else pd.NA)
)
df = df.dropna(subset=['COVID_CONFIRMED'])

# --- Opret aldersgrupper ---
df['AGE_GROUP'] = np.where(df['AGE'] < 60, '<60', '≥60')

# --- 1. Basisanalyse ---
print("\nAntal rækker og kolonner:", df.shape)
print("\nFørste rækker i datasættet:\n", df.head())

# --- 2. Fordeling af køn og COVID-status ---
plt.figure(figsize=(6,4))
ax = sns.countplot(data=df, x='SEX', hue='COVID_CONFIRMED')
plt.title("Kønsfordeling opdelt på COVID-status")
plt.xlabel("Køn")
plt.ylabel("Antal patienter")
plt.xticks([0, 1], ['Kvinder (1)', 'Mænd (2)'])
plt.legend(title='COVID-status', labels=['Ikke smittet', 'Smittet'])
for container in ax.containers:
    ax.bar_label(container, fmt="%d", label_type="edge")
plt.tight_layout()
plt.show()

# --- 3. Aldersfordeling med matchende farver i legend ---
plt.figure(figsize=(8,5))
sns.histplot(
    data=df,
    x='AGE',
    hue='COVID_CONFIRMED',
    bins=30,
    kde=True,
    element='bars',   # Ændret fra 'step' til 'bars'
    palette={0: '#1f77b4', 1: '#ff7f0e'}  # valgfri farver, matcher legend
)
plt.title("Aldersfordeling opdelt på COVID-status")
plt.xlabel("Alder")
plt.ylabel("Antal patienter")
plt.legend(title='COVID-status', labels=['Smittet', 'Ikke Smittet'])
plt.tight_layout()
plt.show()


# --- 4. Dødelighedsrate ---
death_rate = df['DIED'].mean() * 100
print(f"\nSamlet dødelighedsrate: {death_rate:.2f}%")

plt.figure(figsize=(7,5))
ax = sns.countplot(data=df, x='DIED', hue='COVID_CONFIRMED')
plt.title("Antal døde vs. overlevede opdelt på COVID-status")
plt.xticks([0,1], ['Overlevede', 'Døde'])
plt.legend(title='COVID-status', labels=['Ikke smittet', 'Smittet'])
for container in ax.containers:
    ax.bar_label(container, fmt="%d", label_type="edge")
plt.tight_layout()
plt.show()

# --- 5. Sammenhæng mellem alder og dødelighed ---
plt.figure(figsize=(8,5))
palette = {'Ikke smittet': "#0062FF", 'Smittet': "#FF5E00"}  
df['COVID-status'] = df['COVID_CONFIRMED'].map({0: 'Ikke smittet', 1: 'Smittet'})
ax = sns.boxplot(data=df, x='DIED', y='AGE', hue='COVID-status', palette=palette)
plt.title("Alder vs. dødelighed opdelt på COVID-status")
plt.xticks([0,1], ['Overlevede', 'Døde'])
plt.legend(title='COVID-status')
plt.tight_layout()
plt.show()

# --- 6. Komorbiditeter ---
comorbidities = ['DIABETES', 'HYPERTENSION', 'OBESITY', 'RENAL_CHRONIC',
                 'CARDIOVASCULAR', 'COPD', 'ASTHMA', 'INMSUPR', 'TOBACCO']

existing_cols = [col for col in comorbidities if col in df.columns]
print(f"\nKolonner fundet i datasættet: {existing_cols}")

for col in existing_cols:
    df[col] = df[col].replace({2: 0})

df[existing_cols] = df[existing_cols].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=existing_cols)

# --- Antal sygdomme ---
df['NUM_COMORBIDITIES'] = df[existing_cols].sum(axis=1)
df['MULTIPLE_COMORBIDITIES'] = (df['NUM_COMORBIDITIES'] >= 2).astype(int)

# --- 7. Relativ risiko pr. sygdom og COVID-status ---
risk_ratios = []

for disease in existing_cols + ['MULTIPLE_COMORBIDITIES']:
    for covid_status in [0, 1]:
        subset = df[df['COVID_CONFIRMED'] == covid_status]
        with_disease = subset[subset[disease] == 1]
        without_disease = subset[subset[disease] == 0]
        mortality_with = with_disease['DIED'].mean() * 100
        mortality_without = without_disease['DIED'].mean() * 100
        rr = mortality_with / mortality_without if mortality_without > 0 else np.nan
        risk_ratios.append({
            'COVID-status': 'Smittet' if covid_status == 1 else 'Ikke smittet',
            'Sygdom': disease if disease != 'MULTIPLE_COMORBIDITIES' else '≥2 sygdomme',
            'Dødelighed med sygdom (%)': mortality_with,
            'Dødelighed uden sygdom (%)': mortality_without,
            'Relativ risiko (×)': rr
        })

risk_df = pd.DataFrame(risk_ratios)
print("\n📊 Relativ risiko for død pr. sygdom opdelt på COVID-status:\n", risk_df.round(2))

# --- 8. Dødelighed med/uden sygdom pr. COVID-status ---
plt.figure(figsize=(14,6))
ax = sns.barplot(data=risk_df, x='Sygdom', y='Dødelighed med sygdom (%)', hue='COVID-status', errorbar=None)
plt.title("Dødelighed (%) blandt patienter med sygdom opdelt på COVID-status")
plt.ylabel("Dødelighed (%)")
plt.xticks(rotation=45)
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f", label_type="edge")
plt.tight_layout()
plt.show()


# --- 9. Relativ risiko for død ved COVID-19 pr. sygdom (korrigeret) ---

cross_risk = []

for disease in existing_cols + ['MULTIPLE_COMORBIDITIES']:
    with_disease = df[df[disease] == 1]
    # Dødelighed for smittede og ikke-smittede med denne sygdom
    mortality_covid = with_disease[with_disease['COVID_CONFIRMED'] == 1]['DIED'].mean() * 100
    mortality_noncovid = with_disease[with_disease['COVID_CONFIRMED'] == 0]['DIED'].mean() * 100
    rr = mortality_covid / mortality_noncovid if mortality_noncovid > 0 else np.nan
    cross_risk.append({
        'Sygdom': disease if disease != 'MULTIPLE_COMORBIDITIES' else '≥2 sygdomme',
        'Død smittet (%)': mortality_covid,
        'Død ikke smittet (%)': mortality_noncovid,
        'Relativ risiko (×)': rr
    })

cross_df = pd.DataFrame(cross_risk)
print("\n📊 Relativ risiko for død ved COVID-19 pr. sygdom (smittet vs ikke smittet):\n", cross_df.round(2))

# --- Graf: Dødelighed for smittede vs ikke smittede pr. sygdom ---
plt.figure(figsize=(14,6))
ax = sns.barplot(data=cross_df.melt(id_vars='Sygdom',
                                    value_vars=['Død smittet (%)','Død ikke smittet (%)'],
                                    var_name='Gruppe', value_name='Dødelighed (%)'),
                 x='Sygdom', y='Dødelighed (%)', hue='Gruppe', errorbar=None)
plt.title("Dødelighed (%) for smittede vs. ikke smittede pr. sygdom")
plt.xticks(rotation=45)
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f", label_type="edge")
plt.tight_layout()
plt.show()

# --- Graf: Relativ risiko (COVID-smittet / ikke smittet) ---
plt.figure(figsize=(14,6))
ax = sns.barplot(data=cross_df, x='Sygdom', y='Relativ risiko (×)', color="#0062FF", errorbar=None)
plt.axhline(1, color='red', linestyle='--', label='Samme risiko')
plt.title("Relativ risiko for død ved COVID-19 pr. sygdom (smittet vs. ikke smittet)")
plt.ylabel("Relativ risiko (×)")
plt.xticks(rotation=45)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge")
plt.legend()
plt.tight_layout()
plt.show()


# --- 10. Komorbiditet × alder × COVID ---
summary_list = []
analysis_cols = existing_cols + ['MULTIPLE_COMORBIDITIES']

for disease in analysis_cols:
    for age_group in ['<60', '≥60']:
        for covid_status in [0, 1]:
            subset = df[(df['AGE_GROUP'] == age_group) & (df['COVID_CONFIRMED'] == covid_status)]
            with_disease = subset[subset[disease] == 1]
            without_disease = subset[subset[disease] == 0]
            summary_list.append({
                'COVID-status': 'Smittet' if covid_status == 1 else 'Ikke smittet',
                'Sygdom': disease if disease != 'MULTIPLE_COMORBIDITIES' else '≥2 sygdomme',
                'Aldersgruppe': age_group,
                'Dødelighed med sygdom (%)': with_disease['DIED'].mean() * 100,
                'Dødelighed uden sygdom (%)': without_disease['DIED'].mean() * 100,
                'Relativ risiko (×)': (with_disease['DIED'].mean() / without_disease['DIED'].mean()) if without_disease['DIED'].mean() != 0 else float('nan')

            })

age_comorbidity_df = pd.DataFrame(summary_list)
print("\n📊 Komorbiditet × alder × COVID-status:\n", age_comorbidity_df.round(2))


# --- Heatmap pr. COVID-status ---
for covid_status, label in [(0, "Ikke smittet"), (1, "Smittet")]:
    heatmap_df = age_comorbidity_df[age_comorbidity_df['COVID-status'] == label] \
        .pivot(index='Sygdom', columns='Aldersgruppe', values='Dødelighed med sygdom (%)')
    plt.figure(figsize=(8,6))
    sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap="Reds", cbar_kws={'label': 'Dødelighed (%)'})
    plt.title(f"Dødelighed (%) blandt patienter med sygdom – {label}")
    plt.ylabel("Sygdom")
    plt.xlabel("Aldersgruppe")
    plt.tight_layout()
    plt.show()

# --- 11. Fordeling af antal sygdomme pr. aldersgruppe og COVID-status ---
df['COMORB_CAT'] = pd.cut(
    df['NUM_COMORBIDITIES'],
    bins=[-1, 0, 1, df['NUM_COMORBIDITIES'].max()],
    labels=['Ingen sygdom', 'Én sygdom', '≥2 sygdomme']
)
comorb_age_counts = df.groupby(['AGE_GROUP', 'COVID_CONFIRMED', 'COMORB_CAT']).size().reset_index(name='Antal')
comorb_age_counts['COVID-status'] = comorb_age_counts['COVID_CONFIRMED'].map({0: 'Ikke smittet', 1: 'Smittet'})

plt.figure(figsize=(9,5))
ax = sns.barplot(data=comorb_age_counts, x='AGE_GROUP', y='Antal', hue='COMORB_CAT',errorbar=None)
plt.title("Fordeling af patienter med 0, 1 eller ≥2 sygdomme pr. aldersgruppe")
for container in ax.containers:
    ax.bar_label(container, fmt="%d", label_type="edge")
plt.tight_layout()
plt.show()
