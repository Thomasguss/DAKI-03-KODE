import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas.plotting import scatter_matrix

data = pd.read_csv("CovidData.csv")

print(data.head(10))

sampled_data = data.sample(n=10000, random_state=42)
print(data.describe())       # Mean, std, min, max
print(data.info())           # Datatyper og missing values
print(data.isnull().sum())   # Missing values per column

numeric_data = sampled_data.select_dtypes(include='number')

# Histogrammer for numeriske features
numeric_data.hist(bins=50, figsize=(15,10))
plt.tight_layout()
plt.show()

# Boxplots for numeriske features
plt.figure(figsize=(15,10))
sns.boxplot(data=numeric_data)
plt.xticks(rotation=90)
plt.show()

# Korrelationsmatrix og heatmap
corr = numeric_data.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.show()

# Scatter matrix
scatter_matrix(numeric_data, alpha=0.2, figsize=(15,15))
plt.show()