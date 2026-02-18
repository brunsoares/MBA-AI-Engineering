import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

plt.style.use('default')
pd.set_option('display.max_columns', None)

# Carregando um conjunto de dados
wine = load_wine()

x = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name='target')

df = pd.concat([x, y], axis=1)
df.head()

# Inserindo valores para simular uma base com valores incorretos
df_dirty = df.copy()

np.random.seed(42)
sample_lines = np.random.choice(df_dirty.index, size=20, replace=False)
sample_columns = np.random.choice(df_dirty.columns, size=3, replace=False)

for col in sample_columns:
  df_dirty.loc[sample_lines, col] = np.nan

df_dirty = pd.concat([df_dirty, df_dirty.sample(5, random_state=42)], ignore_index=True)

df_dirty.head()

# Inspecionando valores ausentes de duplicatas
print("Formato original:", df.shape)
print("Formato com valores ausentes:", df_dirty.shape)

print("Quantidade de valores ausentes por coluna:")
missing_counts = df_dirty.isna().sum()
print(missing_counts)

print("Percentual de valores ausentes por coluna:")
missing_percentages = (missing_counts / len(df_dirty)) * 100
print(missing_percentages.round(2))

print("Quantidade de duplicatas:")
num_duplicates = df_dirty.duplicated().sum()
print(num_duplicates)

# Tratando valores ausentes
# Remover linhas com mais de 5 valores nulos
limit_nan = 5
df_clean_step1 = df_dirty[df_dirty.isna().sum(axis=1) <= limit_nan].copy()

# Outra forma de remover porém generica, pode não ser bom em determinados cenários
# df_dirty.dropna(inplace=True)

# Imputando dados com a mediana
col_num = df_clean_step1.select_dtypes(include=['float64', 'int64']).columns

for col in col_num:
  mediana = df_clean_step1[col].median()
  df_clean_step1[col].fillna(mediana, inplace=True)

print("Formato após remoção de nulos e imputação de mediana:")
print(df_clean_step1.shape)
print("Verificando valores ausentes presentes:")
print(df_clean_step1.isna().sum().sum())

# Tratando duplicatas
num_duplicates = df_clean_step1.duplicated().sum()
print("Quantidade de duplicatas:", num_duplicates)

df_clean = df_clean_step1.drop_duplicates().copy()

num_duplicates = df_clean.duplicated().sum()
print("Quantidade de duplicatas após remoção:", num_duplicates)

print("Formato final do dataframe limpo:", df_clean.shape)

# EDA Simples antes x depois da limpeza
colunas_example = ['alcohol', 'malic_acid', 'color_intensity']

print("Estatisticas descritivas - Dados originais:")
display(df[colunas_example].describe().T)

print("Estatistica descritiva - Dados sujos:")
display(df_dirty[colunas_example].describe().T)

print("Estatisticas descritivas - Dados limpos:")
display(df_clean[colunas_example].describe().T)

# Visualização gráfica
df_clean[colunas_example].hist(bins=15, figsize=(10,4))
plt.suptitle('Distribuição de algumas features numericas - dados limpos', y=1.02)
plt.tight_layout()
plt.show()
