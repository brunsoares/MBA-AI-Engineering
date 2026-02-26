import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

plt.style.use('default')
pd.set_option('display.max_columns', None)

# Carregando o conjunto de dados base
wine = load_wine()

x = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name='target')

df = pd.concat([x, y], axis=1)
df.head()

# Criando coluna temporal
df['data_registro'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
df[['data_registro']].head()

# Geração de features temporais
df['ano_registro'] = df['data_registro'].dt.year
df['mes_registro'] = df['data_registro'].dt.month
df['dia_semana'] = df['data_registro'].dt.dayofweek
df['is_fim_de_semana'] = df['dia_semana'].isin([5, 6]).astype(int)

df[['data_registro', 'ano_registro', 'mes_registro', 'dia_semana', 'is_fim_de_semana']].head()

# Explorando novas features temporais
print('Registros por mes:')
print(df['mes_registro'].value_counts().sort_index())

print('\nRegistros por dia da semana:')
print(df['dia_semana'].value_counts().sort_index())

print('\nRegistros por fim de semana:')
print(df['is_fim_de_semana'].value_counts(normalize=True))

# Geração de features numericas derivadas
df['alcool_por_acido_malico'] = df['alcohol'] / (df['malic_acid'] + 1e-6)
df['intensidade_menos_matiz'] = df['color_intensity'] - df['hue']
df['fenol_total'] = df['total_phenols'] / df['flavanoids']

df[['alcohol', 'malic_acid', 'alcool_por_acido_malico',
    'color_intensity', 'hue', 'intensidade_menos_matiz',
    'total_phenols', 'flavanoids', 'fenol_total']].head()

# Analise das novas features numericas
df[['alcool_por_acido_malico', 'intensidade_menos_matiz', 'fenol_total']].describe().T

# Visualizando as distribuições das novas features
df[['alcool_por_acido_malico', 'intensidade_menos_matiz', 'fenol_total']].hist(bins=15, figsize=(10, 4))
plt.suptitle('Distribuição de novas features numericas',y=1.02)
plt.tight_layout()
plt.show()

# Agregações por grupo
agg_class = df.groupby('target').agg({
    'alcohol': 'mean',
    'color_intensity': 'mean',
    'proline': 'mean',
}).rename(columns={
    'alcohol': 'alcohol_mean',
    'color_intensity': 'color_intensity_mean',
    'proline': 'proline_mean'
})

agg_class

# Incorporando agregações como novas features
df = df.merge(agg_class, left_on='target', right_index=True, how='left')

df[['target', 'alcohol_mean', 'color_intensity_mean', 'proline_mean']].head()

# Visão geral das features originais e geradas
print('Formato final do dataframe:', df.shape)

df.head()