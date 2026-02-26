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

# Geração de features numericas derivadas
df['alcool_x_intensidade'] = df['alcohol'] * df['color_intensity']
df['proline_por_alcool'] = df['proline'] / (df['alcohol'] + 1e-6)
df['flavanoid_menos_nonflavanoid'] = df['flavanoids'] - df['nonflavanoid_phenols']

df[['alcohol', 'color_intensity', 'alcool_x_intensidade', 
    'proline', 'proline_por_alcool', 'flavanoids', 'nonflavanoid_phenols',
    'flavanoid_menos_nonflavanoid']].head()

# Analise das novas features
df[['alcool_x_intensidade', 'proline_por_alcool', 'flavanoid_menos_nonflavanoid']].describe().T

# Visualização distribuição das features
df[['alcool_x_intensidade', 'proline_por_alcool', 'flavanoid_menos_nonflavanoid']].hist(bins=15, figsize=(10,4))
plt.suptitle('Distribuição das novas features', y=1.02)
plt.tight_layout()
plt.show()

# Bins e categorias derivadas
bins_intensity = [
    df['color_intensity'].min() - 0.01,
    df['color_intensity'].quantile(1/3),
    df['color_intensity'].quantile(2/3),
    df['color_intensity'].max() + 0.01
]
labels_intensity = ['Baixo', 'Média', 'Alta']

df['categoria_intensidade'] = pd.cut(df['color_intensity'], bins=bins_intensity, labels=labels_intensity)
df[['color_intensity', 'categoria_intensidade']].head()

# Feature de agrupamento com categorias derivadas
agg_faixa = df.groupby('categoria_intensidade').agg({
    'alcool_x_intensidade': ['mean', 'count'],
})
agg_faixa

# incorporando agregações
agg_faixa.columns = ['media_intensidade', 'contagem_intensidade']

df = df.merge(agg_faixa, left_on='categoria_intensidade', right_index=True, how='left')

df[['color_intensity','categoria_intensidade', 'media_intensidade', 'contagem_intensidade']].head()

# Agregações temporais
df = df.sort_values('data_registro').reset_index(drop=True)

df['media_movel_alcool_7d'] = df['alcohol'].rolling(window=7, min_periods=1).mean()
df['soma_movel_alcool_7d'] = df['alcohol'].rolling(window=7, min_periods=1).sum()

df[['data_registro','alcohol', 'media_movel_alcool_7d', 'soma_movel_alcool_7d']].head()

# Visualizando agregações temporais
plt.figure(figsize=(10,4))
plt.plot(df['data_registro'], df['alcohol'], label='Alcool diário')
plt.plot(df['data_registro'], df['media_movel_alcool_7d'], label='Media alcool 7d')
plt.xlabel('Data')
plt.ylabel('Alcool')
plt.title('Agregações temporais')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Visão geral das novas features
print('Formato final do DataFrame:', df.shape)

df.head()