import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler

plt.style.use('default')
pd.set_option('display.max_columns', None)

# Carregando conjunto de dados
wine = load_wine()

x = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name='target')

df = pd.concat([x, y], axis=1)
df.head()

# Analise inicial das escalas das features
statics = df.describe().T[['mean', 'std', 'min', 'max']]
print(statics)

# Selecionando features para exemplo
features = ['alcohol', 'malic_acid', 'color_intensity']
df_example=df[features].copy()
df_example.head()

# Visualização inicial das distribuições
df_example.hist(bins=15, figsize=(10,4))
plt.suptitle('Distribuição das features originais', y=1.02)
plt.tight_layout()

# Aplicando normalização (min-max)
scaler = MinMaxScaler(feature_range=(0,1))

df_minmax_array = scaler.fit_transform(df_example)
df_minmax = pd.DataFrame(df_minmax_array, columns=[col + '_minmax' for col in features])

df_minmax.describe().T

# Visualizando após normalização
df_minmax.hist(bins=15, figsize=(10,4))
plt.suptitle('Distribuição das features normalizadas', y=1.02)
plt.tight_layout()

# Aplicando padronização
scaler = StandardScaler()

df_standard_array = scaler.fit_transform(df_example)
df_standard = pd.DataFrame(df_standard_array, columns=[col + '_standard' for col in features])

df_standard.describe().T

# Visualizando após padronização
df_standard.hist(bins=15, figsize=(10,4))
plt.suptitle('Distribuição das features padronizadas', y=1.02)
plt.tight_layout()

# Comparando estatisticas antes x depois das transformações
df_compare = pd.concat([
    df_example.add_suffix('_original'),
    df_minmax,
    df_standard
], axis=1)

df_compare.describe().T