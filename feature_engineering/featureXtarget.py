import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

plt.style.use('default')
pd.set_option('display.max_columns', None)

# Carregando um conjunto dos dados
wine = load_wine()

x = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name='target')

df = pd.concat([x, y], axis=1)
df.head()

# Visão geral do conjunto de dados
print('Formato do DataFrame:', df.shape)

print('Tipo de dados (dtypes):', df.dtypes)

print('Valores faltantes:', df.isna().sum())

# Estatistica descritivas básicas
df.describe().T

# Analise exploratoria inicial
cols_example = ['alcohol', 'malic_acid', 'ash', 'color_intensity']
df[cols_example].hist(bins=15, figsize=(10,8))
plt.suptitle('Distribuição de algumas features numericas', y=1.02)
plt.tight_layout()
plt.show()
sns.pairplot(df[['alcohol', 'malic_acid', 'color_intensity', 'hue', 'target']], hue='target', diag_kind='hist')
plt.show()

# Separando features e alvo
x = df.drop(columns=['target'])
y = df['target']

print('Formato de X (features):',x.shape)
print('Formato de Y (target):',y.shape)