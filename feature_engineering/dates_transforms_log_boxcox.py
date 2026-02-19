import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer

plt.style.use('default')
pd.set_option('display.max_columns', None)

# Carregar dados
wine = load_wine()

x = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name='target')

df = pd.concat([x, y], axis=1)
df.head()

# Selecionando features e avaliando assimetria
feature_example = ['malic_acid', 'color_intensity', 'proline']
df_example = df[feature_example].copy()

print('Skewness antes das transformações')
df_example.skew()

# Visualização inicial das distribuições
df_example.hist(bins=15, figsize=(10,4))
plt.suptitle('Distribuições antes das transformações', y=1.02)
plt.tight_layout()

# Aplicando transformação logaritmica
df_log = df_example.apply(np.log)

print('Skewness após a transformação')
df_log.skew()

# Visualizando após trasnformação logaritmica
df_log.hist(bins=15, figsize=(10,4))
plt.suptitle('Distribuições após a transformação logaritmica', y=1.02)
plt.tight_layout()

# Transformação box-cox
proline = df_example['proline']

proline_boxcox, lambda_proline = boxcox(proline)

print('Lambda estiamdo para proline:', lambda_proline)

df_boxcox_proline = pd.Series(proline_boxcox, name='proline_boxcox')

print('Skewness antes proline:', proline.skew())
print('Skewness após proline:', df_boxcox_proline.skew())

# Visualizando antes x depois box-cox
fig, ax = plt.subplots(1,2, figsize=(10,4))

ax[0].hist(proline, bins=15)
ax[0].set_title('Antes')

ax[1].hist(df_boxcox_proline, bins=15)
ax[1].set_title('Depois')

plt.tight_layout()

# Aplicando box-cox em multiplas features
power = PowerTransformer(method='box-cox')

df_boxcox_array = power.fit_transform(df_example)
df_boxcox = pd.DataFrame(df_boxcox_array, columns=[col + '_boxcox' for col in feature_example])

print('Skewness após box-cox em multiplas features')
df_boxcox.skew()

# Visualizando após box-cox com multiplas features
df_boxcox.hist(bins=15, figsize=(10,4))
plt.suptitle('Distribuição das features após box-cox', y=1.02)
plt.tight_layout()

# Comparando skewness antes x depois das transformações

skew = df_example.skew()
skew_log = df_log.skew()
skew_boxcox = df_boxcox.skew()

df_skew = pd.DataFrame({
    'antes': skew,
    'log': skew_log,
    'boxcox': skew_boxcox
})
df_skew