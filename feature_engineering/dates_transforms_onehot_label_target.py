import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)

# Carregando dados
wine = load_wine()
x = pd.DataFrame(data=wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name='target')

df = pd.concat([x, y], axis=1)

# Cirando coluna categorica
mapping_class = {0: 'class_0', 1: 'class_1', 2: 'class_2'}
df['classe_vinho'] = df['target'].map(mapping_class)

# Criando faixas de teor alcoolico
bins = [
    df['alcohol'].min() - 0.01,
    df['alcohol'].quantile(1/3),
    df['alcohol'].quantile(2/3),
    df['alcohol'].max() + 0.01
]
labels = ['baixo', 'medio', 'alto']
df['faixa_teor_alcoolico'] = pd.cut(df['alcohol'], bins=bins, labels=labels)

df[['alcohol', 'target', 'classe_vinho', 'faixa_teor_alcoolico']].head()

# Visão geral das variaveis categoricas
print('Valores unicos em classe_vinho:', df['classe_vinho'].unique())
print('Distribuição de classe_vinho:', df['classe_vinho'].value_counts())
print('Valores unicos em faixa_teor_alcoolico:', df['faixa_teor_alcoolico'].unique())
print('Distribuição de faixa_teor_alcoolico:', df['faixa_teor_alcoolico'].value_counts())

# One-hot encoding
col_cat = ['classe_vinho', 'faixa_teor_alcoolico']
df_one = pd.get_dummies(df[col_cat], drop_first=True)

print('Colunas resultantes do one-hot encoding:', df_one.columns.tolist())
df_one.head()

# One-hot enconding com OneHotEncoder
try:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
except TypeError:
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

onehot_array = encoder.fit_transform(df[col_cat])
onehot_cols = encoder.get_feature_names_out(col_cat)

df_onehot = pd.DataFrame(onehot_array, columns=onehot_cols)
df_onehot.head()

# Label encoding
le_class = LabelEncoder()
le_range = LabelEncoder()

df['classe_vinho_le'] = le_class.fit_transform(df['classe_vinho'])
df['faixa_teor_alcoolico_le'] = le_range.fit_transform(df['faixa_teor_alcoolico'])

df[['classe_vinho', 'classe_vinho_le', 'faixa_teor_alcoolico', 'faixa_teor_alcoolico_le']].head()

# Target Encoding
df['target_binario'] = (df['target'] == 1).astype(int)

df[['target', 'target_binario', 'classe_vinho', 'faixa_teor_alcoolico']].head()

# Separando treino e teste
col_use = ['classe_vinho', 'faixa_teor_alcoolico', 'target_binario']
df_sub = df[col_use].dropna().copy()

train_df, test_df = train_test_split(df_sub, test_size=0.3, random_state=42, stratify=df_sub['target_binario'])

print('Tamanho treino:', train_df.shape)
print('Tamanho teste:', test_df.shape)

# Target Encoding com groupby
mean_by_class = train_df.groupby('classe_vinho')['target_binario'].mean()
print('Medias de target_binario por classe de vinho:', mean_by_class)

train_df['classe_vinho_te'] = train_df['classe_vinho'].map(mean_by_class)
test_df['classe_vinho_te'] = test_df['classe_vinho'].map(mean_by_class)

train_df[['classe_vinho', 'classe_vinho_te', 'target_binario']].head()

# Target Encoding em faixa_teor_alcoolico
mean_by_range = train_df.groupby('faixa_teor_alcoolico')['target_binario'].mean()
print('Medias de target_binario por faixa de teor alcoolico:', mean_by_range)

train_df['faixa_teor_alcoolico_te'] = train_df['faixa_teor_alcoolico'].map(mean_by_range)
test_df['faixa_teor_alcoolico_te'] = test_df['faixa_teor_alcoolico'].map(mean_by_range)

train_df[['faixa_teor_alcoolico', 'faixa_teor_alcoolico_te', 'target_binario']].head()
    