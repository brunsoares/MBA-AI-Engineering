import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

plt.style.use('default')
pd.set_option('display.max_columns', None)

# Carregando um conjunto de dados numericos
wine = load_wine()
x = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name='target')

df_base = pd.concat([x, y], axis=1)
df_base.head()

# Criando diferentes tipos de dados
df = df_base.copy()

mapping_class = {0: 'Classe_A', 1: 'Classe_B', 2: 'Classe_C'}
# Coluna categorica
df['classe_vinho'] = df['target'].map(mapping_class)

# Coluna textual
df['comentario'] = "Amostra de vinho para estudo de tipos de dados"

# Coluna data
df['data_registro'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')

# Coluna booleana
df['alto_teor_alcool'] = df['alcohol'] > df['alcohol'].median()

df.head()

# Inspecionar a estrutura do data frame
print("Formado do dataframe:", df.shape)
print("Colunas do dataframe:", df.columns.tolist())
print("Tipos de dados do dataframe:", df.dtypes)
print("Resumo de informações:", df.info())

# Separando colunas por tipo de dado
col_number = df.select_dtypes(include=['int64', 'float64']).columns
col_object = df.select_dtypes(include=['object']).columns
col_datetime = df.select_dtypes(include=['datetime64[ns]']).columns
col_boolean = df.select_dtypes(include=['bool']).columns

print("Colunas numericas:", col_number.tolist())
print("Colunas object:", col_object.tolist())
print("Colunas datetime:", col_datetime.tolist())
print("Colunas boolean:", col_boolean.tolist())

# Visualizando alguns exemplos por tipo de dado
print("Exemplos de dados numericos:")
print(df[col_number].describe().T)
print("\nExemplos de dados object:")
print(df[col_object].value_counts())
print("\nExemplos de dados datetime:")
print(df[col_datetime].head())
print("\nExemplos de dados boolean:")
print(df[col_boolean].value_counts())

# Visualizações gráficas por tipo de dado
# Numerico
cols_example_num = ['alcohol', 'malic_acid', 'color_intensity']
df[cols_example_num].hist(bins=15, figsize=(10, 4))
plt.tight_layout()
plt.show()

# Categorico
sns.countplot(x='classe_vinho', data=df)
plt.title('Distribuição de Classes de Vinhos')
plt.xlabel('Classe')
plt.ylabel('Contagem')
plt.show()

# Conversão explicita de tipos
df.dtypes
# Converter categorico
df['classe_vinho'] = df['classe_vinho'].astype('category')

# Converter string
df['comentario'] = df['comentario'].astype('string')

# Converter data
df['data_registro'] = pd.to_datetime(df['data_registro'])

df.dtypes