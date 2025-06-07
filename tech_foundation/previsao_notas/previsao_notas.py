import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

dados = pd.DataFrame({
    'nota1': [ 7.5, 8.0, 6.5, 9.0, 5.5, 8.5, 7.0, 6.0, 8.0, 9.5],
    'nota2': [ 8.0, 7.5, 9.0, 6.5, 8.5, 7.0, 9.5, 6.0, 8.0, 7.5],
    'frequencia': [ 80, 90, 85, 95, 70, 80, 90, 75, 85, 95],
    'aprovado': [ True, True, True, True, False, True, True, False, True, True]
})

print("Dados Brutos:")
print(dados)


x = dados[['nota1', 'nota2', 'frequencia']]
y = dados['aprovado']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

modelo = LogisticRegression()
modelo.fit(x_train, y_train)

new_data = [[10, 5, 80]]
previsao = modelo.predict_proba(new_data)

print("\nPrevisão de Aprovação:")
print(f"Probabilidade de aprovação: {previsao[0][1]*100:.2f}%")
print(f"Probabilidade de reprovação: {previsao[0][0]*100:.2f}%")


