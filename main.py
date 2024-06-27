import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

dataset = 'datasets/Cartao de credito.csv'
data = pd.read_csv(dataset, delimiter=';', decimal=',', encoding='latin-1')
limiar = 0.29

print(data.columns)

vars_independentes = data.drop(columns=['Identificador da transação', 'Bandeira do Cartão', 'Fraude'])
rotulos = data['Fraude']

# Dividir os dados em conjuntos de treino e teste
vars_indep_treino, vars_indep_teste, rotulos_treino, rotulos_teste = train_test_split(vars_independentes, rotulos,
                                                                                      test_size = 0.25, random_state = random.randint(0,4294967295))

# Normalizar os dados
normalizador = StandardScaler()
vars_indep_treino_normal = normalizador.fit_transform(vars_indep_treino)
vars_indep_teste_normal = normalizador.transform(vars_indep_teste)

# Treinar o modelo de regressão logística
modelo = LogisticRegression()
modelo.fit(vars_indep_treino_normal, rotulos_treino)

# Fazer previsões
previsao_probabilidade = modelo.predict_proba(vars_indep_teste_normal)
previsao_teste = np.where(previsao_probabilidade[:, 1] > limiar, 'SIM', 'NÃO')

# Avaliar o modelo
matriz_conf = confusion_matrix(rotulos_teste, previsao_teste)
print("Precisão:", accuracy_score(rotulos_teste, previsao_teste))
print("Matriz de Confusão:\n", matriz_conf)
print("Relatório:\n", classification_report(rotulos_teste, previsao_teste))

# Visualizar a matriz de confusão
sns.heatmap(matriz_conf, annot=True, fmt='d')
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.show()

# Carregar dataset de validação
dataset_validacao = 'datasets/Dataset Validacao.csv'
dados_validacao = pd.read_csv(dataset_validacao, delimiter=';', decimal=',', encoding='latin-1')
vars_indep_validacao = dados_validacao.drop(columns=['Identificador da transação', 'Bandeira do Cartão', 'Fraude'])
vars_indep_validacao_normal = normalizador.transform(vars_indep_validacao)

# Fazer previsões
previsao_probabilidade = modelo.predict_proba(vars_indep_validacao_normal)
previsao_validacao = np.where(previsao_probabilidade[:, 1] > limiar, 'SIM', 'NÃO')

# Salvar os dados com as previsões em um CSV
dados_validacao['Fraude'] = previsao_validacao
dados_validacao.to_csv('datasets/validacao previsao.txt', index=False)