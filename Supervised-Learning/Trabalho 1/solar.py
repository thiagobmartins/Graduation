# -*- coding: utf-8 -*-
"""
Created on Tue May 25 10:20:58 2021

@author: Thiago Bruschi Martins
"""

import pandas as pd

# 1) Leia o arquivo solar-flare.csv
colnames = ['class','largest_spot', 'spot_distribution','activity', 'evolution', 'previous_activity',
            'historically_complex', 'new_historically_complex', 'area', 'largest_area',
            'C_flares','M_class','X_class']

df = pd.read_csv('solar-flare.csv', sep=' ', header=None, skiprows=1, names=colnames, 
                 dtype={'class':'category', 'largest_spot':'category', 'spot_distribution':'category'})

# Imprima usando o pandas.head() o inicio e o fim desse conjunto de dados.
print('Exercicio 1)')
print('Inicio dos dados iniciais:', df.head())
print('Fim dos dados iniciais:', df.tail())

# 2) Usando o one-hot-enconder, converta todos os atributos categóricos para numéricos.
one_hot = pd.get_dummies(df)

# Imprima usando o pandas.head() o inicio e o fim desse conjunto de dados transformados.
print('\nExercicio 2)')
print('Inicio dos dados com one_hot_encodding:', one_hot.head())
print('Fim dos dados com one_hot_encodding:', one_hot.tail())

# 3) Faca o centering and standard scaling para todos os atributos de entrada 
from sklearn.preprocessing import StandardScaler

targets = ['C_flares','M_class','X_class']
X = one_hot.drop(targets, axis=1)
outcomes = one_hot[targets]

scaler = StandardScaler()
centered = scaler.fit_transform(X)

# 4.1) Quantas dimensões restarão se mantivermos 90% da variância dos dados? Resposta: 13
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(centered)
variance_ratio = pca.explained_variance_ratio_
print(f'\nExercicio 4) Soma da proporção da variância alcançada com {pca.n_components_} componentes: {variance_ratio.sum():.2f}%')

# 4.2) Use o scree plot para determinar quantas dimensões devem ser mantidas
import matplotlib.pyplot as plt
import numpy as np

plt.plot(range(len(variance_ratio)), variance_ratio)
#plt.yticks(np.linspace(0,1,11))
plt.title('4.2) Scree plot')
plt.xlabel('Principal Component')
plt.ylabel('Proporção de Variância Alcançada (explicada)')
plt.show()
# Resposta: Analisando o joelho do gráfico, eu escolheria utilizar 4 PCAs

# 4.3) Converta os dados usando o PCA com 90% das variância
pca_90 = PCA(n_components=pca.n_components_)
reduced = pca_90.fit_transform(centered)

# 5) Validação cruzada e regressão linear
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

scores = {}
for col in outcomes:  # iterates over each column of outcomes
    y = outcomes[col]
    ss = ShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
    rmse_scores = -np.round(cross_val_score(LinearRegression(), reduced,  y, cv=ss, scoring='neg_mean_squared_error'),3)
    mae_scores = -np.round(cross_val_score(LinearRegression(), reduced,  y, cv=ss, scoring='neg_mean_absolute_error'),3)
    scores[col] = {'RMSE':rmse_scores, 'MAE':mae_scores}

def print_dict(d):
    for x in d:
        print (x)
        for y in d[x]:
            print ('\t',y,':',d[x][y])
            
print('\nExercicio 5) Resultado de 5 repetições cruzadas sobre cada uma das saídas')
print_dict(scores)

avg_scores = {}
for col in outcomes:
    avg_scores[col] = {'RMSE':round(np.mean(scores[col]['RMSE']),3),
                       'MAE':round(np.mean(scores[col]['MAE']),3)}
print('\nImprima também a média do RMSE e do MAE')
print_dict(avg_scores)