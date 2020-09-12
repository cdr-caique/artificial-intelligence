# -*- coding: utf-8 -*-

"""
trabalho_04.py

Aluno: Caíque Cléber Dias de Rezende
Matrícula: 2016003750

ECO904 Inteligência Artificial - Professor Carlos Henrique Valério de Moares
Universidade Federal de Itajubá

**Trabalho sobre Mineração de Dados**
"""

import time
import warnings
import random

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn import cluster, mixture, metrics
from sklearn.neighbors import kneighbors_graph

import matplotlib.pyplot as plt
from itertools import cycle, islice

from prettytable import PrettyTable

# carrega dataset em aluguel
aluguel = pd.read_csv('aluguel_brasil.csv')

# imprime as primeiras linhas de aluguel
print("\nDATASET CARREGADO:\n")
print(aluguel.head())

# fixar semente
matricula = 2016003750
random.seed(matricula)

# escolha da cidade
cidades = ['Belo Horizonte', 'Campinas', 'Porto Alegre', 'Rio de Janeiro', 'São Paulo']
selCidade = random.choice(cidades)
print("\n\t---> CIDADE ESCOLHIDA PARA ALUGUEL:", selCidade.upper())
ehCidade = aluguel['city'] == selCidade
aluguel = aluguel[ehCidade]
aluguel = aluguel.drop('city', axis=1)
print("\nTABELA PARA A CIDADE:\n")
print(aluguel.head())

# remover 3 colunas aleatórias de aluguel
colunas = aluguel.columns.values.copy()
random.shuffle(colunas)
colunas = colunas[:3]
print("\n\t---> COLUNAS SORTEADAS:", colunas)
aluguel = aluguel.drop(colunas, axis=1)
print("\nTABELA PARA A CIDADE SEM AS COLUNAS SORTEADAS:\n")
print(aluguel.head())

# contar número de imóveis em aluguel
print("\n\t---> NÚMERO DE IMÓVEIS = ", len(aluguel))

# descrever as colunas, funciona apenas para valores numéricos
print("\nDESCREVER TABELA:\n")
print(aluguel.describe())

# verificar se as colunas 'floor', 'animal' ou 'furniture' foram sorteadas para exclusão
removed_floor = False
removed_animal = False
removed_furniture = False
for i in range(len(colunas)):
    if(colunas[i] == 'floor'):
        removed_floor = True
    if(colunas[i] == 'animal'):
        removed_animal = True
    if(colunas[i] == 'furniture'):
        removed_furniture = True

# transformar as colunas de texto ('floor', 'animal', 'furniture') em colunas numéricas, caso NÃO tenham sido removidas de aluguel
labelEncoder = LabelEncoder()
if(removed_floor == False):
    labelEncoder.fit(aluguel['floor'])
    aluguel['floor'] = labelEncoder.transform(aluguel['floor'])
if(removed_animal == False):
    labelEncoder.fit(aluguel['animal'])
    aluguel['animal'] = labelEncoder.transform(aluguel['animal'])
if(removed_furniture == False):
    labelEncoder.fit(aluguel['furniture'])
    aluguel['furniture'] = labelEncoder.transform(aluguel['furniture'])
print("\nTABELA PARA A CIDADE SEM AS COLUNAS SORTEADAS E COM VALORES APENAS NUMÉRICOS:\n")
print(aluguel.head())

# normalizando dataset
X = StandardScaler().fit_transform(aluguel)
print("\nDATASET NORMALIZADO:\n")
print(X)

# técnicas de agrupamento, NECESSÁRIO ESTUDAR E OTIMIZAR OS PARÂMETROS DE CADA TÉCNICA
two_means = cluster.MiniBatchKMeans(n_clusters=2, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)

bandwidth = cluster.estimate_bandwidth(X, quantile=0.95)
ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)

spectral = cluster.SpectralClustering(n_clusters=8, eigen_solver='arpack', affinity="nearest_neighbors", random_state=0)

connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
connectivity = 0.5 * (connectivity + connectivity.T)
ward = cluster.AgglomerativeClustering(n_clusters=3, linkage='ward', connectivity=connectivity)
average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock", n_clusters=3, connectivity=connectivity)

dbscan = cluster.DBSCAN(eps=5)
birch = cluster.Birch(n_clusters=3, threshold=0.7)
gmm = mixture.GaussianMixture(n_components=2, max_iter=300, random_state=0, n_init=10)

clustering_algorithms = (
    ('KMeans', two_means),
    ('MeanShift', ms),
    ('Espectral', spectral),
    ('Ward', ward),
    ('Hierarquico', average_linkage),
    ('DBSCAN', dbscan),
    ('Birch', birch),
    ('GaussianMixture', gmm)
)

# impressão de resultados
print("\nTABELA DE RESULTADOS PARA AS TÉCNICAS DE AGRUPAMENTO:\n")
table = PrettyTable()
table.field_names = ["Algoritmo", "Tempo(s)", "Grupos", "Ruidos", "Coef.Silhueta"]

for name, algorithm in clustering_algorithms:
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)  # oculta erros
        algorithm.fit(X)
    t1 = time.time()

    if hasattr(algorithm, 'labels_'):
        grupos = algorithm.labels_.astype(np.int)
    else:
        grupos = algorithm.predict(X)

    numGrupos = len(set(grupos)) - (1 if -1 in grupos else 0)  # ignora ruídos
    numRuido = list(grupos).count(-1)
    silhueta = metrics.silhouette_score(X, grupos)

    table.add_row([name, '%.2f' % (t1 - t0), numGrupos, numRuido, '%.2f' % silhueta])

print(table)

"""Analisando resultados"""

print("\nRESULTADOS PARA ANÁLISE:\n")

for name, algorithm in clustering_algorithms:
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(X)
    print(name, {x: list(y_pred).count(x) for x in y_pred})

print("\n")

plt.figure(figsize=(20, 10))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)

# plotando resultados
plot_num = 1
for i in range(0, len(aluguel.columns)-1, 2):
    for name, algorithm in clustering_algorithms:
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        # local do grafico
        plt.subplot(4, 8, plot_num)
        if i == 0:
            plt.title(name, size=18)
        # valores do gráfico
        plt.scatter(X[:, i], X[:, i+1], s=10, color=colors[y_pred])
        plt.xticks(())
        plt.yticks(())
        plot_num += 1

plt.savefig('resultados.png')
