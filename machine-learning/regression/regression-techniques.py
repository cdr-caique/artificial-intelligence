import random
import matplotlib.pyplot as plt
import time

from prettytable import PrettyTable

from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
from sklearn.metrics import explained_variance_score,max_error,mean_absolute_error,mean_squared_error,median_absolute_error,r2_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import SGDRegressor

def media(lst):
    return sum(lst) / len(lst)

# número de matrícula do aluno
matricula=2016003750
# criando um dataset regressivo para análise das técnicas
X, y = make_regression(n_samples=1000, n_features=2, noise=10.0,  random_state=matricula)

# gerando o gráfico do dataset
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title('Dataset Gerado Original')
for r in range(X.shape[0]):
    ax.scatter(X[r,0],X[r,1],y[r],c='r',marker='x')
ax.grid(True)

# usar kFold para avaliar a técnica com dataset
kf = KFold(n_splits=10, random_state=matricula,shuffle=True) # usando famoso 10-fold

# Tecnicas de aprendizagem utilizadas
nomes = ['Arvore','RNA','SVM-R','Bayesiano','Linear','Vizinhanca','Gaussiano','Gradiente']
tecnicas = [
    DecisionTreeRegressor(max_depth=5),
    MLPRegressor(solver='sgd',max_iter=1200,activation='relu'),
    SVR(kernel='poly',degree=1),
    BayesianRidge(),
    LinearRegression(),
    KNeighborsRegressor(n_neighbors=15),
    GaussianProcessRegressor(normalize_y=True,n_restarts_optimizer=0,alpha=0.2),
    SGDRegressor()
    ]

# Gerar uma tabela bonita com os dados
table = PrettyTable()
table.field_names = ['Tecnica', 'EVS', 'ME', 'MAE', 'MSE', 'MDA', 'R2', 'Tempo']

# quebrando dataset em treinamento e testes por indices de vetores
i=0
for ml in tecnicas:

    EVS=[]
    MEr=[]
    MAE=[]
    MSE=[]
    MDA=[]
    R2=[]
    tempo=[]

    for treinaID, testeID in kf.split(X):
        # treinando técnica
        X_treina, Y_treina = X[treinaID], y[treinaID]
        inicio = time.time()
        modelo = ml.fit(X_treina,Y_treina) # treinamento
        tempo.append(time.time() - inicio)
        # Conjunto de teste
        X_teste, Y_teste = X[testeID], y[testeID]
        Y_previsto = modelo.predict(X_teste) # previsão pelo modelo treinado
        # metricas sobre previsão
        EVS.append(explained_variance_score(Y_teste,Y_previsto))
        MEr.append(max_error(Y_teste,Y_previsto))
        MAE.append(mean_absolute_error(Y_teste,Y_previsto))
        MSE.append(mean_squared_error(Y_teste,Y_previsto))
        MDA.append(median_absolute_error(Y_teste,Y_previsto))
        R2.append(r2_score(Y_teste,Y_previsto))

    table.add_row([nomes[i],media(EVS),media(MEr),media(MAE),media(MSE),media(MDA),media(R2),media(tempo)])
    i+=1

print(table)

plt.show()
