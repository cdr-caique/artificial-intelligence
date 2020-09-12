# bibliotecas
from sklearn import datasets, model_selection, tree, metrics, svm, neural_network

print('_____________________________________________________________________')
print('ECO904 - I.A. - Trabalho 2 - Classificação com Aprendizado de Maquina')

# dataset utilizado
banco = datasets.load_wine() #wine dataset
# banco = datasets.load_iris() #iris dataset
X = banco.data
Y = banco.target

# salvando analises em arquivo
arquivo = open('Wine.txt', 'w') 
# arquivo = open('Iris.txt', 'w') 

# conjunto para treinamento e teste
X_treina, X_teste, Y_treina, Y_teste = model_selection.train_test_split(X, Y, random_state=0)

# treinando
print('> Treinamento por árvore de decisão')
arvore = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_features='sqrt', max_depth=9, random_state=514).fit(X_treina, Y_treina)
# valores previstos para teste
Y_previsto = arvore.predict(X_teste) 
# metricas de acerto
mc = metrics.plot_confusion_matrix(arvore, X_teste, Y_teste)
acuracia = metrics.accuracy_score(Y_teste,Y_previsto)

print('---------------------------', file = arquivo)
print('  DecisionTreeClassifier', file = arquivo)
print('---------------------------', file = arquivo)
print('===Matriz de Confusao===', file = arquivo)
print(mc.confusion_matrix, file = arquivo)
print('Acuracia = ',acuracia, file = arquivo)

# treinando
print('> Treinamento por SVM')
modelo = svm.SVC(kernel='linear', shrinking=False, probability=True, gamma='scale', random_state=514).fit(X_treina, Y_treina)
# valores previstos para teste
Y_previsto = modelo.predict(X_teste) 
# metricas de acerto
mc = metrics.plot_confusion_matrix(modelo, X_teste, Y_teste)
acuracia = metrics.accuracy_score(Y_teste,Y_previsto)

print('---------------------------', file = arquivo)
print('           SVC', file = arquivo)
print('---------------------------', file = arquivo)
print('===Matriz de Confusao===', file = arquivo)
print(mc.confusion_matrix, file = arquivo)
print('Acuracia = ',acuracia, file = arquivo)

# treinando
print('> Treinamento por Redes Neurais Multi-camadas')
rna = neural_network.MLPClassifier(activation='identity', solver='adam', learning_rate='constant').fit(X_treina, Y_treina)
# valores previstos para teste
Y_previsto = rna.predict(X_teste) 
# metricas de acerto
mc = metrics.plot_confusion_matrix(rna, X_teste, Y_teste)
acuracia = metrics.accuracy_score(Y_teste,Y_previsto)

print('---------------------------', file = arquivo)
print('  DecisionTreeClassifier', file = arquivo)
print('---------------------------', file = arquivo)
print('===Matriz de Confusao===', file = arquivo)
print(mc.confusion_matrix, file = arquivo)
print('Acuracia = ',acuracia, file = arquivo)

# fechando arquivo
arquivo.close()

print('Fim do Programa - Arquivo salvo!')
