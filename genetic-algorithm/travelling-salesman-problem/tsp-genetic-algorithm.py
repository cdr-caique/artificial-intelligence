import numpy as np
import random
import operator
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import time


#   PARA EXECUTAR, COPIE PARA O ARQUIVO MAIN.PY


#   Setar os parâmetros como desejado
SEED            =   30      # Semente fixada
POPULATION_SIZE =   100     # Tamanho da população = número de rotas randômicas geradas para percorrer a lista de cidades
ELITE_SIZE      =   20      # Número de melhores indivíduos da geração
CROSSOVER_RATE  =   0.72    # Taxa de cruzamento
MUTATION_RATE   =   0.63    # Taxa de mutação
GENERATIONS     =   600     # Número de gerações

# SEED            =   30      # Semente fixada
# POPULATION_SIZE =   100     # Tamanho da população = número de rotas randômicas geradas para percorrer a lista de cidades
# ELITE_SIZE      =   20      # Número de melhores indivíduos da geração
# CROSSOVER_RATE  =   0.09    # Taxa de cruzamento
# MUTATION_RATE   =   0.07    # Taxa de mutação
# GENERATIONS     =   600     # Número de gerações


#   Criação da classe que representará uma cidade
class City:

    # Inicialização dos atributos de classe
    def __init__(self, x, y, idx):  
        self.x = x
        self.y = y
        self.idx = idx # ID da cidade

    # Função para cálculo da distância
    def distance(self, city): 
        xDis = abs(self.x - city.x) # Valor absoluto da distância no eixo x
        yDis = abs(self.y - city.y) # Valor absoluto da distância no eixo y
        distance = xDis + yDis      # Cálculo da distância Manhattan
        return distance             # Retorna a distância Manhattan

    # Formato de retorno quando as informações de uma cidade forem printadas
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


#   Criação da função fitness
#   Avalia cada indivíduo de uma população
class Fitness:

    # Definição de atributos e valores iniciais
    def __init__(self, route): 
        self.route = route 
        self.distance = 0 
        self.fitness = 0.0

    # Retorna o valor da distância calculada para percorrer uma rota
    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0

            for i in range(0, len(self.route)):
                fromCity = self.route[i]    # Cidade de partida
                toCity = None               # Cidade de chegada, inicialmente nula

                # Escolher a próxima cidade da lista
                if i + 1 < len(self.route):     
                    toCity = self.route[i + 1]
                # Quando a cidade de partida for a de última posição da lista, 
                # então a próxima cidade deverá ser a de posição zero (retornar para ponto de partida)
                else:
                    toCity = self.route[0]

                pathDistance += fromCity.distance(toCity)   # Distância total da rota é a soma das distâncias entre cada duas cidades em sequência

            self.distance = pathDistance

        return self.distance

    # Avalia a rota, atribuindo uma nota de 0 a 1
    def routeFitness(self): 
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())  # Quanto maior a distância total da rota, menor a nota que recebe
        return self.fitness


#   Gerar uma rota aleatória a partir da lista de cidades
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route


#   Gerar a população de rotas inicial
def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))    # Cria um vetor contendo várias rotas, formando a população
    return population


#   Rankear os indivíduos (rotas) da população
#   Retorna uma lista ordenada descrecentemente com 
#   a avaliação de cada individíduo da população
def rankRoutes(population):
    fitnessResults = {} # Mapa e não lista?
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()   # Monta a lista das avaliações de acordo com a população dada
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True) # Ordena a lista das avaliações


#   Função de seleção
#   Dada a lista ordenada com os melhores indivíduos (rotas) da população,
#   selecionar os indivíduos elite
def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0]) 
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults # Retorna os índices dos melhores candidatos a se tornarem pais


#   Piscina de cruzamento
def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):   # Percorre a lista de índices dos indivíduos selecionados
        index = selectionResults[i]             # index recebe o índice do indivíduo selecionado durante a função selection
        matingpool.append(population[index])    # Cria o vetor matingpool agregando cada melhor indivíduo selecionado
    return matingpool


#   Função de cruzamento
#   Cruza dois pais (rotas) para gerar uma filha
def breed(parent1, parent2, crossoverRate):
    child = []
    childP1 = []
    childP2 = []

    startGene = int(random.random() * len(parent1))             # Sorteia qual será o gene inicial do cruzamento
    while(startGene==len(parent1)):                             # Gene inicial não pode ser maior que o último índice válido de parent1
        startGene = int(random.random() * len(parent1))         # Sorteia novamente até que startGene<len(parent1)

    walkingTroughGene = int(crossoverRate * len(parent1))       # É a quantiade de genes (cidades) que serão adquiridas do parent1

    for i in range(0, walkingTroughGene): 
        childP1.append(parent1[startGene])                      # Preenche a primeira parte da filha com os genes do pai 1
        startGene = startGene+1                                 # Incrementa o índice do gene (cidade) que será agregada à lista
        if startGene==len(parent1):                             # Caso tiver chegado no último gene da lista
            startGene=0                                         # A lista volta para agregar a partir do primeiro gene

    childP2 = [item for item in parent2 if item not in childP1] # Adquire os genes do pai 2 que não foram adquiridos do pai 1
    child = childP1 + childP2                                   # Concatena as listas para formar a fiha (rota)

    return child


#   Gerar o cruzamento da população, retornando a lista de filhas (rotas)
def breedPopulation(matingpool, eliteSize, crossoverRate):
    children = []
    length = len(matingpool) - eliteSize                # Tamanho da lista fora os indíviduos da elite
    pool = random.sample(matingpool, len(matingpool))   # Escolhe aleatoriamente indivíduos da Piscina de Cruzamento

    for i in range(0, eliteSize):
        children.append(matingpool[i])  # Agrega às filhas os melhores indivíduos da população

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1], crossoverRate)    # Gerando as filhas
        children.append(child)  # Agrega cada filha na lista de filhas
    return children


#   Gerar uma mutação aleatória
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if (random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


#   Gerar uma mutação em toda a população
def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


#   Retorna a próxima geração utilizando passo a passo as funções definidas anteriormente
def nextGeneration(currentGen, eliteSize, crossoverRate, mutationRate):
    popRanked = rankRoutes(currentGen)									# Gera a população ranqueada
    selectionResults = selection(popRanked, eliteSize)					# Gera os indíces ranqueados
    matingpool = matingPool(currentGen, selectionResults)				# Gera a Piscina de Cruzamento
    children = breedPopulation(matingpool, eliteSize, crossoverRate)	# Gera as filhas a partir da Piscina de Cruzamento
    nextGeneration = mutatePopulation(children, mutationRate)			# Cria a lista da próxima geração (lista de rotas)
    return nextGeneration


#   Algoritmo Genético
def geneticAlgorithm(population, popSize, eliteSize, crossoverRate, mutationRate, generations):
    start_time = time.time()	# Salvar o instante inicial da execução da função
    pop = initialPopulation(popSize, population)	# Gerar a população inicial

    print("Initial distance (Manhattan): " + str(1 / rankRoutes(pop)[0][1]))	# Printa a distância da primeira geração

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, crossoverRate, mutationRate)
        end_time = time.time()
        if(end_time > start_time + 2*60):  # Se já se passaram 2min desde o instante inicial
            global GENERATIONS
            GENERATIONS = i+1   # Atualizar o tamanho da geração com o número atingido durante a execução
            break   # Finalizar a execução devido ao estouro de tempo máximo
        
    print("Final distance (Manhattan): " + str(1 / rankRoutes(pop)[0][1]))  # Printa a distância final em Manhattan

    if(end_time > start_time + 2*60):
        print("\nTIMEOUT")
        print("Execution failed after ", end_time - start_time, "[s] and ", GENERATIONS, "generations\n")
    else:
        print("Execution finished after ", end_time - start_time, "[s] and ", i+1, "generations\n")

    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]   

    print("Best route found: \n")  # Printa a melhor rota encontrada
    for i in range(len(bestRoute)):
        print("[", bestRoute[i].idx, "]:\t", bestRoute[i])

    return bestRoute


def geneticAlgorithmPlot(population, popSize, eliteSize, crossoverRate, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, crossoverRate, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])

    plt.plot(progress)
    plt.title('CR = ' + str(crossoverRate) + ' X MR = ' + str(mutationRate), fontsize=16)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    # plt.show()  <-  not supported by repl.it
    plt.savefig('graph1.png')
    print("\nGraph saved on your folder")


#   "main" do programa ocorre daqui para baixo

random.seed(SEED)   #   Semente fixa

#   Lista de todas as cidades
cityList = [
    City(x=97,  y=89,   idx=0),
    City(x=26,  y=96,   idx=1),
    City(x=46,  y=69,   idx=2),
    City(x=5,   y=38,   idx=3),
    City(x=90,  y=38,   idx=4),
    City(x=8,   y=64,   idx=5),
    City(x=94,  y=16,   idx=6),
    City(x=71,  y=84,   idx=7),
    City(x=35,  y=12,   idx=8),
    City(x=93,  y=100,  idx=9),
    City(x=80,  y=36,   idx=10),
    City(x=43,  y=78,   idx=11),
    City(x=31,  y=87,   idx=12),
    City(x=86,  y=26,   idx=13),
    City(x=46,  y=32,   idx=14),
    City(x=4,   y=3,    idx=15),
    City(x=48,  y=17,   idx=16),
    City(x=29,  y=96,   idx=17),
    City(x=93,  y=65,   idx=18),
    City(x=20,  y=80,   idx=19),
    City(x=30,  y=85,   idx=20),
    City(x=81,  y=75,   idx=21),
    City(x=45,  y=86,   idx=22),
    City(x=13,  y=34,   idx=23),
    City(x=76,  y=43,   idx=24),
    City(x=6,   y=64,   idx=25),
    City(x=87,  y=33,   idx=26),
    City(x=30,  y=31,   idx=27),
    City(x=81,  y=42,   idx=28),
    City(x=53,  y=74,   idx=29),
    City(x=46,  y=96,   idx=30),
    City(x=20,  y=95,   idx=31),
    City(x=4,   y=75,   idx=32),
    City(x=61,  y=85,   idx=33),
    City(x=92,  y=12,   idx=34),
    City(x=50,  y=80,   idx=35),
    City(x=45,  y=8,    idx=36),
    City(x=29,  y=64,   idx=37),
    City(x=53,  y=42,   idx=38),
    City(x=25,  y=51,   idx=39),
    City(x=65,  y=62,   idx=40),
    City(x=27,  y=16,   idx=41),
    City(x=87,  y=40,   idx=42),
    City(x=84,  y=82,   idx=43),
    City(x=62,  y=62,   idx=44),
    City(x=41,  y=86,   idx=45),
    City(x=48,  y=18,   idx=46),
    City(x=7,   y=7,    idx=47),
    City(x=8,   y=86,   idx=48),
    City(x=81,  y=65,   idx=49)
]

print("\nCity coordinates: \n")   # Printa todas as cidades na ordem como foram fornecidas
for i in range(len(cityList)):
    print("[", cityList[i].idx, "]:\t", cityList[i])
print("\n")

#   Chamada do algoritmo genético com passagem dos parâmetros
geneticAlgorithm(
    population=cityList,
    popSize=POPULATION_SIZE,
    eliteSize=ELITE_SIZE,
    crossoverRate=CROSSOVER_RATE,
    mutationRate=MUTATION_RATE,
    generations=GENERATIONS)

print("\nWant to plot the graph? 'N' finish the program: ")
option = input()
if option == 'N':
    print("\nFinished\n")
    sys.exit()
else:
    print("\nWaiting")

#   Chamada da função para plotar o algoritmo genético
#   Caso a execução do algoritmo genético tenha estourado
#   o tempo máximo, então o gráfico será plotado corretamente
#   até a última geração que foi processada
geneticAlgorithmPlot(
    population=cityList,
    popSize=POPULATION_SIZE,
    eliteSize=ELITE_SIZE,
    crossoverRate=CROSSOVER_RATE,
    mutationRate=MUTATION_RATE,
    generations=GENERATIONS)

#   Sinalizar o fim do programa
print("\nEND OF PROGRAM\n")
