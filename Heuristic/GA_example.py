import numpy as np
import copy
import time
import matplotlib.pyplot as plt

np.random.seed(500)

#初始化種群: 大小為 population_size*chrom_length(=座標點位)
def IntilaGenes(population_size, chrom_length):
    population = np.zeros((population_size,chrom_length), dtype=np.int64)
    code = np.arange(chrom_length)
    for i in range(population_size):
        population[i] = copy.deepcopy(code)
        np.random.shuffle(population[i])
    return population  

class Genetic_Algorithm():
    def __init__(self, population, population_size, chrom_length, distance, 
                       coords, crossover_rate=0.5, mutation_rate=0.8):
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.pop = population
        self.population_size = population_size
        self.chrom_length = chrom_length
        self.coords = coords
        self.distance = distance # 點位間的距離 

    # 移動總距離
    def compute_fitness(self, pop:list):
        fitness = np.zeros(self.population_size, dtype=np.float32)
        for idx, each_route in enumerate(pop):
            for j in range(self.chrom_length-1):
                fitness[idx] += self.distance[int(each_route[j])][int(each_route[j+1])]
        
        dis = copy.copy(fitness)
        fitness = np.reciprocal(fitness)  #計算導數(不是必需的，但在某些情況下，它可以提高優化算法的效率和準確性)
        return fitness, dis

    # Roulette Wheel Selection
    def select_population(self, fitness):
        # Calculate the probability of selection p for each chromosome(fit[i]/total_fitness)
        indx = np.random.choice(np.arange(self.population_size),
                                size=self.population_size,
                                replace=True,             #可以重複的chromosome
                                p=fitness/fitness.sum())  #the best chromosomes get more copies
        self.pop = self.pop[indx] #更新種群體

    # 對新的種群體內，進行基因交叉(one-point/two-point/many-point)
    # 以下使用 Subtour Exchange Crossover
    # 方法參考: https://blog.csdn.net/ztf312/article/details/82793295
    def genetic_crossover(self):
        # 在某個父代上選擇1組基因，在另一父代上找到這些基因的位置
        for parent1 in self.pop:
            # when crossover rate is met, do crossover
            if np.random.rand() < self.crossover_rate: 
                n = np.random.randint(self.population_size)
                parent2 = self.pop[n,:]
                #print(f'第一組父代: {parent1}')
                #print(f'第二組父代: {parent2}\n')

                #隨機取要交換的基因片段，size=2，會取出兩個indx，因此基因片段是這兩個indx之間的座標
                chrom_slice = np.random.randint(self.chrom_length, size=2)
                l,r = min(chrom_slice), max(chrom_slice)
                #print(f'基因片段區間: {chrom_slice}')

                parent1_slice = copy.copy(parent1[l:r])
                #print(f'第一組父代基因片段: {parent1_slice}')

                parent2_exchange_chrom_indexs = []
                for chrom_idx in range(self.chrom_length):
                    if parent2[chrom_idx] in parent1_slice:
                        parent2_exchange_chrom_indexs.append(chrom_idx)
                #print(f'第二組父代基因片段 index: {parent2_exchange_chrom_indexs}')

                # 更新兩組父代
                parent2_copy = parent2.copy()
                for idx, chrom_value in enumerate(parent1_slice):
                    parent2[parent2_exchange_chrom_indexs[idx]] = chrom_value

                for idx, i in enumerate(range(l,r)):
                    parent1[i] = parent2_copy[parent2_exchange_chrom_indexs[idx]]
                
                #print(f'\nnew 第一組父代: {parent1}')
                #print(f'new 第二組父代: {parent2}')

    #總群中所有皆進行突變
    def genetic_mutation(self):
        pops_copy = self.pop.copy()
        for pop_idx, pop in enumerate(pops_copy):
            if np.random.rand() < self.mutation_rate:
                mutation_position = np.random.randint(self.chrom_length, size=2)
                pop[mutation_position[0]],pop[mutation_position[1]] = pop[mutation_position[1]],pop[mutation_position[0]]
                self.pop[pop_idx] = pop

def TSP(coords, population_size, chrom_length:int, distance, generations:int):
    st = time.time()
    population = IntilaGenes(population_size, chrom_length) #初始化種群
    GA = Genetic_Algorithm(population, population_size, chrom_length, distance, coords)
    best_distance = 1e10 #因為要取最小，所以先設定一個最大值
    best_move_order = []
    best_x = []
    best_y = []
    fitness_process = [] #適應度(導數)變化曲線

    for _iter_i in range(generations):
        iter_i = _iter_i + 1
        if time.time()-st > 100:
            break

        # !!!! Avoid getting the same chromosomes. If we get same, we cannot generate new chromosomes during crossover. !!!!
        fitness, dis = GA.compute_fitness(GA.pop)

        #紀錄當前最佳解
        num = np.argmax(fitness)
        move_order = GA.pop[num,:]
        fitness_process.append(max(fitness))
        
        if best_distance > min(dis):
            print(f"The step is {iter_i}, the current best distance is {min(dis)}")
            best_distance = min(dis)
            best_move_order = move_order
            #還原成座標
            best_x,best_y = [],[]
            for i in move_order:
                best_x.append(coords[0][i])
                best_y.append(coords[1][i])
        show(best_x, best_y, coords, best_distance, iter_i)

        GA.select_population(fitness) #選擇新的種群
        GA.genetic_crossover()        #基因交叉
        GA.genetic_mutation()         #基因突變

    print(f"The best route order is {best_move_order}")
    print(f"The route distance is {best_distance}")
    best_show(best_x, best_y, coords, fitness_process, best_distance)

def show(lx, ly, coords, dis, iter_i):
    plt.cla()
    plt.scatter(coords[0], coords[1], color='r')
    plt.plot(lx,ly)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"iter {iter_i} - Total distance={dis}")
    plt.pause(0.01)

def best_show(x,y,coords,best_fitness,dis):
    fig,ax=plt.subplots(1,2, figsize=(12,5), facecolor='#ccddef')
    ax[0].set_title(f"Best route (from {int(init_distance)} to {int(dis)})")
    ax[1].set_title("Fitness Change Procession")
    ax[0].plot(x,y)
    ax[0].scatter(coords[0],coords[1],color = 'r')
    ax[1].plot(range(len(best_fitness)),best_fitness)
    plt.show()

if __name__ == "__main__":
    start_time = time.time()
    x = [499,267,703,408,437,491,74,532,416,626,42,271,359,163,508,229,576,147,560,35,714,757,517,64,314,675,690,391,628,87,240,705,699,258,428,]
    y = [556,57,401,305,421,267,105,525,381,244,330,395,169,141,380,153,442,528,329,232,48,498,265,343,120,165,50,433,63,491,275,348,222,288,490]
    coords = np.array([x,y])

    #計算距離矩陣,distance[i][j]表示 i to j 的距離
    distance = np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            distance[i][j] = np.sqrt(np.square(x[i]-x[j])+np.square(y[i]-y[j]))

    init_distance = 0
    for j in range(len(x)-1):
        init_distance += distance[j][j+1]
    print(init_distance)
    
    population_size = 400 # 種群數量
    chrom_length = len(x) # 幾個座標點(染色體長度)
    print(f'共幾個點位 {chrom_length}')
    generations = 100 # 迭代次數
    TSP(coords, population_size, chrom_length, distance, generations)
    print(f'Exec time: {time.time()-start_time}')