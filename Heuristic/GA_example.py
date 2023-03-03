import numpy as np
import copy
from time import time
import matplotlib.pyplot as plt

np.random.seed(500)

class Genetic_Algorithm():
    def __init__(self, population, population_size, chrom_length, distance, coords, crossover_rate=0.1555, mutation_rate=0.025):
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.pop = population
        self.population_size = population_size
        self.chrom_length = chrom_length
        self.coords = coords
        self.distance = distance # 點位間的距離 

    # 計算各個群體的 fitness
    def compute_fitness(self, pop:list):
        fitness = np.zeros(self.population_size, dtype= np.float32)
        for i,e in enumerate(pop):
            for j in range(self.chrom_length-1):
                fitness[i] += self.distance[int(e[j])][int(e[j+1])]
        
        dis = copy.copy(fitness)
        fitness = np.reciprocal(fitness)  #TODO: 适应度等于距离的倒数(?)
        return fitness, dis

    #轮盘赌，选择种群中的个体
    def select_population(self,fitness):
        #从种群中选择，population_size个个体，每个个体被选择的概率为fitness / fitness.sum()
        indx = np.random.choice(np.arange(self.population_size),size = self.population_size,replace = True,p = fitness / fitness.sum())
        self.pop = self.pop[indx] #花式索引，更新种群

    #对新种群中的所有个体进行基因交叉
    def genetic_crossover(self):
        #遍历种群每个个体
        for parent1 in self.pop:
            #判断是否会基因交叉
            if np.random.rand() <self.crossover_rate:
                #### Subtour Exchange Crossover
                #### 基因交换方法参考6
                #####基因交叉参考https://blog.csdn.net/ztf312/article/details/82793295
                #寻找父代2
                n = np.random.randint(self.population_size)
                parent2 = self.pop[n,:]
                #随机产生基因交换片段
                pos = np.random.randint(self.chrom_length,size = 2)
                #区间左右端点
                l = min(pos)
                r = max(pos)
                #记录区间
                seq = copy.copy(parent1[l:r])
                poss = []
                #交换
                for i in range(self.chrom_length):
                    if parent2[i] in seq:
                        poss.append(i)
                a = 0
                for i in seq:
                    parent2[poss[a]] = i
                    a+=1
                b = 0
                for i in range(l,r):
                    parent1[i] = parent2[poss[b]]
                    b+=1

    #种群中的所有个体基因突变
    def genetic_mutation(self):
        #枚举个体
        for e in self.pop:
            #变异的可能
            if np.random.rand() < self.mutation_rate:
                #随机变异交换点
                position = np.random.randint(self.chrom_length,size = 2)
                e[position[0]],e[position[1]] = e[position[1]] , e[position[0]]

#初始化種群: 大小為 population_size*chrom_length(=座標點位)
def IntilaGenes(population_size,chrom_length):
    population = np.zeros((population_size,chrom_length))
    code = np.arange(chrom_length)
    for i in range(population_size):
        population[i] = copy.deepcopy(code)
        np.random.shuffle(population[i])
    return population       

def TSP(coords, population_size, chrom_length:int, distance, generations:int):
    population = IntilaGenes(population_size,chrom_length) #初始化種群
    GA = Genetic_Algorithm(population, population_size, chrom_length, distance, coords)
    best_distance = 1e10
    route = None
    x = None #保存最佳x坐标
    y = None #保存最佳y坐标
    fitness_process = [] #保存适应度变化曲线

    for i in range(generations):
        #返回适应度，和距离函数
        fitness, dis= GA.compute_fitness(GA.pop)
        
        GA.select_population(fitness) #選擇新的種群
        GA.genetic_crossover() #基因交叉
        GA.genetic_mutation() #基因突變

        #记录当前状态最优解，返回最优解索引
        num = np.argmax(fitness)
        #记录DNA
        DNA = GA.pop[num,:]
        print(f"The step is {i} ,the current best distance is {min(dis)} ,fitness is {max(fitness)}")
        lx,ly = [],[]
        #DNA转化为记录坐标
        fitness_process.append(max(fitness))
        for i in DNA:
            i = int(i) 
            lx.append(coords[0][i])
            ly.append(coords[1][i])
        #保存最佳方案
        if best_distance > min(dis):
            best_distance = min(dis)
            route = DNA = GA.pop[num,:]
            x = copy.copy(lx)
            y = copy.copy(ly)
        show(lx,ly,coords,min(dis))

    print(f"The best route is {route}")
    print(f"The route distance is {best_distance}")
    best_show(x,y,coords,fitness_process,best_distance)

def show(lx, ly,coords,dis):
    plt.cla()
    plt.scatter(coords[0],coords[1],color = 'r')
    plt.plot(lx,ly)
    plt.xticks([])
    plt.yticks([])
    plt.text(-5.25, -14.05, "Total distance=%.2f" % dis, fontdict={'size': 20, 'color': 'red'})
    plt.pause(0.01)

def best_show(x,y,coords,best_fitness,dis):
    fig,ax=plt.subplots(1,2,figsize=(12,5),facecolor='#ccddef')
    ax[0].set_title("Best route")
    ax[1].set_title("Fitness Change Procession")
    ax[0].plot(x,y)
    ax[0].scatter(coords[0],coords[1],color = 'r')
    ax[1].plot(range(len(best_fitness)),best_fitness)
    plt.show()

if __name__ == "__main__":
    x = [499,267,703,408,437,491,74,532,416,626,42,271,359,163,508,229,576,147,560,35,714,757,517,64,314,675,690,391,628,87,240,705,699,258,428,]
    y = [556,57,401,305,421,267,105,525,381,244,330,395,169,141,380,153,442,528,329,232,48,498,265,343,120,165,50,433,63,491,275,348,222,288,490]
    coords = np.array([x,y])

    #計算距離矩陣,distance[i][j]表示 i to j 的距離
    distance = np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            distance[i][j] = np.sqrt(np.square(x[i]-x[j])+np.square(y[i]-y[j]))
    
    population_size = 400 # 種群數量
    chrom_length = len(x) # 幾個座標點(染色體長度)
    print(f'共幾個點位 {chrom_length}')
    generations = 50 # 迭代次數
    TSP(coords, population_size, chrom_length, distance, generations)