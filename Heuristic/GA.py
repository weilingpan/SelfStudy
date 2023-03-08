from math import sin, cos
import math
import random
import numpy as np

def IntilaGenes_1(population_size, chrom_length): 
    for i in range(population_size):
        temp = []
        for j in range(chrom_length): 
            temp.append(random.randint(0, 1))
        population.append(temp) 
    return population

# pre-censoring
def IntialGenes_2(population_size, chrom_length):
    i = 0
    while True:
        if i == population_size:
            break
        else:
            temp = []
            for j in range(chrom_length): 
                temp.append(random.randint(0, 1))

            x = Xbase10(temp[:6])
            y = Ybase10(temp[-5:])
            
            if x+y > 5: # pre-censoring
                pass
            else:
                population.append(temp)
                i = i + 1
    return population

# constraint: 0 <= coc2 row <= 2699
def Xbase10(xbits):
    # from base 2 to 10
    Sx = 0
    for i in range(len(xbits)):
        Sx = Sx + xbits[i]*2**(len(xbits)-i-1)
    # print(Sx)

    # find real number
    x = 0+Sx*((2699-0)/(2**6-1))
    # print(x)
    return x

# constraint: 0 <= coc2 col <= 2219
def Ybase10(ybits):
    Sy = 0
    for i in range(len(ybits)):
        Sy = Sy + ybits[i]*2**(len(ybits)-i-1)
    # print(Sy)

    # find real number
    y = 0+Sy*(3/((2219-0)**5-1))
    # print(y)
    return y

# fitness
# def fitness(x,y):
#     return sin(5*math.pi*(x**(3/4)-0.1))**2-(y-1)**4
def fitness(points):
    n = points.shape[0]
    dist = 0
    for i in range(n):
        for j in range(i+1, n):
            dist += np.sqrt(np.sum((points[i]-points[j])**2))
    return dist


population_size = 100  # 種群數量
chrom_length = 11  # 染色體長度 
population = []
generations = 100
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.1

population = IntilaGenes(population_size,chrom_length)

log = []
max_log = []
max_log.append(-10000000)
optimal_value = (0,0)

points = np.array([[1, 1], [2, 3], [4, 2], [3, 5]])
for epoch in range(generations):
    print('================================= generation '+str(epoch+1)+' =================================')
    # ---------------------------- selection(roulette wheel) ----------------------------
    total_fitness = 0
    fit = []

    for chrom in range(len(population)):
        x = Xbase10(population[chrom][:6]) # calculate x
        y = Ybase10(population[chrom][-5:]) # calculate y
        print(x,y)
        f = fitness(x,y) # (1)  calculate the fitness value
        fit.append(f)

    fit = [i+ abs(max(fit)) for i in fit] # 標準化
    total_fitness = sum(fit) # (2)  find the total fitness
    
    # (3) calculate the probability of selection p for each chromosome
    portion = []
    for j in range(len(fit)):
        portion.append(fit[j]/total_fitness) 
   
    # (4) calculate a cumulative probability q for each chromosome
    cumulative_p = []
    for p in range(len(portion)):
        cumulative_p.append(round(sum(portion[:p+1]),3)) # precision being 10^-3
    
    # (5),(6)
    parent = []
    parent_index = []
    while True:
        if len(parent) == 2:
            break
        else:
            randomNumber = random.uniform(0, 1) # (5) Generate a random numbrt
            # find index
            for p in range(len(cumulative_p)):
                if cumulative_p[p] >= randomNumber:
                    break
            if population[p] in parent:
                pass
            else:
                parent.append(population[p])
                print('  parent %s'%len(parent),'in %s : '%p,population[p]) # (6)
                parent_index.append(p)
    print('  parent: ',parent)            
    # # ------------------------------------two point crossover ------------------------------------           
    # # two point crossover, CROSSOVER_RATE = 0.9
    # crossover_location = []
    
    # while True:
    #     if len(crossover_location) == 2:
    #         break
    #     else:
    #         randomNumber = random.uniform(0, 1)
    #         if randomNumber < CROSSOVER_RATE:
    #             # find crossover location
    #             randomNumber = random.uniform(0, 1)
    #             location = math.ceil(randomNumber * 10)
    #             crossover_location.append(location)
    # print('  crossover_location',crossover_location)
    
    # # generate two children
    # temp1 = parent[0][min(crossover_location):max(crossover_location)]
    # temp2 = parent[1][min(crossover_location):max(crossover_location)]

    # children = []
    # child1 = parent[0].copy()
    # child1[min(crossover_location):max(crossover_location)]=temp2
    # children.append(child1)
    
    # child2 = parent[1].copy()
    # child2[min(crossover_location):max(crossover_location)]=temp1

    # children.append(child2)
    # print('  after crossover: ',children)
    
    # # ------------------------------------ mutation ------------------------------------  
    # # one bit mutation, MUTATION_RATE = 0.1
    # print('  -------mutation------')
    # for i in range(2):
    #     randomNumber = random.uniform(0, 1)
    #     print('  ',children[i],randomNumber)
    #     if randomNumber > MUTATION_RATE:
    #         print('    No mutation for #',i+1)
    #         pass
    #     else:
    #         # do mutation
    #         print('    Do mutation for #',i+1)
    #         # find mutation location
    #         randomNumber = random.uniform(0, 1)
            
    #         location = math.ceil(randomNumber*11) 
    #         location = location - 1 # python index從0開始
    #         print('  mutation location: ',location)
    #         if children[i][location] == 1:
    #             children[i][location] = 0
    #         else:
    #             children[i][location] = 1
    # print('  after mutation: ',children)
    
    # # children chromosomes replace their parents and do for-loop
    # print('  -----------replace their parents------------')
    # # patents
    # print('  patent1 ',population[parent_index[0]])
    # print('  patent2 ',population[parent_index[1]])
    # print('')
    
    # # children chromosomes(check is it feasible) replace their parents
    # # calculate the optimal
    # temp1_c = children[0]
    # temp2_c = children[1]
    
    # newX_1 = Xbase10(temp1_c[:6])
    # newY_1 = Ybase10(temp2_c[-5:])
    # if newX_1+newY_1 > 5:
    #     print('     mutation1 非可行解，沒有更新後代',newX_1,newY_1) # pre-censoring
    # else:
    #     population[parent_index[0]] = children[0]
    #     f1 = fitness(newX_1,newY_1)
    #     print('     result1: ',f1,' (x,y)',newX_1,'  ',newY_1)
    # newX_2 = Xbase10(temp2[:6])
    # newY_2 = Ybase10(temp2[-5:])
    # if newX_2+newY_2 > 5:
    #     print('     mutation2 非可行解，沒有更新後代',newX_2,newY_2) # pre-censoring
    # else:
    #     population[parent_index[1]] = children[1]
    #     f2 = fitness(newX_2,newY_2)
    #     print('     result2: ',f2,' (x,y)',newX_2,'  ',newY_2)
    
    # # ------------------------------------ log ------------------------------------  
    # log.append(max(f1,f2))
    # if max(f1,f2) > max_log[-1]:
    #     max_log.append(max(f1,f2))
    #     if f1>f2:
    #         optimal_value = (newX_1,newY_1)
    #     else:
    #         optimal_value = (newX_2,newY_2)
    # else:
    #     max_log.append(max_log[-1])