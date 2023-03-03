import random
from deap import creator, base, tools, algorithms
import array

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)
# weights 1.0, 求最大值
# weights -1.0, 求最小值
# 可以多個參數，weights=(1.0,-1.0) 表示，第一個參數求最大值，第二參數求最小值
# 創建染色體個體為一個array，typecode i 表示 single int