from python_tsp.heuristics import solve_tsp_simulated_annealing, solve_tsp_local_search #return not optimal solution
from python_tsp.exact import solve_tsp_dynamic_programming, solve_tsp_brute_force #return the optimal solution
import time
import numpy as np
import matplotlib.pyplot as plt


def show(lx, ly, coords, dis):
    plt.cla()
    plt.scatter(coords[0], coords[1], color='r')
    plt.plot(lx,ly)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Total distance={dis}")
    plt.show()

def py_solve_tsp_local_search(distance, initial_order, method=''):
    print(f'solve_tsp_local_search')
    new_order, distance_ls = solve_tsp_local_search(
            distance,
            x0=initial_order,
            max_processing_time=100,
            #perturbation_scheme="two_opt" #ps3/two_opt
        )
    return new_order

def py_solve_tsp_simulated_annealing(distance):
    print(f'solve_tsp_simulated_annealing')
    new_order, distance = solve_tsp_simulated_annealing(distance)
    return new_order

def new_result(x,y,new_order):
    #print(new_order)
    x = np.array(x)[new_order]
    y = np.array(y)[new_order]
    new_distance = np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            new_distance[i][j] = np.sqrt(np.square(x[i]-x[j])+np.square(y[i]-y[j]))
    
    result = 0
    for j in new_order[1:]:
        result += new_distance[j-1][j]
    print(result)
    show(x, y, coords, result)

if __name__ == "__main__":
    start_time = time.time()
    x = [499,267,703,408,437,491,74,532,416,626,42,271,359,163,508,229,576,147,560,35,714,757,517,64,314,675,690,391,628,87,240,705,699,258,428,]
    y = [556,57,401,305,421,267,105,525,381,244,330,395,169,141,380,153,442,528,329,232,48,498,265,343,120,165,50,433,63,491,275,348,222,288,490]
    #x = [5,9,2,7,1]
    #y = [1,1,1,1,1]
    coords = np.array([x,y])

    #計算距離矩陣,distance[i][j]表示 i to j 的距離
    distance = np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            distance[i][j] = np.sqrt(np.square(x[i]-x[j])+np.square(y[i]-y[j]))

    initial_order = [i for i in range(len(x))]
    #print(initial_order)

    init_distance = 0
    for j in initial_order[1:]:
        init_distance += distance[j-1][j]
    print(init_distance)
    print()

    x_copy, y_copy = x.copy(), y.copy()
    # ##################################################
    new_order = py_solve_tsp_local_search(distance, initial_order, 'ps3')
    new_result(x_copy,y_copy,new_order)

    new_order = py_solve_tsp_local_search(distance, initial_order, 'two_opt')
    new_result(x_copy,y_copy,new_order)

    new_order = py_solve_tsp_simulated_annealing(distance)
    new_result(x_copy,y_copy,new_order)

    #new_order, distance = solve_tsp_dynamic_programming(distance)
    # ##################################################
    
    