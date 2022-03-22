import argparse
import time
from mother_card import MotherCard
import solver_iterated_tabu as st
import numpy as np
import networkx as nx
import solver_init_algos as init

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--infile', type=str, default='input')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    mother = MotherCard(args.infile)

    print("***********************************************************")
    print("[INFO] Start of the test")
    print("[INFO] input file: %s" % args.infile)
    print("[INFO] number of components: %s" % mother.n_components)
    print("***********************************************************")

    start_time = time.time()

    flows = nx.to_numpy_array(mother.graph, dtype=np.uint8, weight="flow")
    dists = nx.to_numpy_array(mother.graph, dtype=np.uint8, weight="dist")
    n = mother.n_components

    print("[INFO] Test informations:")
    #Tests here
    test_type="init"

    if test_type == "calcul":
        values = np.random.randint(256926, 259758, size=12)
        standard_values = (values - values.mean()) / values.std()
        normal_standard_values = (standard_values - standard_values.min()) / (standard_values.max() - standard_values.min()) * 0.8 + 0.1
        normal_standard_values = 1 / normal_standard_values
        normal_standard_values = normal_standard_values / normal_standard_values.sum()
        print(values)
        print(normal_standard_values)
        solution = st.random_init(n)

    if test_type=="evaluation": #tests d'évaluation
        sum = 0
        for i in range(10000):
            solution = st.random_init(n)
            cost = st.evaluation(solution, flows, dists)
            cost_ta = mother.get_total_cost(solution)
            if cost == cost_ta:
                sum += 1
        print("Evaluation custom : " + str(sum))

    if test_type=="init": #tests d'initialisation
        n_tests = 10000
        sum=0
        best_cost=10000000000
        for i in range(n_tests):
            solution = st.random_init(n)
            cost = init.evaluation(solution, flows, dists)
            sum+=cost
            if cost<best_cost:
                best_cost = cost
        print("Mean random : ", str(sum/n_tests))
        print("Best random : ", str(best_cost))
        sum = 0
        best_cost=10000000000
        for i in range(n_tests):
            solution = init.greedy_init1(n, flows, dists)
            cost = st.evaluation(solution, flows, dists)
            sum += cost
            if cost<best_cost:
                best_cost = cost
        print("Mean greedy1 : ", str(sum / n_tests))
        print("Best greedy1 : ", str(best_cost))
        sum = 0
        best_cost=10000000000
        for i in range(n_tests):
            solution = init.greedy_init2(n, flows, dists)
            cost = st.evaluation(solution, flows, dists)
            sum += cost
            if cost<best_cost:
                best_cost = cost
        print("Mean greedy2 : ", str(sum / n_tests))
        print("Best greedy2 : ", str(best_cost))
        sum = 0
        best_cost=10000000000
        for i in range(n_tests):
            solution = init.greedy_init3(n, flows, dists)
            cost = st.evaluation(solution, flows, dists)
            sum += cost
            if cost<best_cost:
                best_cost = cost
        print("Mean greedy3 : ", str(sum / n_tests))
        print("Best greedy3 : ", str(best_cost))
        sum = 0
        best_cost = 10000000000
        for i in range(n_tests):
            solution = init.greedy_init4(n, flows, dists)
            cost = st.evaluation(solution, flows, dists)
            sum += cost
            if cost < best_cost:
                best_cost = cost
        print("Mean greedy4 : ", str(sum / n_tests))
        print("Best greedy4 : ", str(best_cost))
        sum = 0
        best_cost = 10000000000
        for i in range(10):
            index = np.random.permutation(n)
            solution = init.idof(n, flows, dists, index)
            cost = st.evaluation(solution, flows, dists)
            sum += cost
            if cost < best_cost:
                best_cost = cost
        print("Mean idof : ", str(sum / n_tests))
        print("Best idof : ", str(best_cost))
        sum = 0
        best_cost = 10000000000
        for i in range(100):
            index = np.random.permutation(n)
            solution = init.greedy_statistical_init(n, flows, dists, max(n**2, 1000))
            cost = st.evaluation(solution, flows, dists)
            sum += cost
            if cost < best_cost:
                best_cost = cost
        print("Mean statistical : ", str(sum / n_tests))
        print("Best statistical : ", str(best_cost))
        print("Init finished")

    if test_type=="search": #Test de recherche
        solution = st.solve(mother)
        print("Final cost : " + str(st.evaluation(solution, flows, dists)))



    solving_time = round((time.time() - start_time) / 60, 2)

    print("***********************************************************")
    print("[INFO] Test successfully finished. Infos on final solution")
    print("[INFO] Execution time : %s minutes" % solving_time)
    print("[INFO] Total cost : %s" % mother.get_total_cost(solution))
    print("[INFO] Sanity check passed : %s" % mother.verify_solution(solution))
    print("***********************************************************")