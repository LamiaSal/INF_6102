import argparse
import time
from mother_card import MotherCard
import solver_tabu as st
import numpy as np
import networkx as nx

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
    test_type="search" #"init

    if test_type=="init": #tests d'initialisation
        n_tests = 1000
        sum=0
        best_cost=10000000000
        for i in range(n_tests):
            solution = st.random_init(n)
            cost = st.evaluation(solution, flows, dists)
            sum+=cost
            if cost<best_cost:
                best_cost = cost
        print("Mean random : ", str(sum/n_tests))
        print("Best random : ", str(best_cost))
        sum = 0
        best_cost=10000000000
        for i in range(n_tests):
            solution = st.greedy_init1(n, flows, dists)
            cost = st.evaluation(solution, flows, dists)
            sum += cost
            if cost<best_cost:
                best_cost = cost
        print("Mean greedy1 : ", str(sum / n_tests))
        print("Best greedy1 : ", str(best_cost))
        sum = 0
        best_cost=10000000000
        for i in range(n_tests):
            solution = st.greedy_init2(n, flows, dists)
            cost = st.evaluation(solution, flows, dists)
            sum += cost
            if cost<best_cost:
                best_cost = cost
        print("Mean greedy2 : ", str(sum / n_tests))
        print("Best greedy2 : ", str(best_cost))
        sum = 0
        best_cost=10000000000
        for i in range(n_tests):
            solution = st.greedy_init3(n, flows, dists)
            cost = st.evaluation(solution, flows, dists)
            sum += cost
            if cost<best_cost:
                best_cost = cost
        print("Mean greedy3 : ", str(sum / n_tests))
        print("Best greedy3 : ", str(best_cost))
        sum = 0
        best_cost = 10000000000
        for i in range(n_tests):
            solution = st.greedy_init4(n, flows, dists)
            cost = st.evaluation(solution, flows, dists)
            sum += cost
            if cost < best_cost:
                best_cost = cost
        print("Mean greedy4 : ", str(sum / n_tests))
        print("Best greedy4 : ", str(best_cost))
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