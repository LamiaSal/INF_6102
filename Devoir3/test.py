import argparse
import time
from factory import Factory
import solver_advanced2 as sa2
import solver_advanced as sa
import random as r
import numpy as np
#python test.py --infile=instances/factory_B_10_5.txt


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument('--infile', type=str, default='input')
    parser.add_argument('--outfile', type=str, default='out_test.txt')
    parser.add_argument('--visufile', type=str, default='visu_test.png')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()
    r.seed(10)
    np.random.seed(10)

    factory = Factory(args.infile)

    print("***********************************************************")
    print("[INFO] Start testing")
    print("[INFO] input file: %s" % args.infile)
    print("[INFO] output file: %s" % args.outfile)
    print("***********************************************************")

    start_time = time.time()

    # Test à réaliser
    '''n_jobs, n_machines, n_ope, n_total_ope, durations = sa2.transform(factory)
    solution = sa2.init_base(n_jobs, n_ope)
    np.random.shuffle(solution)
    #solution, score = sa2.init_random(n_jobs,n_ope, n_machines, n_total_ope, durations, 100)
    start = time.time()
    for i in range(100):
        eval = sa2.evaluation(factory, sa2.interpret_base(solution, n_jobs, n_machines, n_ope, n_total_ope, durations))
    end = time.time()
    print("Decodeur greedy : ", end - start)
    start = time.time()
    for i in range(100):
        eval = sa.f(factory, sa.decode_sol(factory, solution+1))
    end = time.time()
    print("ancien décodeur : ", end - start)
    start = time.time()
    for i in range(100):
        eval = sa2.f(factory, sa2.interpret_python_greedy(factory, solution+1))
    end = time.time()
    print("décodeur python greedy: ", end - start)'''
    #print(complete_solution)
    #print("Evaluation : ", sa2.evaluation(complete_solution))
    #print("Ancienne evaluation : ",sa.f(factory, sa.decode_sol(factory, solution+1)))
    solution = sa2.solve(factory)
    solution2 = sa.solve(factory)
    #solution = sa2.decode(sa2.interpret_base_exact(solution, n_jobs, n_machines, n_ope, n_total_ope, durations))
    #solution = sa2.solve(factory)

    solving_time = round((time.time() - start_time) / 60, 2)

    print("***********************************************************")
    print("[INFO] Test finished")
    print("[INFO] Results on last solution :")
    print("[INFO]   Execution time : %s minutes" % solving_time)
    print("[INFO]   Total cost : %s" % factory.get_total_cost(solution))
    print("[INFO]   Sanity check passed : %s" % factory.verify_solution(solution))
    print("[INFO]   Writing the solution")
    factory.save_solution(solution, args.outfile)
    print("[INFO]   Drawing the solution")
    factory.display_solution(solution,args.visufile)
    print("***********************************************************")