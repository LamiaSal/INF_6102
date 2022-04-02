import argparse
import time
from factory import Factory
import solver_naive
import solver_advanced
import random as r
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument('--infile', type=str, default='input')
    parser.add_argument('--outfile', type=str, default='out.txt')
    parser.add_argument('--visufile', type=str, default='visu.png')
    parser.add_argument('--agent', type=str, default='naive')
    parser.add_argument('--dates', type=str, default=True)

    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_arguments()
    r.seed(10)
    np.random.seed(10)

    factory = Factory(args.infile)

    print("***********************************************************")
    print("[INFO] Start the solving of Factory Problem")
    print("[INFO] input file: %s" % args.infile)
    print("[INFO] output file: %s" % args.outfile)
    print("[INFO] visualization file: %s" % args.visufile)
    print("***********************************************************")

    start_time = time.time()

    # Méthode à implémenter
    if args.agent == "naive":
        # assign a different time slot for each course
        solution = solver_naive.solve(factory)
    elif args.agent == "advanced":
        # Your nice agent
        solution = solver_advanced.solve(factory)
    else:
        raise Exception("This agent does not exist")

    solving_time = round((time.time() - start_time) / 60,2)
    
    print("***********************************************************")
    print("[INFO] Solution obtained")
    print("[INFO] Execution time : %s minutes" % solving_time)
    print("[INFO] Total cost : %s" % factory.get_total_cost(solution))
    print("[INFO] Sanity check passed : %s" % factory.verify_solution(solution))
    print("[INFO] Writing the solution")
    factory.save_solution(solution, args.outfile)
    print("[INFO] Drawing the solution")
    factory.display_solution(solution,args.visufile)
    print("***********************************************************")
