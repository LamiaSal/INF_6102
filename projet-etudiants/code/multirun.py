import argparse
import time
import eternity_puzzle
import solver_advanced_tabu as sat
import numpy as np
import random as r
#nohup python multirun.py --infile=instances/eternity_complet.txt --nrun=4 --runduration=10800 > logs.txt 2>errors.txt &
#process 5469


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument('--infile', type=str, default='input')
    parser.add_argument('--outfile', type=str, default='solution')
    parser.add_argument('--visufile', type=str, default='visualization')
    parser.add_argument('--nrun', type=int, default=1)
    parser.add_argument('--runduration', type=int, default=3600)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    for tag in range(args.nrun):
        r.seed(tag)
        np.random.seed(tag)
        e = eternity_puzzle.EternityPuzzle(args.infile)

        print("***********************************************************")
        print("[INFO] Start the solving Eternity II - Run : " + str(tag))
        print("[INFO] input file: %s" % args.infile)
        print("[INFO] output file: %s" % args.outfile+str(tag)+".txt")
        print("[INFO] visualization file: %s" % args.visufile+str(tag)+".png")
        print("[INFO] board size: %s x %s" % (e.board_size,e.board_size))
        print("***********************************************************")

        start_time = time.time()

        solution, n_conflict = sat.solve_advanced(e,tag,args.runduration)

        solving_time = round((time.time() - start_time) / 60,2)

        e.display_solution(solution, args.visufile+str(tag)+".png")
        e.print_solution(solution, args.outfile+str(tag)+".txt")


        print("***********************************************************")
        print("[INFO] Solution obtained")
        print("[INFO] Execution time: %s minutes" % solving_time)
        print("[INFO] Number of conflicts: %s" % n_conflict)
        print("[INFO] Feasible solution: %s" % (n_conflict == 0))
        print("[INFO] Sanity check passed: %s" % e.verify_solution(solution))
        print("***********************************************************")
