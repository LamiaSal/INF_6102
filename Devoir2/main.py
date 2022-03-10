import argparse
import time
from mother_card import MotherCard
import solver_naive
import solver_advanced

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument('--infile', type=str, default='input')
    parser.add_argument('--outfile', type=str, default='out.txt')
    parser.add_argument('--visufile', type=str, default='visu.png')
    parser.add_argument('--agent', type=str, default='naive')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    mother = MotherCard(args.infile)

    print("***********************************************************")
    print("[INFO] Start the solving of Mother Card Problem")
    print("[INFO] input file: %s" % args.infile)
    print("[INFO] output file: %s" % args.outfile)
    print("[INFO] visualization file: %s" % args.visufile)
    print("[INFO] number of components: %s" % (mother.n_components))
    print("***********************************************************")

    start_time = time.time()

    # Méthode à implémenter
    if args.agent == "naive":
        # assign a different time slot for each course
        solution = solver_naive.solve(mother)
    elif args.agent == "advanced":
        # Your nice agent
        solution = solver_advanced.solve(mother)
    else:
        raise Exception("This agent does not exist")

    solving_time = round((time.time() - start_time) / 60,2)

    mother.display_solution(solution,args.visufile)
    mother.save_solution(solution, args.outfile)
    
    print("***********************************************************")
    print("[INFO] Solution obtained")
    print("[INFO] Execution time : %s minutes" % solving_time)
    print("[INFO] Total cost : %s" % mother.get_total_cost(solution))
    print("[INFO] Sanity check passed : %s" % mother.verify_solution(solution))
    print("***********************************************************")