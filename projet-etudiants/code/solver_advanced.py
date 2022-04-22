import solver_advanced_tabu as sat
import numpy as np
import random as r
r.seed(10)
np.random.seed(10)
#python main.py --agent=advanced --infile=instances/eternity_B.txt

def solve_advanced(eternity_puzzle):
    """
    Your solver for the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    # TODO : Threading
    # TODO : Tracé des courbes et enregistrement dans des fichiers de visu
    # TODO : Moyenne sur plusieurs runs

    return sat.solve_advanced(eternity_puzzle)
