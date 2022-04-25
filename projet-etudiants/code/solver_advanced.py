import solver_advanced_tabu_graph as sat
import numpy as np
import random as r
r.seed(0)
np.random.seed(0)


def solve_advanced(eternity_puzzle):
    """
    Your solver for the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """

    return sat.solve_advanced(eternity_puzzle, 0, 3600)
