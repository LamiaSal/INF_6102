import numpy as np
import copy
import time as t


def solve_random(eternity_puzzle):
    """
    Random solution of the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """

    solution = []
    remaining_piece = copy.deepcopy(eternity_puzzle.piece_list)

    for i in range(eternity_puzzle.n_piece):
        range_remaining = np.arange(len(remaining_piece))
        piece_idx = np.random.choice(range_remaining)

        piece = remaining_piece[piece_idx]

        permutation_idx = np.random.choice(np.arange(4))

        piece_permuted = eternity_puzzle.generate_rotation(piece)[permutation_idx]

        solution.append(piece_permuted)

        remaining_piece.remove(piece)

    return solution, eternity_puzzle.get_total_n_conflict(solution)

def solve_best_random(eternity_puzzle, n_trial):
    """
    Random solution of the problem (best of n_trial random solution generated)
    :param eternity_puzzle: object describing the input
    :param n_trial: number of random solution generated
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution, the solution is the best among the n_trial generated ones
    """
    best_n_conflict = 1000000

    best_solution = None

    t0 = t.time()
    
    somme = 0

    for i in range(n_trial):

        cur_sol, cur_n_conflict = solve_random(eternity_puzzle)
        somme += cur_n_conflict
        if cur_n_conflict < best_n_conflict:
            best_n_conflict = cur_n_conflict
            best_solution = cur_sol

    assert best_solution != None

    tf =t.time()
    a = open("moyenne_random_"+eternity_puzzle.instance_file[10:], 'w')
    
    a.write(" num itération: " + str(n_trial) + "\n")
    a.write(" score total: " + str(somme) + "\n")
    a.write(" score moyen : " + str(somme/n_trial)+ "\n")
    a.write(" temps total: " + str(tf-t0)+ "\n")
    a.write(" temps moyen : " + str((tf-t0)/n_trial))

    a.close()

    return best_solution, best_n_conflict
