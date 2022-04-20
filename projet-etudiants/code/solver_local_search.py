import networkx as nx
import numpy as np
import time as t
from itertools import product
rng = np.random.default_rng()
import copy

import solver_random
import solver_heuristic_greedy as gd

GRAY = 0
BLACK = 23
RED = 24
WHITE = 25

NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3



def solve_local_search(eternity_puzzle):
    """
    Local search solution of the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """


    print("Solveur local search")
    RESTART = 30
    board_size = eternity_puzzle.board_size
    max_duration = 200 # Time allocated
    
    n = eternity_puzzle.n_piece # size of the niegbor set considered each time
    
    # initialize random solution
    #solution, sol_cost = solver_random.solve_best_random(eternity_puzzle, 1)
    solution, sol_cost = gd.solve_heuristic(eternity_puzzle, 10)
    best_sol, best_sol_cost = solution.copy(), sol_cost 
    print("cost départ",sol_cost)

    nb_iter_restart =0
    t0 = t.time()
    while t.time()-t0 < max_duration:

        solution, sol_cost = hill_climbing(solution, n, board_size, eternity_puzzle)

        if best_sol_cost > sol_cost :
            best_sol = copy.deepcopy(solution)
            best_sol_cost = sol_cost
            nb_iter_restart = 0
        else:
            nb_iter_restart += 1
        if nb_iter_restart >= RESTART:
            #solution, sol_cost = solver_random.solve_best_random(eternity_puzzle, 1)
            solution, sol_cost = gd.solve_heuristic(eternity_puzzle, 10)
            print("cost départ",sol_cost)
            print("RESTART numéro", nb_iter_restart, "best cost:",best_sol_cost, " new cost départ",sol_cost)
            nb_iter_restart = 0
            
    return best_sol, best_sol_cost


def hill_climbing(solution, n, board_size, eternity_puzzle):
    # TODO: si on considére pas tous les vosinages, prendre un subset 
    # mais randomizer le choix du i et du j de tel sortes à ce qu'ils soient unique tout de même
    for i in range(n):#eternity.board_size
        for j in range(i):#eternity.board_size
            # TODO : question la matrix est symétrique non ??
            #print("i : ",i, " j : ", j)
            new_sol = solution.copy()

            # 2-swap
            new_sol[i], new_sol[j] = solution[j], solution[i]
            
            # rotating
            i_i, i_j, new_sol_i_south, new_sol_i_east = south_east_pos(i, new_sol, board_size)
            j_i, j_j, new_sol_j_south, new_sol_j_east = south_east_pos(j, new_sol, board_size)
            

            r_new_sol_i = new_sol[i]
            r_new_sol_j = new_sol[j]
            r_new_sol_i_score = eval_rotation(i_i, i_j, r_new_sol_i, new_sol_i_south, new_sol_i_east, board_size)
            r_new_sol_j_score = eval_rotation(j_i, j_j, r_new_sol_j, new_sol_j_south, new_sol_j_east, board_size)
            
            
            for permutation_idx in range(4):
                r_new_sol_i_c = eternity_puzzle.generate_rotation(new_sol[i])[permutation_idx]
                r_new_sol_i_c_score = eval_rotation(i_i, i_j, r_new_sol_i_c, new_sol_i_south, new_sol_i_east, board_size)
                
                if r_new_sol_i_c_score < r_new_sol_i_score  :
                    r_new_sol_i  = r_new_sol_i_c
                    r_new_sol_i_score = r_new_sol_i_c_score
                

                r_new_sol_j_c = eternity_puzzle.generate_rotation(new_sol[j])[permutation_idx]
                r_new_sol_j_c_score = eval_rotation(j_i, j_j, r_new_sol_j_c, new_sol_j_south, new_sol_j_east, board_size)
                
                if r_new_sol_j_score > r_new_sol_j_c_score :
                    r_new_sol_j  = r_new_sol_j_c
                    r_new_sol_j_score = r_new_sol_j_c_score
            
            new_sol[i], new_sol[j] =r_new_sol_i, r_new_sol_j

            # if value is positive it means the neighbor is a better solution
            neighbor_score = eval_neighbours(i, j,\
                i_i, i_j, new_sol_i_south, new_sol_i_east,\
                j_i, j_j, new_sol_j_south, new_sol_j_east,\
                solution,new_sol, eternity_puzzle,board_size )
            if neighbor_score > 0:
                solution = new_sol
        #print("i : ",i, " score : ", eternity_puzzle.get_total_n_conflict(solution))
    return solution, eternity_puzzle.get_total_n_conflict(solution)


def south_east_pos(i, solution, board_size):
    i_i= i%board_size
    i_j= i//board_size

    i_east = board_size * i_j + (i_i - 1)
    i_south = board_size * (i_j - 1) + i_i
    new_sol_i_south  = solution[i_south]
    new_sol_i_east = solution[i_east]

    return i_i, i_j, new_sol_i_south, new_sol_i_east
def eval_rotation(i, j, sol_k,sol_k_south,sol_k_east,board_size):
        
        n_conflict = 0

        if i > 0 and sol_k[WEST] != sol_k_east[EAST]:
            n_conflict += 1

        if i == 0 and sol_k[WEST] != GRAY:
            n_conflict += 1

        if i == board_size - 1 and sol_k[EAST] != GRAY:
            n_conflict += 1

        if j > 0 and sol_k[SOUTH] != sol_k_south[NORTH]:
            n_conflict += 1

        if j == 0 and sol_k[SOUTH] != GRAY:
            n_conflict += 1

        if j == board_size - 1 and sol_k[NORTH] != GRAY:
            n_conflict += 1
        return n_conflict




def eval_neighbours(i, j,\
    i_i, i_j, new_sol_i_south, new_sol_i_east,\
    j_i, j_j, new_sol_j_south, new_sol_j_east,\
    solution,new_sol, eternity_puzzle,board_size ) :

    p_prev_1 = eval_rotation(i_i, i_j, solution[i], new_sol_i_south, new_sol_i_east, board_size)
    p_prev_2 = eval_rotation(j_i, j_j, solution[j], new_sol_j_south, new_sol_j_east, board_size)

    prev_local_n_conflicts= p_prev_1 + p_prev_2

    p_aft_1 = eval_rotation(i_i, i_j, new_sol[i], new_sol_i_south, new_sol_i_east, board_size)
    p_aft_2 = eval_rotation(j_i, j_j, new_sol[j], new_sol_j_south, new_sol_j_east, board_size)
    
    aft_local_n_conflicts= p_aft_1 + p_aft_2

    return prev_local_n_conflicts - aft_local_n_conflicts



## BROUILLON
def eval_neighbours__(i, j, solution,new_sol, eternity_puzzle) : 

    board_size = eternity_puzzle.board_size
    i_i= i%board_size
    i_j= i//board_size

    j_i= j%board_size
    j_j= j//board_size

    p_prev_1 = eternity_puzzle.get_piece_n_confllicts(i_i, i_j, solution)
    p_prev_2 = eternity_puzzle.get_piece_n_confllicts(j_i, j_j, solution)
    
    prev_local_n_conflicts= p_prev_1 + p_prev_2

    p_aft_1 = eternity_puzzle.get_piece_n_confllicts(i_i, i_j, new_sol)
    p_aft_2 = eternity_puzzle.get_piece_n_confllicts(j_i, j_j, new_sol)
    aft_local_n_conflicts= p_aft_1 + p_aft_2

    return prev_local_n_conflicts - aft_local_n_conflicts