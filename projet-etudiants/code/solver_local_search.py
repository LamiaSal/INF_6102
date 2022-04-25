import networkx as nx
import numpy as np
import time as t
from itertools import product
rng = np.random.default_rng()
import copy

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

    print(eternity_puzzle.instance_file[10:])

    print("Solveur local search")
    RESTART = 100
    board_size = eternity_puzzle.board_size
    max_duration = 1200*3 # Time allocated
    
    n = eternity_puzzle.n_piece # size of the niegbor set considered each time
    
    # initialize random solution
    solution, sol_cost = gd.solve_heuristic(eternity_puzzle, 10)
    best_sol, best_sol_cost = solution.copy(), sol_cost 
    print("cost départ",sol_cost)

    nb_iter_restart =0
    num_iter = 0
    t0 = t.time()
    moyenne = 0
    while t.time()-t0 < max_duration and  num_iter <1000000:
        solution, sol_cost = hill_climbing_first_improv(solution,sol_cost, n, board_size, eternity_puzzle)

        
        if best_sol_cost > sol_cost :
            best_sol = copy.deepcopy(solution)
            best_sol_cost = sol_cost
            nb_iter_restart = 0
        else:
            nb_iter_restart += 1
        if nb_iter_restart >= RESTART:
            print("RESTART numéro", num_iter, "best cost:",best_sol_cost, " previous cost",sol_cost)
            moyenne+=sol_cost
            solution, sol_cost = gd.solve_heuristic(eternity_puzzle, 10)
            
            nb_iter_restart = 0
            num_iter+=1
    
    tf =t.time()
    a = open("moyenne_"+eternity_puzzle.instance_file[10:], 'w')
    a.write(" num itération: " + str(num_iter) + "\n")
    a.write(" score total: " + str(moyenne) + "\n")
    a.write(" score moyen : " + str(moyenne/num_iter)+ "\n")
    a.write(" temps total: " + str(tf-t0)+ "\n")
    a.write(" temps moyen : " + str((tf-t0)/num_iter))

    a.close()
            
    return best_sol, best_sol_cost


def hill_climbing_first_improv(best_sol,best_sol_cost, n, board_size, eternity_puzzle):
    """
    hill climbing
    :best_sol: meilleur solution jusqu'à présent
    :best_sol_cost: cout de la solution
    :n: nombre de pièce dans le puzzle
    :board_size: largeur du puzzle
    :param eternity_puzzle: object describing the input
    :return: a solution after hill climbing
    """
    solution = best_sol.copy()
    for i in range(n):
        for j in range(i):
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
                solution,new_sol,board_size )
            
            # update if nighbor better
            if neighbor_score > 0:
                new_sol_cost = eternity_puzzle.get_total_n_conflict(new_sol)
                return new_sol, new_sol_cost
    return best_sol, best_sol_cost




def south_east_pos(i, solution, board_size):
    """
    function that output the position of pieces next to the piece i
    input :
    :i: index of piece i in solution
    :solution: a soluation
    :board_size: largeur du puzzle
    :param eternity_puzzle: object describing the input
    
    output :
    :i_i: x axis of piece of index i in solution
    :i_j: y axis of piece of index i in solution
    :new_sol_i_south: piece on the right of the piece solution[i]
    :new_sol_i_east: piece below the piece solution[i]
    """
    i_i= i%board_size
    i_j= i//board_size

    i_east = board_size * i_j + (i_i - 1)
    i_south = board_size * (i_j - 1) + i_i
    new_sol_i_south  = solution[i_south]
    new_sol_i_east = solution[i_east]

    return i_i, i_j, new_sol_i_south, new_sol_i_east

def eval_rotation(i, j, sol_k,sol_k_south,sol_k_east,board_size):
        """
        function qui calcul le nombre de conflit du pièce k si elle se trouvait 
        à la position (i,j) entouré des pièces sol_k_south,sol_k_east
        input :
        :i: coordonnée x de la piece considéré
        :j: coordonnée y de la piece considéré
        :sol_k: une pièce considéré
        :sol_k_south: pièce en dessous de sol_k
        :sol_k_east: pièce à droite de sol_k
        :board_size: largeur du puzzle
        
        output :
        :n_conflict: le nombre de conflits
        """
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
    solution,new_sol, board_size ) :
    """
    function qui calcul le nombre de conflit du pièce k si elle se trouvait 
    à la position (i,j) entouré des pièces sol_k_south,sol_k_east
    input :
    :(i_i, i_j): coordonnées de la pièce de coordonnée i considéré dans le 2-swap
    :(j_i, j_j): coordonnées de la pièce de coordonnée j considéré dans le 2-swap

    :(new_sol_i_south, new_sol_i_east): piéces voisins de la pièce i considéré dans le 2-swap
    :(new_sol_j_south, new_sol_j_east): piéces voisins de la pièce j considéré dans le 2-swap

    :solution: solution avant le 2-swap
    :new_sol: solution après le 2-swap
    :board_size: largeur du puzzle
    
    output :
    différence de conflits locales entre la solution de départ et la solution après le 2-swap
    """


    p_prev_1 = eval_rotation(i_i, i_j, solution[i], new_sol_i_south, new_sol_i_east, board_size)
    p_prev_2 = eval_rotation(j_i, j_j, solution[j], new_sol_j_south, new_sol_j_east, board_size)

    prev_local_n_conflicts= p_prev_1 + p_prev_2

    p_aft_1 = eval_rotation(i_i, i_j, new_sol[i], new_sol_i_south, new_sol_i_east, board_size)
    p_aft_2 = eval_rotation(j_i, j_j, new_sol[j], new_sol_j_south, new_sol_j_east, board_size)
    
    aft_local_n_conflicts= p_aft_1 + p_aft_2

    return prev_local_n_conflicts - aft_local_n_conflicts


