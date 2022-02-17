#Marche pas car ne garantit pas une solution acceptable

import networkx as nx
import numpy as np
import time
import random as rd
from collections import deque

def solve(schedule):
    """
    Your solution of the problem
    :param schedule: object describing the input
    :return: a list of tuples of the form (c,t) where c is a course and t a time slot. 
    """
    max_duration = 120
    patience = 200 #TODO : implémenter la patience dans la sous recherche pour utiliser au mieux le temps et faire un restart


    constraints = nx.to_numpy_matrix(schedule.conflict_graph, dtype=np.uint8)
    n = len(constraints)


    #initialisation générale
    solution = greedy_init(constraints)
    best_k = solution.max()
    best_sol = solution.copy()
    k = best_k


    start_time = time.time()
    stagnation = 0

    # Recherche par paliers
    while time.time() < max_duration + start_time : #Boucle ILS

        # Mémoire pour fonction d'évaluation
        nb_conflits = np.zeros(n)
        colors = np.bincount(solution)
        colors.resize(n)

        for x in range(n):
            i = solution[x]
            for y in range(x):  # On traite les voisins inférieurs pour éviter les duplications
                if constraints[x,y]:
                    if i == solution[y]:
                        nb_conflits[i] += 1

        score = np.sum(2 * nb_conflits * colors) - np.sum(np.square(colors))

        # Initialisation de la tabu queue
        if n < 20:
            tabu_length = 1
            old_tabu_length = 1
        else:
            tabu_length = round(rd.randint(1, 11) + 0.6 * nb_conflits)  # sqrt ? nb de noeuds en conflits ?
            old_tabu_length = tabu_length

        tabu = deque()

        # Recherche locale
        while time.time() < max_duration + start_time and stagnation < patience:

            # Sélection du premier voisin améliorant non tabou
            voisin = voisinage_first(n, k, solution, constraints, score, colors, nb_conflits, tabu)

            if voisin == (-1, -1):  # Si on ne peut pas améliorer on est dans un minima local, on restart
                break

            # Mise à jour de la mémoire et du nombre de couleurs
            colors[solution[voisin[0]]] -= 1
            colors[voisin[1]] += 1
            nb_conflits[voisin[1]] += voisin[3]
            nb_conflits[solution[voisin[0]]] -= voisin[4]

            # Mise à jour du score
            score = voisin[2]
            solution[voisin[0]] = voisin[1]
            k = solution.max()

            # Mise à jour de la tabu queue
            tabu.appendleft((voisin[0],voisin[1]))

            if n >= 20:
                old_tabu_length = tabu_length
                tabu_length = round(rd.randint(1, 11) + 0.6 * nb_conflits)

            for _ in range(old_tabu_length - tabu_length):
                tabu.pop()

        if solution.max() < best_k:  # On garde la solution si elle est meilleure
            best_sol = solution.copy()
            best_k = solution.max()

        if time.time() < max_duration + start_time or stagnation < patience:
            break

        # Arrivé ici on veut faire un restart en essayant de sortir du minima local
        solution = perturbation_greedy(solution, constraints, 0.1)
        k = solution.max()


    return dict(zip(schedule.course_list, best_sol.tolist()))


def greedy_init(constraints):
    n = len(constraints)
    res = np.zeros(n, dtype=np.uint16)

    for i in range(n):
        colors = np.zeros(n)
        for j in range(n):
            if constraints[i,j]:
                colors[res[j]] += 1
        res[i] = np.argmin(colors)

    return res


def random_init(n, k):
    '''
    Init total random
    :param n:
    :param k:
    :return:
    '''
    return np.random.randint(k, size=n)


def voisinage_min():
    pass


def voisinage_maxmin():
    pass


def voisinage_complet():
    pass


def voisinage_first(n, k, solution, constraints, score, colors, nb_conflits, tabu):
    for x in range(n):
        i = solution[x]
        for j in range(k + 1):
            if not (x,j) in tabu:
                added_j = 0
                removed_i = 0
                for y in np.nonzero(constraints[x]):
                    l = solution[y]
                    if l == i:
                        removed_i += 1
                    if l == j:
                        added_j += 1
                coltemp = colors.copy()
                coltemp[i] -= 1
                coltemp[j] += 1
                nb_conflits_temp = nb_conflits.copy()
                nb_conflits_temp[i] -= removed_i
                nb_conflits_temp[j] += added_j
                #scoretemp = score + ((colors[i]-1)**2 - colors[i]**2 + (colors[j]+1)**2 - colors[j]**2) - 2 * (nb_conflits[i] + colors[i]*removed_i - removed_i) + 2 * (nb_conflits[j] + colors[j]*added_j - added_j)
                scoretemp = np.sum(2 * nb_conflits * colors) - np.sum(np.square(colors))
                if scoretemp < score:
                    return (x, j, scoretemp, added_j, removed_i)

    return -1


def voisinage_sample():
    pass


def perturbation_search_repair():
    pass


def perturbation_greedy(solution, constraints, gamma):
    n = len(solution)
    colors = rd.sample(range(0, solution.max()),round(solution.max() * gamma))
    for x in range(n):
        if solution[x] in colors:
            solution[x] = n+1

    for i in range(n):
        if solution[i] == n+1 :
            colors = np.zeros(n)
            for j in range(n):
                if constraints[i,j]:
                    colors[solution[j]] += 1
            solution[i] = np.argmin(colors)

    return solution


def perturbation_random():
    pass


def perturbation_dsatur():
    pass