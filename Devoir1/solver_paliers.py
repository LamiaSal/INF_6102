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
    k = solution.max()
    best_sol = solution.copy()

    k -= 1  # On démarre la recherche par paliers au niveau de la solution gourmande

    start_time = time.time()
    stagnation = 0

    palier = 0
    inner = 0
    # Recherche par paliers
    while time.time() < max_duration + start_time :
        palier += 1

        # Solution initiale
        solution = greedy_init_k(constraints, k)

        # Matrice de coût de transition pour accélérer le calcul du meilleur voisin
        cost_matrix = np.zeros((n,k), dtype=np.int64)

        # Fonction d'évaluation
        nb_conflits = 0

        for x in range(n):
            i = solution[x]
            for y in range(x):  # On traite les voisins inférieurs pour éviter les duplications
                if constraints[x,y]:
                    l = solution[y]
                    if i == l:
                        nb_conflits += 1
                    for j in range(k):
                        cost_matrix[x,j] += int((l == j)) - int((l == i))
                        cost_matrix[y, j] += int((i == j)) - int((i == l))

        # Initialisation de la tabu queue
        if n < 20:
            tabu_length = 1
            old_tabu_length = 1
        else:
            tabu_length = round(rd.randint(1, 11) + 0.6 * nb_conflits)  # sqrt ? nb de noeuds en conflits ?
            old_tabu_length = tabu_length

        tabu = deque()

        # Recherche locale pour satisfaction à k couleurs
        while nb_conflits > 0 and time.time() < max_duration + start_time:
            inner += 1

            # Sélection du meilleur voisin non tabou
            ind1, ind2 = np.unravel_index(np.argsort(cost_matrix, axis=None), cost_matrix.shape)
            v = len(ind1)
            m = 0
            while m < v:
                x,j = ind1[m], ind2[m]
                if (x,j) not in tabu:
                    break
                m += 1

            # Mise à jour de la solution et du nombre de conflits
            nb_conflits += cost_matrix[x,j]
            i = solution[x]
            solution[x] = j

            # Mise à jour de la matrice des coûts
            for y in np.nonzero(constraints[x].transpose())[0]:
                if solution[y] != i:
                    cost_matrix[y, i] -= 1
                if solution[y] != j:
                    cost_matrix[y, j] += 1
                if solution[y] == j:
                    for c in range(k):
                        if c != j:
                            cost_matrix[y, c] -= 1
                        cost_matrix[x, c] -= 1
                if solution[y] == i:
                    for c in range(k):
                        if c != i:
                            cost_matrix[y, c] += 1
                        cost_matrix[x, c] += 1


            # Mise à jour de la tabu queue
            tabu.appendleft((x,j))

            if n >= 20:
                old_tabu_length = tabu_length
                tabu_length = round(rd.randint(1, 11) + 0.6 * nb_conflits)

            for _ in range(int(old_tabu_length - tabu_length)):
                if tabu:
                    tabu.pop()

        if nb_conflits == 0:  # Si on a réussi on continue au palier suivant
            best_sol = solution.copy()
            k -= 1
        elif nb_conflits < 0:
            raise(Exception("Evaluation des conflits négative"))
        else:  # Arrivé ici on est forcément sortie de la boucle par manque de temps donc fin
            break

    print(palier)
    print(inner)
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


def greedy_init_k(constraints, k):
    '''
    Init greedy à k couleurs, on essaye de minimiser les conflits sinon random tie break
    :param constraints:
    :param k:
    :return:
    '''
    n = len(constraints)
    res = np.zeros(n, dtype=np.uint16)

    for i in range(n):
        colors = np.zeros(k)
        for j in range(n):
            if constraints[i,j]:
                colors[res[j]] += 1
        res[i] = np.random.choice(np.flatnonzero(colors == colors.min()))

    return res


def random_init(n, k):
    '''
    Init total random
    :param n:
    :param k:
    :return:
    '''
    return np.random.randint(k, size=n)