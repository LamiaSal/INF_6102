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
    start_time = time.time()
    print("Solveur direct greedy")

    max_duration = 1195
    patience = 200 #TODO : implémenter la patience dans la sous recherche pour utiliser au mieux le temps et faire un restart


    constraints = nx.to_numpy_matrix(schedule.conflict_graph, dtype=np.uint8)
    n = len(constraints)


    #initialisation générale
    solution = RLF_init(constraints)
    print("Init finished")
    best_k = solution.max()
    print("k Init : " + str(best_k + 1))
    best_sol = solution.copy()
    k = best_k

    stagnation = 0

    ILS = 0
    inner = 0

    # Boucle ILS
    while time.time() < max_duration + start_time :
        ILS += 1

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
            tabu_length = round(rd.randint(1, 11) + 0.6 * np.sum(nb_conflits))  # sqrt ? nb de noeuds en conflits ?
            old_tabu_length = tabu_length

        tabu = deque()
        for _ in range(int(tabu_length)):
            tabu.appendleft((-1,-1))

        minima = False

        # Recherche locale
        while time.time() < max_duration + start_time and stagnation < patience:
            inner += 1

            # Sélection du premier voisin améliorant non tabou
            voisin = voisinage_first(n, k, solution, constraints, score, colors, nb_conflits, tabu)

            if voisin == -1:  # Si on ne peut pas améliorer on est dans un minima local, on restart
                minima = True
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
                tabu_length = round(rd.randint(1, 11) + 0.6 * np.sum(nb_conflits))

            for _ in range(int(old_tabu_length - tabu_length)):
                if tabu:
                    tabu.pop()

        if solution.max() < best_k and minima:  # On garde la solution si elle est meilleure
            best_sol = solution.copy()
            best_k = solution.max()

        if time.time() > max_duration + start_time:
            break

        # Arrivé ici on veut faire un restart en essayant de sortir du minima local
        #solution = perturbation_greedy(solution, constraints, 0.1)  # Diversification
        solution = perturbation_greedy(best_sol, constraints, 0.1)  # Intensification
        k = solution.max()

    print("Nb de boucles ILS :" + str(ILS))
    print("Nb de boucles internes :" + str(inner))
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


def RLF_init(constraints):
    n = len(constraints)
    solution = np.zeros(n, dtype=np.uint16)
    uncolored_graph = constraints.copy()
    uncolored_nodes = list(np.arange(n))
    k = 0
    while uncolored_nodes:
        initial = uncolored_nodes[np.argmax(np.sum(uncolored_graph[uncolored_nodes, :], axis=1))]
        color_nodes = [initial]
        open_nodes = uncolored_nodes.copy()
        open_graph = uncolored_graph.copy()
        adjacent_graph = np.zeros(constraints.shape, dtype=np.uint8)

        open_nodes.remove(initial)
        open_graph[initial, :] = np.full((1, n), 0)
        open_graph[:, initial] = np.full((n, 1), 0)

        neighbours = np.nonzero(uncolored_graph[initial, :].transpose())[0]
        adjacent_graph[:, neighbours] = uncolored_graph[:, neighbours]

        for i in neighbours:
            open_nodes.remove(i)
        open_graph[neighbours, :] = np.full((len(neighbours), n), 0)
        open_graph[:, neighbours] = np.full((n, len(neighbours)), 0)

        while open_nodes:
            adjacent_counts = np.sum(adjacent_graph[open_nodes, :], axis=1)
            selected = np.array(open_nodes)[np.argwhere(adjacent_counts == adjacent_counts.max())[:, 0]]
            open_counts = np.sum(open_graph[selected, :], axis=1)
            subselected = selected[np.argwhere(open_counts == open_counts.min())[:, 0]]
            final = np.random.choice(subselected)
            color_nodes.append(final)

            neighbours = np.nonzero(open_graph[final, :].transpose())[0]
            adjacent_graph[:, neighbours] = open_graph[:, neighbours]

            for i in neighbours:
                open_nodes.remove(i)
            open_graph[neighbours, :] = np.full((len(neighbours), n), 0)
            open_graph[:, neighbours] = np.full((n, len(neighbours)), 0)

            open_nodes.remove(final)
            open_graph[final, :] = np.full((1, n), 0)
            open_graph[:, final] = np.full((n, 1), 0)

        for i in color_nodes:
            uncolored_nodes.remove(i)

        uncolored_graph[color_nodes, :] = np.full((len(color_nodes), n), 0)
        uncolored_graph[:, color_nodes] = np.full((n, len(color_nodes)), 0)

        solution[color_nodes] = k

        k = k + 1

    return solution


def dsatur_init(constraints):
    n = len(constraints)
    solution = np.zeros(n, dtype=np.uint16)
    uncolored_graph = constraints.copy()
    uncolored_nodes = list(np.arange(n))
    colors = np.full((n, n + 1), n)
    saturation = np.zeros(n, dtype=np.uint16)

    while uncolored_nodes:
        selected = np.array(uncolored_nodes)[
            np.argwhere(saturation[uncolored_nodes] == saturation[uncolored_nodes].max())[:, 0]]
        uncolored_counts = np.sum(uncolored_graph[selected, :], axis=1)
        subselected = selected[np.argwhere(uncolored_counts == uncolored_counts.max())[:, 0]]
        final = np.random.choice(subselected)
        uncolored_nodes.remove(final)

        k = 0
        while k in colors[final, :]:
            k += 1
        solution[final] = k

        uncolored_graph[final, :] = np.full(1, 0)
        uncolored_graph[:, final] = np.full(n, 0)

        neighbours = np.nonzero(np.squeeze(np.asarray(constraints[final, :])))[0]
        colors[neighbours, final] = np.full(len(neighbours), k)

        saturation = np.zeros(n, dtype=np.uint16)
        for i in range(n):
            saturation[i] = len(np.unique(colors[i, :])) - 1

    return solution


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
                for y in np.nonzero(constraints[x].transpose())[0]:
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
                scoretemp = np.sum(2 * nb_conflits_temp * coltemp) - np.sum(np.square(coltemp))
                if scoretemp < score:
                    return x, j, scoretemp, added_j, removed_i

    return -1


def voisinage_sample():
    pass


def perturbation_search_repair():
    pass


def perturbation_random(solution, gamma):
    n = len(solution)
    k = solution.max() + 1
    colors = rd.sample(range(0, int(solution.max())), int(round(solution.max() * gamma)))
    for x in range(n):
        if solution[x] in colors:
            solution[x] = n+1  # On utilise n+1 pour tag les cases vides

    for i in range(n):
        if solution[i] == n+1:
            solution[i] = np.random.randint(k)

    return solution


def perturbation_greedy(solution, constraints, gamma):
    n = len(solution)
    colors = rd.sample(range(0, int(solution.max())), int(round(solution.max() * gamma)))
    for x in range(n):
        if solution[x] in colors:
            solution[x] = n + 1  # On utilise n+1 pour tag les cases vides

    for i in range(n):
        if solution[i] == n + 1:
            colors = np.zeros(n)
            for j in range(n):
                if solution[j] != n + 1 and constraints[i, j]:
                    colors[solution[j]] += 1
            solution[i] = np.argmin(colors)

    return solution


def perturbation_dsatur():
    pass