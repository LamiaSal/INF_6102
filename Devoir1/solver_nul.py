#Marche pas car ne garantit pas une solution acceptable

import networkx as nx
import numpy as np
import time

def solve(schedule):
    """
    Your solution of the problem
    :param schedule: object describing the input
    :return: a list of tuples of the form (c,t) where c is a course and t a time slot. 
    """
    max_duration = 120
    patience = 200


    constraints = nx.to_numpy_matrix(schedule.conflict_graph, dtype=np.uint8)
    n = len(constraints)

    #initialisation
    nodes = greedy_init(constraints)
    k = nodes.max()
    best_sol = nodes.copy()
    best_k = k
    print(k)

    start_time = time.time()
    stagnation = 0

    while time.time() < max_duration + start_time and stagnation < patience :

        #voisinage

        #On choisit un voisin qui crée des conflits en priorité
        voisins_possibles = set()
        nb_conflits = 0
        for i in range(n):
            for j in range(i):
                if nodes[i] == nodes[j]:
                    voisins_possibles.update([i,j])
                    nb_conflits += 1

        if nb_conflits == 0 and k < best_k:
            best_k = k
            best_sol = nodes.copy()

        if len(voisins_possibles) != 0:
            #Si on a trouvé un voisin avec des conflit on essaie de lui assigner une couleur qui n'en a pas ou on ajoute une couleur

            voisin = np.random.choice(np.array(list(voisins_possibles)))

            colors = np.zeros(k + 1)
            for i in range(n):
                if i != voisin and constraints[voisin, i]:
                    colors[nodes[i]] += 1

            if np.any(colors == 0):
                color = np.argmin(colors)
            else:
                color = k + 1
                k += 1

        else:
            # Si on n'a pas de conflits on choisit un voisin avec une couleur la moins représentée et on la retire

            color_counts = np.bincount(nodes)
            color = np.random.choice(np.flatnonzero(color_counts == color_counts.min()))
            voisin = np.random.choice(np.flatnonzero(nodes == color))

            colors = np.zeros(k)
            for i in range(n):
                if i != voisin and constraints[voisin, i]:
                    colors[nodes[i]] += 1

            color = np.random.choice(np.flatnonzero(colors == colors.min()))
            if color == nodes[voisin]:
                nodes[voisin] = k
            else:
                nodes[voisin] = color

            k -= 1

    #On retourne le dernier résultat acceptable vu
    res = dict(zip(schedule.course_list, best_sol.tolist()))
    return res




def greedy_init(constraints):
    n = len(constraints)
    res = np.zeros(n, dtype=np.uint16)

    for i in range(n):
        colors = np.zeros(n)
        for j in range(n):
            if constraints[i,j]:
                colors[res[j]] = 1
        res[i] = np.argmin(colors)

    return res


def voisinage_min():
    pass


def voisinage_maxmin():
    pass


def voisinage_complet():
    pass


def voisinage_sample():
    pass


def perturbation_search_repair():
    pass


def perturbation_dsatur():
    pass
