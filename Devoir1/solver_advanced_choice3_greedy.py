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

    max_duration = 1195  # Temps alloué

    # On représente le graphe par sa matrice d'adjacence
    constraints = nx.to_numpy_matrix(schedule.conflict_graph, dtype=np.uint8)

    # Une solution est un tableau de taille n avec la couleur assignée à chaque noeud
    n = len(constraints)


    #initialisation générale
    solution = RLF_init(constraints)
    print("Init finished")
    best_k = solution.max()
    print("k Init : " + str(best_k + 1))
    best_sol = solution.copy()
    k = best_k

    ILS = 0
    inner = 0

    # Boucle ILS
    while time.time() < max_duration + start_time :
        ILS += 1

        # Mémoire pour fonction d'évaluation
        # Nombre d'arrêtes en conflit par couleur
        nb_conflits = np.zeros(n)
        # Nombre de noeuds par couleurs
        colors = np.bincount(solution)
        colors.resize(n)

        # Initialisation des mémoires
        for x in range(n):
            i = solution[x]
            for y in range(x):  # On traite les voisins inférieurs pour éviter les duplications
                if constraints[x,y]:
                    if i == solution[y]:
                        nb_conflits[i] += 1

        # Score d'avaluation de la sclution initiale
        score = np.sum(2 * nb_conflits * colors) - np.sum(np.square(colors))

        # Initialisation de la tabu queue
        if n < 20:
            # On n'interdit que le dernier mouvement pour les petites instances
            tabu_length = 1
            old_tabu_length = 1
        else:
            # La taille de la queue diminue quand on améliore la solution pour une meilleure diversification
            tabu_length = round(rd.randint(1, 11) + 0.6 * np.sum(nb_conflits))
            old_tabu_length = tabu_length

        tabu = deque()
        for _ in range(int(tabu_length)):
            tabu.appendleft((-1,-1))

        # Flag indiquant si on est sorti en trouvant un minima ou par limite de temps
        minima = False

        # Recherche locale interne
        while time.time() < max_duration + start_time:
            inner += 1

            # Sélection du premier voisin améliorant non tabou
            voisin = voisinage_first(n, k, solution, constraints, score, colors, nb_conflits, tabu)

            # Si on ne trouve pas de voisin améliorant on est dans un minima local et on passe à la prochaine boucle ILS
            # On sort immédiatement
            if voisin == -1:
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

        # On vérifie la raison de sortie
        # Si c'est un minima la solution est valide et donc on regarde si c'est la meilleure connue
        if minima and solution.max() < best_k:
            best_sol = solution.copy()
            best_k = solution.max()

        # Si le temps est écoulé on s'arrête
        if time.time() > max_duration + start_time:
            break

        # Si le temps n'est pas écoulé on passe à la prochaine boule ILS et un perturbe la solution
        # Deux bases sont possibles selon si on préfère privilégier l'intensification ou la diversification
        #solution = perturbation_greedy(solution, constraints, 0.1)  # Diversification
        solution = perturbation_greedy(best_sol, constraints, 0.1)  # Intensification
        k = solution.max()

    print("Nb de boucles ILS :" + str(ILS))
    print("Nb de boucles internes :" + str(inner))
    return dict(zip(schedule.course_list, best_sol.tolist()))


def greedy_init(constraints):
    '''
    Init greedy, on prends la couleur minimale disponible poru chaque noeud
    :param constraints: matrice d'adjacence
    :return res: solution
    '''
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
    :param n: nombre de noeuds
    :param k: nombres de couleurs
    :return: une solution aléatoire
    '''
    return np.random.randint(k, size=n)


def RLF_init(constraints):
    '''
    Algorithme RLF. Priorise les noeuds qui créent les plus grosses contraintes et olore simultanément les noeuds qui ne se contraignent pas entre eux
    :param constraints: matrice d'adjacence
    :return solution: solution
    '''
    n = len(constraints)
    solution = np.zeros(n, dtype=np.uint16)
    # Graphe des noeuds libres
    uncolored_graph = constraints.copy()
    uncolored_nodes = list(np.arange(n))
    k = 0
    # Temps qu'on a des noeuds libres on génère un ensemble à colorer d'une même couleur
    while uncolored_nodes:
        # Le noeud initial a le plsu de voisins non colorés
        initial = uncolored_nodes[np.argmax(np.sum(uncolored_graph[uncolored_nodes, :], axis=1))]
        # Liste des nouds de la partition
        color_nodes = [initial]

        # Noeuds libre dans la boucle
        open_nodes = uncolored_nodes.copy()
        open_graph = uncolored_graph.copy()

        # Graphe de noeuds ayant au moins un voisin dans la partition choisie
        adjacent_graph = np.zeros(constraints.shape, dtype=np.uint8)

        # Initialisationd es graphes
        open_nodes.remove(initial)
        open_graph[initial, :] = np.full((1, n), 0)
        open_graph[:, initial] = np.full((n, 1), 0)

        neighbours = np.nonzero(uncolored_graph[initial, :].transpose())[0]
        adjacent_graph[:, neighbours] = uncolored_graph[:, neighbours]

        for i in neighbours:
            open_nodes.remove(i)
        open_graph[neighbours, :] = np.full((len(neighbours), n), 0)
        open_graph[:, neighbours] = np.full((n, len(neighbours)), 0)

        # On complète la partition
        while open_nodes:
            # On choisit en priorité le noeuds qui qui le plus de voisins déjà contraints par les noeuds de la partition
            adjacent_counts = np.sum(adjacent_graph[open_nodes, :], axis=1)
            selected = np.array(open_nodes)[np.argwhere(adjacent_counts == adjacent_counts.max())[:, 0]]
            open_counts = np.sum(open_graph[selected, :], axis=1)

            # On sépare les égalités en privilégiant les voisins les moins contraigants sur les neoeuds encore ouverts et non adjacents
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

        # On applique la partition déterminée
        for i in color_nodes:
            uncolored_nodes.remove(i)

        uncolored_graph[color_nodes, :] = np.full((len(color_nodes), n), 0)
        uncolored_graph[:, color_nodes] = np.full((n, len(color_nodes)), 0)

        solution[color_nodes] = k

        # On passe à la couleur suivante
        k = k + 1

    return solution


def dsatur_init(constraints):
    '''
    Algorithme DSatur. Priorise les noeuds de haut degré avec déjà beaucoup de contraintes sur les couleurs libres
    :param constraints: matrice d'adjacence
    :return solution: solution
    '''
    n = len(constraints)
    solution = np.zeros(n, dtype=np.uint16)
    uncolored_graph = constraints.copy()
    uncolored_nodes = list(np.arange(n))
    colors = np.full((n, n + 1), n)
    saturation = np.zeros(n, dtype=np.uint16)

    # On colore chaque noeud successivement
    while uncolored_nodes:
        # On choisit le noeud avec le moins de couleurs disponible
        selected = np.array(uncolored_nodes)[np.argwhere(saturation[uncolored_nodes] == saturation[uncolored_nodes].max())[:, 0]]
        uncolored_counts = np.sum(uncolored_graph[selected, :], axis=1)

        # On sépare les égaux par le degré
        subselected = selected[np.argwhere(uncolored_counts == uncolored_counts.max())[:, 0]]
        final = np.random.choice(subselected)
        uncolored_nodes.remove(final)

        # On attribue la plus petite couleur libre
        k = 0
        while k in colors[final, :]:
            k += 1
        solution[final] = k

        # Mise à jour des mémoires
        uncolored_graph[final, :] = np.full(1, 0)
        uncolored_graph[:, final] = np.full(n, 0)

        neighbours = np.nonzero(np.squeeze(np.asarray(constraints[final, :])))[0]
        colors[neighbours, final] = np.full(len(neighbours), k)

        saturation = np.zeros(n, dtype=np.uint16)
        for i in range(n):
            saturation[i] = len(np.unique(colors[i, :])) - 1

    return solution


def voisinage_first(n, k, solution, constraints, score, colors, nb_conflits, tabu):
    """
    Recherche du premier voisin améliorant
    :param n: nombre de noeuds
    :param k: nombre couleurs
    :param solution: solution en cours
    :param constraints: matrice d'adjacence
    :param score: valeur déavaluation de la solution courante
    :param colors: mémoire des effectifs des partitions de couleurs
    :param nb_conflits: mémoire du nombre d'arrête dont les deux sommets sont dans la couelur de l'indice
    :param tabu: tabu queue pour gagner du temps sur l'exploration sans perte de garantie
    :return: premier voisin améliorant et mémoires mises à jour ou -1 pour indiquer un minima local
    """
    # Recherche exhaustive du premier voisin améliorant
    for x in range(n):
        i = solution[x]
        for j in range(k + 1):
            # On vérifie que le voisin est non tabu, cela permet de gagner un peu de temps d'exploration
            # Cela ne change pas la garantie de connaisance d'un minima local liée à al recherche exhaustive
            if not (x, j) in tabu:
                # On calcule les changements du nombre de contraintes
                added_j = 0
                removed_i = 0
                for y in np.nonzero(constraints[x].transpose())[0]:
                    l = solution[y]
                    if l == i:
                        removed_i += 1
                    if l == j:
                        added_j += 1

                # Création des mémoires mises à jour
                coltemp = colors.copy()
                coltemp[i] -= 1
                coltemp[j] += 1
                nb_conflits_temp = nb_conflits.copy()
                nb_conflits_temp[i] -= removed_i
                nb_conflits_temp[j] += added_j

                # Calcul explicite du score
                # Même si en théorie moisn d'opérations, la copy des arrays et les calculs numpy restent plus rapides
                # scoretemp = score + ((colors[i]-1)**2 - colors[i]**2 + (colors[j]+1)**2 - colors[j]**2) - 2 * (nb_conflits[i] + colors[i]*removed_i - removed_i) + 2 * (nb_conflits[j] + colors[j]*added_j - added_j)

                # Calcul sur les arrays modifiés
                scoretemp = np.sum(2 * nb_conflits_temp * coltemp) - np.sum(np.square(coltemp))

                if scoretemp < score:
                    # On retourne le voisin retenu ainsi que les nouvelles mémoires pour éviter le recalcul
                    return x, j, scoretemp, added_j, removed_i

    # Si on a exploré tout le voisinage sans améliorations possible on signale que c'est un minima local
    return -1


def perturbation_random(solution, gamma):
    '''
    Modification aléatoire d'une partie de la solution pour une diversification locale avec ILS
    :param solution: solution à modifier
    :param gamma: part de couleurs à retirer
    :return:  solution modifiée
    '''
    # On retire aléatoirement une part des couleurs
    n = len(solution)
    k = solution.max() + 1
    colors = rd.sample(range(0, int(solution.max())), int(round(solution.max() * gamma)))
    for x in range(n):
        if solution[x] in colors:
            solution[x] = n+1  # On utilise n+1 pour tag les cases vides

    # On recosntruit aléatoirement
    for i in range(n):
        if solution[i] == n+1:
            solution[i] = np.random.randint(k)

    return solution


def perturbation_greedy(solution, constraints, gamma):
    '''
    Modification greedy d'une partie de la solution pour une diversification locale avec ILS
    :param solution: solution à modifier
    :param constraints: matrice d'adjacence
    :param gamma: part de couleurs à retirer
    :return: solution modifiée
    '''
    # On retire aléatoirement une part des couleurs
    n = len(solution)
    colors = rd.sample(range(0, int(solution.max())), int(round(solution.max() * gamma)))
    for x in range(n):
        if solution[x] in colors:
            solution[x] = n + 1  # On utilise n+1 pour tag les cases vides

    # On reconstruit avec la couleurs la moins utilisée parmi les voisins
    for i in range(n):
        if solution[i] == n + 1:
            colors = np.zeros(n)
            for j in range(n):
                if solution[j] != n + 1 and constraints[i, j]:
                    colors[solution[j]] += 1
            solution[i] = np.argmin(colors)

    return solution