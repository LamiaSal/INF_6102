import networkx as nx
import numpy as np
import time
from itertools import product
import solver_init_algos as init

rng = np.random.default_rng()


def solve(mother):
    """
    Random resolution of the problem
    :param mother: object describing the input
    :return: a list of integers of size n where the i-th element of the list is the component located in site i
    """
    start_time = time.time()
    print("Solveur iterated tabu")

    max_duration = 1200  # Temps alloué

    # On représente le graphe par ses matrices de flux et de distance
    flows = nx.to_numpy_array(mother.graph, dtype=np.int32, weight="flow")
    dists = nx.to_numpy_array(mother.graph, dtype=np.int32, weight="dist")

    # Une solution est un tableau de taille n qui représente le slot attribué à chaque composant
    n = mother.n_components

    patience = n * 10
    n_init = 10000

    # Initialisation générale
    best_cost = 1000000
    best_sol = None
    for i in range(n_init):
        solution = init.greedy_init4(n, flows, dists)
        cost = evaluation(solution, flows, dists)
        if cost < best_cost:
            best_cost = cost
            best_sol = solution
    solution = best_sol.copy()
    print("Init finished")
    cost = best_cost
    print("Cost Init : " + str(cost))

    delta_matrix = np.full((n, n), np.iinfo(np.int32).max, dtype=np.int32)
    '''neigh = []
    for i in range(n):
        for j in range(i):
            neigh.append((i,j))
    delta_array = np.full(len(neigh), np.iinfo(np.int32).max, dtype=np.int32)'''

    # Initialisation du tabu dictionnary
    keys = [ele for ele in product(range(n), repeat=2)]
    tabu_dict = dict.fromkeys(keys, -1)
    tabu_length = int(rng.integers(0.9 * n, 1.1 * n))
    tabu_change = 0

    ILS = 0
    Tabu = 0

    # Recherche par paliers
    while time.time() < max_duration + start_time:  # On s'arrête de tenter de nouvelles recherches à la fin du temps
        ILS += 1

        stagnation = 0
        while (patience == -1 or stagnation < patience) and time.time() < max_duration + start_time:
            Tabu += 1

            # Calcul de la matrice de voisinage

            for i in range(n):
                for j in range(i):
                    new_sol = solution.copy()
                    new_sol[i], new_sol[j] = solution[j], solution[i]
                    delta_matrix[i, j] = np.sum((flows * dists[new_sol, :][:,new_sol]))
            '''for k,(i,j) in enumerate(neigh):
                new_sol = solution.copy()
                new_sol[i], new_sol[j] = solution[j], solution[i]
                delta_array[k] = np.sum((flows * dists[new_sol, :][:,new_sol]))'''

            # Sélection du meilleur voisin non tabou (pas forcément améliorant)
            ind1, ind2 = np.unravel_index(np.argsort(delta_matrix, axis=None), delta_matrix.shape)
            m = 0
            i, j = ind1[m], ind2[m]
            while j<i: #Les éléments de la moitié supérieure de la matrice sont en fin de liste et ne sont pas considérés
                if delta_matrix[i,j] < best_cost: #Critère d'aspiration
                    break
                elif tabu_dict[(i, solution[j])] == -1 or tabu_dict[(j, solution[i])] == -1 or tabu_dict[(i, solution[j])] + tabu_length < Tabu or tabu_dict[(j, solution[i])] + tabu_length < Tabu:
                    break
                m += 1
                i, j = ind1[m], ind2[m]

            '''ind = np.argsort(delta_array)
            for m in ind:
                i, j = neigh[m]
                if delta_array[m] < best_cost:  # Critère d'aspiration
                    break
                elif tabu_dict[(i, solution[i])] == -1 or tabu_dict[(j, solution[j])] == -1 or tabu_dict[(i, solution[i])] + tabu_length < Tabu or tabu_dict[(j, solution[j])] + tabu_length < Tabu:
                    break'''

            # Mise à jour de la solution et du nombre de conflits
            #cost = delta_array[m]
            cost = delta_matrix[i,j]
            solution[i], solution[j] = solution[j], solution[i]

            # Mise à jour de la tabu queue
            tabu_dict[(i,solution[i])] = Tabu
            tabu_dict[(j, solution[j])] = Tabu
            if tabu_change == round(2.2 * n):
                tabu_length = int(rng.integers(0.9 * n, 1.1 * n))
                tabu_change = 0
            else:
                tabu_change += 1

            if cost >= best_cost:
                stagnation += 1
            else:
                best_cost = cost
                best_sol = solution.copy()
                stagnation = 0

        if patience != -1 and stagnation >= patience:
            #Perturbation et restart
            solution = perturbation_greedy(solution, flows, dists, 0.1)
            cost = evaluation(solution, flows, dists)
            if cost < best_cost:
                best_cost = cost
                best_sol = solution.copy()
        elif cost < 0:
            raise (Exception("Evaluation du coût négative"))
        else:  # Arrivé ici on est forcément sortie de la boucle par manque de temps donc fin
            break

    print("Nb de boucles ILS :", ILS)
    print("Nb total de boucles Tabu :", Tabu)
    return best_sol.tolist()


def evaluation(solution, flows, dists):
    return np.sum(flows * dists[solution,:][:,solution])


def perturbation_random(solution, gamma):
    """
    Perturbation et reconstruction aléatoire d'une partie de la solution
    :param solution: solution à perturber
    :param gamma: part de la solution randomisée
    :return: solution perturbée
    """
    n = len(solution)
    removing = rng.choice(n, 2*round(n * gamma / 2), replace=False)
    for i in range(len(removing)/2):
        solution[removing[i]], solution[removing[i+len(removing)/2]] = solution[removing[i+len(removing)/2]], solution[removing[i]]

    return solution


def perturbation_greedy(solution, flows, dists, gamma):
    """
    Perturbation et reconstruction greedy avec choix d'ordre aléatoire pour diversification
    :param solution: solution à perturber
    :param flows: matrice des fluxs
    :param dists: matrice des distances
    :param gamma: part de la solution randomisée
    :return: solution perturbée
    """
    n = len(solution)
    open_components = rng.choice(n, round(n * gamma), replace=False).tolist()
    open_slots = solution[open_components].tolist()
    assigned_components = [slot for slot in range(n) if slot not in open_components]
    for i in range(round(n * gamma)):
        component = rng.choice(open_components)
        assigned_costs = np.sum(flows[component, :][assigned_components] * dists[:, solution[assigned_components]],axis=1)[open_slots]
        slot = open_slots[np.random.choice(np.argwhere(assigned_costs == assigned_costs.min())[:, 0])]
        solution[component] = slot
        open_components.remove(component)
        open_slots.remove(slot)
        assigned_components.append(slot)
    return solution
