import networkx as nx
import numpy as np
import time
from itertools import product

rng = np.random.default_rng()


def solve(mother):
    """
    Random resolution of the problem
    :param mother: object describing the input
    :return: a list of integers of size n where the i-th element of the list is the component located in site i
    """
    start_time = time.time()
    print("Solveur tabu")

    max_duration = 1200  # Temps alloué
    patience = 30

    # On représente le graphe par ses matrices de flux et de distance
    dtype = np.dtype([("flow", np.uint8), ("dist", np.uint8)])
    graph_dict = nx.to_numpy_array(mother.graph, dtype=dtype, weight=None)
    flows, dists = graph_dict["flow"], graph_dict["dist"]

    # Une solution est un tableau de taille n qui représente la machine attribué à chaque slot
    n = mother.n_components

    # Initialisation générale
    solution = greedy_init3(n, flows, dists)
    print("Init finished")
    cost = evaluation(solution, flows, dists)
    print("Cost Init : " + str(cost))
    best_sol = solution.copy()
    best_cost = cost

    # Initialisation du tabu dictionnary
    keys = [ele for ele in product(range(n), repeat=2)]
    tabu_dict = dict.fromkeys(keys, -1)
    tabu_length = int(rng.randint(0.9 * n, 1.1 * n))
    tabu_change = round(2.2 * n)

    ILS = 0
    Tabu = 0

    # Recherche par paliers
    while time.time() < max_duration + start_time:  # On s'arrête de tenter de nouvelles recherches à la fin du temps
        ILS += 1

        # Recherche locale pour satisfaction à k couleurs
        last_move = None
        stagnation = 0
        while stagnation < patience and time.time() < max_duration + start_time:
            Tabu += 1

            # Calcul de la matrice de coût
            # Matrice de coût de transition pour accélérer le calcul du meilleur voisin
            # delta_matrix[i,j] est le coût ajouté à l'évaluation si on swap les positions des machines i et j
            if last_move is None:
                delta_matrix = np.full((n, n), np.iinfo(np.int16).max, dtype=np.int16)
                for i in range(n):
                    for j in range(i):
                        new_sol = solution.copy()
                        new_sol[i], new_sol[j] = solution[j], solution[i]
                        delta_matrix[i, j] = np.sum((flows[new_sol][:,new_sol] * dists) - (flows[solution][:,solution] * dists))
            else:
                u, v = last_move
                for i in range(n):
                    for j in range(i):
                        new_sol = solution.copy()
                        new_sol[i], new_sol[j] = solution[j], solution[i]
                        if i != u and j != v:
                            #TODO:Debug
                            delta_matrix[i, j] = delta_matrix[i, j] + 2 * (dists[u,i] - dists[u,j] + dists[v,j] - dists[v,i]) * (flows[new_sol[v],new_sol[i]] - flows[new_sol[v],new_sol[j]] + flows[new_sol[u],new_sol[j]] - flows[new_sol[u],new_sol[i]])
                        else:
                            delta_matrix[i, j] = np.sum((flows[new_sol][:,new_sol] * dists) - (flows[solution][:,solution] * dists))

            # Sélection du meilleur voisin non tabou (pas forcément améliorant)
            ind1, ind2 = np.unravel_index(np.argsort(delta_matrix, axis=None), delta_matrix.shape) #TODO:Debug
            v = len(ind1)
            m = 0
            i, j = ind1[m], ind2[m]
            while j<i: #Les éléments de la moitié supérieure de la matrice sont en fin de liste et ne sont pas considérés
                if cost + delta_matrix[i,j] < best_cost: #Critère d'aspiration
                    break
                elif tabu_dict[(i,solution[i])] + tabu_length < Tabu and tabu_dict[(j,solution[j])] + tabu_length < Tabu:
                    break
                m += 1
                i, j = ind1[m], ind2[m]

            last_move = (i,j)

            # Mise à jour de la solution et du nombre de conflits
            cost += delta_matrix[i, j]
            solution[i], solution[j] = solution[j], solution[i]

            # Mise à jour de la tabu queue
            tabu_dict[last_move] = Tabu
            if tabu_change == round(2.2 * n):
                tabu_length = int(rng.randint(0.9 * n, 1.1 * n))
                tabu_change = 0
            else:
                tabu_change += 1


            if cost >= best_cost:
                stagnation += 1
            else:
                best_cost = cost
                best_sol = solution.copy()

        if stagnation < patience:
            #Perturbation et restart
            solution = perturbation_greedy(solution)
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



    return solution


def evaluation(solution, flows, dists):
    return np.sum(flows[solution][:,solution] * dists)


def greedy_init1(n, flows, dists):
    """
    Init greedy. On choisit le noeud avec le plus de flow sortant et on lui attribue le slot avec les plus petites distances sortantes
    On a 3 manières intéressantes d'ordonner les choix des noeuds:
        - random
		- max total flow, random tie break (total assigned flow tie break ?)
		- max assigned flow, max unassigned flow tie break
	Ici on utilise la deuxième
    :param n: nombre de machine/slots
    :param flows: matrice des fluxs
    :param dists: matrice des distances
    :return res: solution
    """
    solution = np.zeros(n, dtype=np.uint8)
    sum_flows = np.sum(flows, axis=1)
    sum_dists = np.sum(dists, axis=1)
    open_machines = list(np.arange(n))
    open_slots = list(np.arange(n))
    for i in range(n):
        open_flows = sum_flows[open_machines]
        machine = open_machines[np.random.choice(np.argwhere(open_flows == open_flows.max())[:, 0])]
        open_dists = sum_dists[open_slots]
        slot = open_slots[np.random.choice(np.argwhere(open_dists == open_dists.min())[:, 0])]
        solution[slot] = machine
        open_machines.remove(machine)
        open_slots.remove(slot)

    return solution


def greedy_init2(n, flows, dists):
    """
    Init greedy.
    On utilise un critère qui synthétise les informations sur les flows et distance pour faire des assignation directes
    2 critères semblent intéressants:
        - minimiser le ratio somme des distances sortantes/ somme des flows sortants poru chaque couple slot/machine
        - minimiser le produit
    Ici on minimise le ratio
    :param n: nombre de machine/slots
    :param flows: matrice des fluxs
    :param dists: matrice des distances
    :return res: solution
    """
    solution = np.zeros(n, dtype=np.uint8)
    sum_flows = np.sum(flows, axis=1)
    sum_dists = np.sum(dists, axis=1)
    ratio = np.expand_dims(sum_flows, axis=1) @ np.expand_dims(sum_dists, axis=0)
    open_machines = list(np.arange(n))
    open_slots = list(np.arange(n))
    for i in range(n):
        open_ratio = ratio[open_machines, open_slots]
        best = np.argwhere(open_ratio == open_ratio.min())
        machine, slot = best[rng.randint(len(best))]
        solution[slot] = machine
        open_machines.remove(machine)
        open_slots.remove(slot)

    return solution


def greedy_init3(n, flows, dists):
    """
    Init greedy. On choisit le noeuds avec le plus de flow sortant et on lui attribue le slot qui minimise le coût par rapport aux slots déjà attribués
    On a 3 manières intéressantes d'ordonner les choix des noeuds:
        - random
        - max total flow, random tie break (total assigned flow tie break ?)
        - max assigned flow, max unassigned flow tie break
    Ici on utilise la troisième
    :param n: nombre de machine/slots
    :param flows: matrice des fluxs
    :param dists: matrice des distances
    :return res: solution
    """
    solution = np.zeros(n, dtype=np.uint8)
    sum_flows = np.sum(flows, axis=1)
    sum_dists = np.sum(dists, axis=1)
    open_machines = list(np.arange(n))
    open_slots = list(np.arange(n))
    assigned_slots = []
    for i in range(n):
        # TODO:Debug et check
        assigned_machines = solution[assigned_slots]
        open_flows = sum_flows[open_machines]
        assigned_flows = np.sum(flows[open_machines,assigned_machines], axis=1)
        selected = np.argwhere(assigned_flows == assigned_flows.max())[:, 0]
        unassigned_flows = open_flows[selected] - assigned_flows[selected]
        subselected = np.array(open_machines)[selected[np.argwhere(unassigned_flows == unassigned_flows.max())[:, 0]]]
        machine = np.random.choice(subselected)

        assigned_costs = np.sum(flows[solution[machine]][:, solution[assigned_slots]] * dists[:,assigned_slots], axis=1)

        #Tie braek aléatoire
        #slot = open_slots[np.random.choice(np.argwhere(assigned_costs == assigned_costs.min())[:, 0])]

        #Tie Break par min distance unassigned
        open_dists = sum_dists[open_slots]
        assigned_dists = np.sum(dists[open_slots, assigned_slots], axis=1)
        unassigned_dists = open_dists[selected] - assigned_dists[selected]
        selected = np.argwhere(assigned_costs == assigned_costs.min())[:, 0]
        subselected = np.array(open_slots)[selected[np.argwhere(unassigned_dists == unassigned_dists.max())[:, 0]]]
        slot = np.random.choice(subselected)

        solution[slot] = machine
        open_machines.remove(machine)
        open_slots.remove(slot)
        assigned_slots.add(slot)

    return solution


def greedy_statistical_init(n, flows, dists, n_samples=30):
    return


def random_init(n):
    """
    Init total random
    :param n: nombre de machines
    :return: une solution aléatoire
    """
    return rng.permutation(n)


def CATCH():
    return


def idof():
    return


def perturbation_random(solution, gamma):
    n = len(solution)
    removing = rng.choice(n, 2*round(n * gamma / 2), replace=False)
    for i in range(len(removing)/2):
        solution[removing[i]], solution[removing[i+len(removing)/2]] = solution[removing[i+len(removing)/2]], solution[removing[i]]

    return solution


def perturbation_greedy(solution, flows, dists, gamma):
    """
    Perturbation et reconstruction greedy avec choix d'ordre aléatoire pour diversification
    :param solution:
    :param flows:
    :param dists:
    :param gamma:
    :return:
    """
    #TODO:Debug et check
    n = len(solution)
    open_slots = list(rng.choice(n, round(n * gamma), replcae=False))
    open_machines = solution[open_slots]
    assigned_slots = [slot for slot in range(n) if slot not in open_slots]
    for i in range(round(n * gamma)):
        machine = rng.choice(open_machines)
        assigned_costs = np.sum(flows[solution[machine]][:, solution[assigned_slots]] * dists[:, assigned_slots],axis=1)
        slot = open_slots[np.random.choice(np.argwhere(assigned_costs == assigned_costs.min())[:, 0])]
        solution[slot] = machine
        open_machines.remove(machine)
        open_slots.remove(slot)
        assigned_slots.add(slot)
    return