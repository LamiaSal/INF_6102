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
        solution = greedy_init4(n, flows, dists)
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
                elif tabu_dict[(i, solution[i])] == -1 or tabu_dict[(j, solution[j])] == -1 or tabu_dict[(i, solution[i])] + tabu_length < Tabu or tabu_dict[(j, solution[j])] + tabu_length < Tabu:
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


def random_init(n):
    """
    Init total random
    :param n: nombre de machines
    :return: une solution aléatoire
    """
    return rng.permutation(n)


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
        solution[machine] = slot
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
        open_ratio = ratio[open_machines, :][:, open_slots]
        best = np.argwhere(open_ratio == open_ratio.min())
        machine_i, slot_i = best[rng.integers(len(best))]
        machine = open_machines[machine_i]
        slot = open_slots[slot_i]
        solution[machine] = slot
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
    assigned_machines = []
    for i in range(n):
        assigned_slots = solution[assigned_machines]
        open_flows = sum_flows[open_machines]
        assigned_flows = np.sum(flows[open_machines, :][:, assigned_machines], axis=1)
        selected = np.argwhere(assigned_flows == assigned_flows.max())[:, 0]
        unassigned_flows = open_flows[selected] - assigned_flows[selected]
        subselected = np.array(open_machines)[selected[np.argwhere(unassigned_flows == unassigned_flows.min())[:, 0]]]
        machine = np.random.choice(subselected)

        assigned_costs = np.sum(flows[machine, :][assigned_machines] * dists[:, assigned_slots], axis=1)[open_slots]

        #Tie braek aléatoire
        #slot = open_slots[np.random.choice(np.argwhere(assigned_costs == assigned_costs.min())[:, 0])]

        #Tie Break par min distance unassigned
        open_dists = sum_dists[open_slots]
        assigned_dists = np.sum(dists[open_slots, :][:, assigned_slots], axis=1)
        selected = np.argwhere(assigned_costs == assigned_costs.min())[:, 0]
        unassigned_dists = open_dists[selected] - assigned_dists[selected]
        subselected = np.array(open_slots)[selected[np.argwhere(unassigned_dists == unassigned_dists.min())[:, 0]]]
        slot = np.random.choice(subselected)

        solution[machine] = slot
        open_machines.remove(machine)
        open_slots.remove(slot)
        assigned_machines.append(slot)

    return solution

def greedy_init4(n, flows, dists, random_degree=1):
    """
    Init greedy. On choisit le noeuds avec le plus de flow sortant et on lui attribue le slot qui minimise le coût par rapport aux slots déjà attribués
    On a 3 manières intéressantes d'ordonner les choix des noeuds:
        - random
        - max total flow, random tie break (total assigned flow tie break ?)
        - max assigned flow, max unassigned flow tie break
    L'ordonnancement et les tie break dépendent du degré d'aléatoire voulu. 1 est le plus performant en général sur 1000 générations.
    :param n: nombre de machine/slots
    :param flows: matrice des fluxs
    :param dists: matrice des distances
    :param random_degree: degré d'aléatoire dans l'initialisation, à 0 on est pareil que greedy3, à 1 on random tie break sur le slot assigné, à 2 on random tie break sur la machine choisie et à 3 on choisi la machine aléatoirement
    :return res: solution
    """
    solution = np.zeros(n, dtype=np.uint8)
    sum_flows = np.sum(flows, axis=1)
    sum_dists = np.sum(dists, axis=1)
    open_machines = list(np.arange(n))
    open_slots = list(np.arange(n))
    assigned_machines = []
    for i in range(n):
        if random_degree < 2:
            assigned_slots = solution[assigned_machines]
            open_flows = sum_flows[open_machines]
            assigned_flows = np.sum(flows[open_machines, :][:, assigned_machines], axis=1)
            selected = np.argwhere(assigned_flows == assigned_flows.max())[:, 0]
            unassigned_flows = open_flows[selected] - assigned_flows[selected]
            subselected = np.array(open_machines)[selected[np.argwhere(unassigned_flows == unassigned_flows.min())[:, 0]]]
            machine = np.random.choice(subselected)
        elif random_degree == 3:
            assigned_slots = solution[assigned_machines]
            assigned_flows = np.sum(flows[open_machines, :][:, assigned_machines], axis=1)
            selected = np.array(open_machines)[np.argwhere(assigned_flows == assigned_flows.max())[:, 0]]
            machine = np.random.choice(selected)
        else:
            machine = np.random.choice(open_machines)

        assigned_costs = np.sum(flows[machine, :][assigned_machines] * dists[:, assigned_slots], axis=1)[open_slots]

        #Tie braek aléatoire
        if random_degree != 0:
            slot = np.array(open_slots)[np.random.choice(np.argwhere(assigned_costs == assigned_costs.min())[:, 0])]
        #Tie Break par min distance unassigned
        else:
            open_dists = sum_dists[open_slots]
            assigned_dists = np.sum(dists[open_slots, :][:, assigned_slots], axis=1)
            selected = np.argwhere(assigned_costs == assigned_costs.min())[:, 0]
            unassigned_dists = open_dists[selected] - assigned_dists[selected]
            subselected = np.array(open_slots)[selected[np.argwhere(unassigned_dists == unassigned_dists.min())[:, 0]]]
            slot = np.random.choice(subselected)

        solution[machine] = slot
        open_machines.remove(machine)
        open_slots.remove(slot)
        assigned_machines.append(slot)

    return solution


def greedy_statistical_init(n, flows, dists, n_samples=30):
    return


def CATCH():
    return


def idof():
    return


def beamsearch_init():
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
    n = len(solution)
    open_machines = rng.choice(n, round(n * gamma), replace=False).tolist()
    open_slots = solution[open_machines].tolist()
    assigned_machines = [slot for slot in range(n) if slot not in open_machines]
    for i in range(round(n * gamma)):
        machine = rng.choice(open_machines)
        assigned_costs = np.sum(flows[machine, :][assigned_machines] * dists[:, solution[assigned_machines]],axis=1)[open_slots]
        slot = open_slots[np.random.choice(np.argwhere(assigned_costs == assigned_costs.min())[:, 0])]
        solution[machine] = slot
        open_machines.remove(machine)
        open_slots.remove(slot)
        assigned_machines.append(slot)
    return solution
