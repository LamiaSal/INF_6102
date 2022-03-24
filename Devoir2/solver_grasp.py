import networkx as nx
import numpy as np
import time
from itertools import product

rng = np.random.default_rng()


def solve(mother):
    start_time = time.time()
    print("Solveur GRASP")

    max_duration = 1200  # Temps alloué

    # On représente le graphe par ses matrices de flux et de distance
    flows = nx.to_numpy_array(mother.graph, dtype=np.int32, weight="flow")
    dists = nx.to_numpy_array(mother.graph, dtype=np.int32, weight="dist")

    # Une solution est un tableau de taille n qui représente le slot attribué à chaque composant
    n = mother.n_components

    patience = n * 10
    max_time = 30  # Temps maximum par recherche tabou
    alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    m = len(alphas)
    means = np.zeros(m)
    counts = np.zeros(m, dtype=np.uint32)
    for i, alpha in enumerate(alphas):
        solution_init = construction(n, flows, dists, alpha)
        solution, _ = search(solution_init, n, flows, dists, patience, 10, max_duration, start_time)
        cost = evaluation(solution, flows, dists)
        means[i] = cost
        counts[i] += 1

    probs = np.full(m, 1 / m)

    best_cost = 1000000

    GRASP = 0
    Tabu = 0

    while time.time() < max_duration + start_time:  # On s'arrête de tenter de nouvelles recherches à la fin du temps
        GRASP += 1

        # Sélection de alpha
        i = rng.choice(m, p=probs)
        alpha = alphas[i]

        # Etape de construction
        solution_init = construction(n, flows, dists, alpha)

        # Etape de recherche
        solution, n_iter = search(solution_init, n, flows, dists, patience, max_time, max_duration, start_time)
        Tabu += n_iter

        # Mise à jour des alphas
        cost = evaluation(solution, flows, dists)
        means[i] = (means[i] * counts[i] + cost) / (counts[i] + 1)
        counts[i] += 1
        q = cost / means
        probs = q / q.sum() # Amélioration avec roulette équilibrée ?


        if cost < best_cost:
            best_cost = cost
            best_sol = solution.copy()

    print("Nb de boucles GRASP :", GRASP)
    print("Nb total de boucles Tabu :", Tabu)
    print("Pondérations alphas : ", probs)
    return best_sol.tolist()


def evaluation(solution, flows, dists):
    return np.sum(flows * dists[solution,:][:,solution])


def construction(n, flows, dists, alpha):
    """
    Construction greedy
    :param n: nombre de component/slots
    :param flows: matrice des fluxs
    :param dists: matrice des distances
    :param alpha: part de move envisagés
    :return res: solution
    """
    solution = np.zeros(n, dtype=np.uint8)
    open_components = list(np.arange(n))
    open_slots = list(np.arange(n))
    assigned_components = []
    for i in range(n):
        assigned_slots = solution[assigned_components]
        heuristic = np.zeros((len(open_components), len(open_slots)), dtype=np.int32)
        for i,component in enumerate(open_components) :
            assigned_costs = np.sum(flows[component, :][assigned_components] * dists[open_slots, :][:, assigned_slots], axis=1)
            heuristic[i] = assigned_costs

        min = heuristic.min()
        max = heuristic.max()

        threshold = round(min + alpha * (max - min))
        mask = heuristic.flatten() <= threshold
        index = np.nonzero(mask)[0]
        values = heuristic.flatten()[mask]
        if values.std() == 0:
            pick = rng.choice(index)
            component = open_components[pick // len(open_slots)]
            slot = open_slots[pick % len(open_slots)]
        else:
            standard_values = (values - values.mean()) / values.std()
            if standard_values.sum() != 0:  # Roulette rééquilibrée
                normal_standard_values = (standard_values - standard_values.min()) / (standard_values.max() - standard_values.min()) * 0.8 + 0.1
                normal_standard_values = 1/normal_standard_values
                normal_standard_values = normal_standard_values/normal_standard_values.sum()
            else :
                normal_standard_values = np.full(len(standard_values), 1 / len(standard_values))
            pick = rng.choice(index, p=normal_standard_values)  # Roulette standardisée
            component = open_components[pick // len(open_slots)]
            slot = open_slots[pick % len(open_slots)]

        solution[component] = slot
        open_components.remove(component)
        open_slots.remove(slot)
        assigned_components.append(slot)

    return solution


def search(solution, n, flows, dists, patience, max_time, max_duration, start_time_total):
    start_time = time.time()

    best_sol = solution.copy()
    best_cost = evaluation(solution, flows, dists)

    delta_matrix = np.full((n, n), np.iinfo(np.int32).max, dtype=np.int32)

    # Initialisation du tabu dictionnary
    keys = [ele for ele in product(range(n), repeat=2)]
    tabu_dict = dict.fromkeys(keys, -1)
    tabu_length = int(rng.integers(0.9 * n, 1.1 * n))
    tabu_change = 0

    Tabu = 0

    # Recherche tabu

    stagnation = 0
    while (patience == -1 or stagnation < patience) and time.time() < max_time + start_time and time.time() < max_duration + start_time_total:
        Tabu += 1

        # Calcul de la matrice de voisinage

        for i in range(n):
            for j in range(i):
                new_sol = solution.copy()
                new_sol[i], new_sol[j] = solution[j], solution[i]
                delta_matrix[i, j] = np.sum((flows * dists[new_sol, :][:, new_sol]))

        # Sélection du meilleur voisin non tabou (pas forcément améliorant)
        ind1, ind2 = np.unravel_index(np.argsort(delta_matrix, axis=None), delta_matrix.shape)
        m = 0
        i, j = ind1[m], ind2[m]
        while j < i:  # Les éléments de la moitié supérieure de la matrice sont en fin de liste et ne sont pas considérés
            if delta_matrix[i, j] < best_cost:  # Critère d'aspiration
                break
            elif tabu_dict[(i, solution[i])] == -1 or tabu_dict[(j, solution[j])] == -1 or tabu_dict[
                (i, solution[i])] + tabu_length < Tabu or tabu_dict[(j, solution[j])] + tabu_length < Tabu:
                break
            m += 1
            i, j = ind1[m], ind2[m]

        # Mise à jour de la solution et du nombre de conflits
        cost = delta_matrix[i, j]
        solution[i], solution[j] = solution[j], solution[i]

        # Mise à jour de la tabu queue
        tabu_dict[(i, solution[i])] = Tabu
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

    return best_sol, Tabu


def init(mother, number, timer):
    start_time = time.time()
    print("Initialiseur GRASP")

    max_duration = timer  # Temps alloué

    # On représente le graphe par ses matrices de flux et de distance
    flows = nx.to_numpy_array(mother.graph, dtype=np.int32, weight="flow")
    dists = nx.to_numpy_array(mother.graph, dtype=np.int32, weight="dist")

    # Une solution est un tableau de taille n qui représente le slot attribué à chaque composant
    n = mother.n_components

    patience = n * 10
    max_time = 10  # Temps maximum par recherche tabou
    alphas = [0, 0.1, 0.2, 0.3, 0.4]

    costs = []
    solution_list = []

    m = len(alphas)
    means = np.zeros(m)
    counts = np.zeros(m, dtype=np.uint32)
    for i, alpha in enumerate(alphas):
        solution_init = construction(n, flows, dists, alpha)
        solution, _ = search(solution_init, n, flows, dists, patience, 10, max_duration, start_time)
        cost = evaluation(solution, flows, dists)
        costs.append(cost)
        solution_list.append(solution.copy())
        means[i] = cost
        counts[i] += 1

    probs = np.full(m, 1 / m)

    GRASP = 0
    Tabu = 0

    while time.time() < max_duration + start_time:  # On s'arrête de tenter de nouvelles recherches à la fin du temps
        GRASP += 1

        # Sélection de alpha
        i = rng.choice(m, p=probs)
        alpha = alphas[i]

        # Etape de construction
        solution_init = construction(n, flows, dists, alpha)

        # Etape de recherche
        solution, n_iter = search(solution_init, n, flows, dists, patience, max_time, max_duration, start_time)
        Tabu += n_iter

        # Mise à jour des alphas
        cost = evaluation(solution, flows, dists)
        means[i] = (means[i] * counts[i] + cost) / (counts[i] + 1)
        counts[i] += 1
        q = cost / means
        probs = q / q.sum() # Amélioration avec roulette équilibrée ?

        costs.append(cost)
        solution_list.append(solution.copy())

    print("Nb de boucles GRASP :", GRASP)
    print("Nb total de boucles Tabu :", Tabu)

    costs = np.array(costs)
    solution_list = np.array(solution_list)
    print("Number of solutions generated : ", str(len(solution_list)))
    return solution_list[np.argsort(costs)][:number]