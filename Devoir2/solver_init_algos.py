import numpy as np

rng = np.random.default_rng()


def evaluation(solution, flows, dists):
    return np.sum(flows * dists[solution,:][:,solution])


def random_init(n):
    """
    Init total random
    :param n: nombre de components
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
    :param n: nombre de component/slots
    :param flows: matrice des fluxs
    :param dists: matrice des distances
    :return res: solution
    """
    solution = np.zeros(n, dtype=np.uint8)
    sum_flows = np.sum(flows, axis=1)
    sum_dists = np.sum(dists, axis=1)
    open_components = list(np.arange(n))
    open_slots = list(np.arange(n))
    for i in range(n):
        open_flows = sum_flows[open_components]
        component = open_components[np.random.choice(np.argwhere(open_flows == open_flows.max())[:, 0])]
        open_dists = sum_dists[open_slots]
        slot = open_slots[np.random.choice(np.argwhere(open_dists == open_dists.min())[:, 0])]
        solution[component] = slot
        open_components.remove(component)
        open_slots.remove(slot)

    return solution


def greedy_init2(n, flows, dists):
    """
    Init greedy.
    On utilise un critère qui synthétise les informations sur les flows et distance pour faire des assignation directes
    2 critères semblent intéressants:
        - minimiser le ratio somme des distances sortantes/ somme des flows sortants poru chaque couple slot/component
        - minimiser le produit
    Ici on minimise le ratio
    :param n: nombre de component/slots
    :param flows: matrice des fluxs
    :param dists: matrice des distances
    :return res: solution
    """
    solution = np.zeros(n, dtype=np.uint8)
    sum_flows = np.sum(flows, axis=1)
    sum_dists = np.sum(dists, axis=1)
    ratio = np.expand_dims(sum_flows, axis=1) @ np.expand_dims(sum_dists, axis=0)
    open_components = list(np.arange(n))
    open_slots = list(np.arange(n))
    for i in range(n):
        open_ratio = ratio[open_components, :][:, open_slots]
        best = np.argwhere(open_ratio == open_ratio.min())
        component_i, slot_i = best[rng.integers(len(best))]
        component = open_components[component_i]
        slot = open_slots[slot_i]
        solution[component] = slot
        open_components.remove(component)
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
    :param n: nombre de component/slots
    :param flows: matrice des fluxs
    :param dists: matrice des distances
    :return res: solution
    """
    solution = np.zeros(n, dtype=np.uint8)
    sum_flows = np.sum(flows, axis=1)
    sum_dists = np.sum(dists, axis=1)
    open_components = list(np.arange(n))
    open_slots = list(np.arange(n))
    assigned_components = []
    for i in range(n):
        assigned_slots = solution[assigned_components]
        open_flows = sum_flows[open_components]
        assigned_flows = np.sum(flows[open_components, :][:, assigned_components], axis=1)
        selected = np.argwhere(assigned_flows == assigned_flows.max())[:, 0]
        unassigned_flows = open_flows[selected] - assigned_flows[selected]
        subselected = np.array(open_components)[selected[np.argwhere(unassigned_flows == unassigned_flows.min())[:, 0]]]
        component = np.random.choice(subselected)

        assigned_costs = np.sum(flows[component, :][assigned_components] * dists[:, assigned_slots], axis=1)[open_slots]

        #Tie braek aléatoire
        #slot = open_slots[np.random.choice(np.argwhere(assigned_costs == assigned_costs.min())[:, 0])]

        #Tie Break par min distance unassigned
        open_dists = sum_dists[open_slots]
        assigned_dists = np.sum(dists[open_slots, :][:, assigned_slots], axis=1)
        selected = np.argwhere(assigned_costs == assigned_costs.min())[:, 0]
        unassigned_dists = open_dists[selected] - assigned_dists[selected]
        subselected = np.array(open_slots)[selected[np.argwhere(unassigned_dists == unassigned_dists.min())[:, 0]]]
        slot = np.random.choice(subselected)

        solution[component] = slot
        open_components.remove(component)
        open_slots.remove(slot)
        assigned_components.append(slot)

    return solution

def greedy_init4(n, flows, dists, random_degree=1):
    """
    Init greedy. On choisit le noeuds avec le plus de flow sortant et on lui attribue le slot qui minimise le coût par rapport aux slots déjà attribués
    On a 3 manières intéressantes d'ordonner les choix des noeuds:
        - random
        - max total flow, random tie break (total assigned flow tie break ?)
        - max assigned flow, max unassigned flow tie break
    L'ordonnancement et les tie break dépendent du degré d'aléatoire voulu. 1 est le plus performant en général sur 1000 générations.
    :param n: nombre de component/slots
    :param flows: matrice des fluxs
    :param dists: matrice des distances
    :param random_degree: degré d'aléatoire dans l'initialisation, à 0 on est pareil que greedy3, à 1 on random tie break sur le slot assigné, à 2 on random tie break sur la component choisie et à 3 on choisi la component aléatoirement
    :return res: solution
    """
    solution = np.zeros(n, dtype=np.uint8)
    sum_flows = np.sum(flows, axis=1)
    sum_dists = np.sum(dists, axis=1)
    open_components = list(np.arange(n))
    open_slots = list(np.arange(n))
    assigned_components = []
    for i in range(n):
        if random_degree < 2:
            assigned_slots = solution[assigned_components]
            open_flows = sum_flows[open_components]
            assigned_flows = np.sum(flows[open_components, :][:, assigned_components], axis=1)
            selected = np.argwhere(assigned_flows == assigned_flows.max())[:, 0]
            unassigned_flows = open_flows[selected] - assigned_flows[selected]
            subselected = np.array(open_components)[selected[np.argwhere(unassigned_flows == unassigned_flows.min())[:, 0]]]
            component = np.random.choice(subselected)
        elif random_degree == 3:
            assigned_slots = solution[assigned_components]
            assigned_flows = np.sum(flows[open_components, :][:, assigned_components], axis=1)
            selected = np.array(open_components)[np.argwhere(assigned_flows == assigned_flows.max())[:, 0]]
            component = np.random.choice(selected)
        else:
            component = np.random.choice(open_components)

        assigned_costs = np.sum(flows[component, :][assigned_components] * dists[:, assigned_slots], axis=1)[open_slots]

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

        solution[component] = slot
        open_components.remove(component)
        open_slots.remove(slot)
        assigned_components.append(slot)

    return solution


def greedy_statistical_init(n, flows, dists, n_samples):
    solution = np.zeros(n, dtype=np.uint8)
    open_components = np.array(n*[True])
    open_slots = np.array(n*[True])
    for _ in range(n):
        estimations = np.zeros((sum(open_components), sum(open_slots)))
        samples_counts = np.zeros((sum(open_components), sum(open_slots)))
        new_sol = solution.copy()
        index_open_components = np.arange(n)[open_components]
        index_open_slots = np.arange(n)[open_slots]
        reverse_index = np.cumsum(open_slots) - 1
        for i in range(sum(open_components)):
            for j in range(sum(open_slots)):
                new_sol[index_open_components[i]] = index_open_slots[j]
                open_components[index_open_components[i]] = False
                open_slots[index_open_slots[j]] = False
                new_sol[open_components] = rng.permutation(np.arange(n)[open_slots])
                cost = evaluation(solution, flows, dists)
                open_components[index_open_components[i]] = True
                open_slots[index_open_slots[j]] = True
                for i in range(sum(open_components)):
                    j = reverse_index[new_sol[index_open_components[i]]]
                    estimations[i, j] = estimations[i, j] * samples_counts[i, j] / (samples_counts[i, j] + 1) + cost / (samples_counts[i, j] + 1)
                    samples_counts[i, j] += 1
        for _ in range(n_samples):
            new_sol[open_components] = rng.permutation(np.arange(n)[open_slots])
            cost = evaluation(solution, flows, dists)
            for i in range(sum(open_components)):
                j = reverse_index[new_sol[index_open_components[i]]]
                estimations[i,j] = estimations[i,j] * samples_counts[i,j] / (samples_counts[i,j] + 1) + cost / (samples_counts[i,j] + 1)
                samples_counts[i,j] += 1
        best = np.argwhere(estimations == estimations.min())
        component_i, slot_i = best[rng.integers(len(best))]
        component, slot = np.arange(n)[open_components][component_i], np.arange(n)[open_slots][slot_i]
        solution[component] = slot
        open_components[component] = False
        open_slots[slot] = False
    return solution


def idof(n, flows, dists, index):
    """
    /!\ NON OPTIMISE /!\ -> calcul quasi n^4 -> pas intéressant
    :param n:
    :param flows:
    :param dists:
    :param index:
    :return:
    """
    solution = np.zeros(n, dtype=np.uint8)
    assigned_components = np.full(n, -1) # Pour chaque component le slot associé, sinon -1
    assigned_slots = np.full(n, -1) # Pour chaque slot la component associée, sinon -1
    for i in index[:len(index)-1]: # Les components
        z = np.zeros(n)
        tag = n * [False]
        zp = np.full((n,n), -1)
        for j in range(n): # Le slot à associer
            if assigned_slots[j] == -1:
                for p in range(n):
                    if assigned_components[p] != -1:
                        z[j] += flows[i,p] * dists[j,assigned_components[p]] + flows[p,i] * dists[assigned_components[p],j]
                for p in range(n):
                    for q in range(n):
                        if assigned_components[p] != -1 and assigned_components[q] != -1:
                            z[j] += flows[p,q] * dists[assigned_components[p], assigned_components[q]]
            else:
                i0 = assigned_slots[j]
                tag[j] = True
                for k in range(n):
                    if assigned_slots[k] == -1:
                        zp[j,k] += flows[i,i0] * dists[j,k] + flows[i0,i] * dists[k,j]
                        for p in range(n):
                            if assigned_components[p] != -1 and p != i0:
                                zp[j,k] += flows[i, p] * dists[j, assigned_components[p]] + flows[p, i] * dists[assigned_components[p], j] + flows[i0, p] * dists[k, assigned_components[p]] + flows[p, i0] * dists[assigned_components[p], k]
                        for p in range(n):
                            for q in range(n):
                                if (assigned_components[p] == -1 or p == i0) and (assigned_components[q] == -1 or q == i0):
                                    zp[j,k] += flows[p, q] * dists[assigned_components[p], assigned_components[q]]
                accept = zp[j,:][zp[j,:] > -1]
                if len(accept) == 0:
                    z[j] = -1
                else:
                    z[j] = min(accept)
        accept = z[z > -1]
        if len(accept) == 0:
            j = np.random.choice(n)
        else:
            j = np.random.choice(np.nonzero((z == z[z > -1].min()) & (z != -1))[0])
        if tag[j]:
            a = zp[j,:]
            k = np.random.choice(np.nonzero((a == a[a > -1].min()) & (a != -1))[0])
            solution[i] = j
            solution[assigned_slots[j]] = k
            assigned_components[i] = j
            assigned_components[assigned_slots[j]] = k
            assigned_slots[k] = assigned_slots[j]
            assigned_slots[j] = i
        else:
            solution[i] = j
            assigned_components[i] = j
            assigned_slots[j] = i
    i = index[-1]
    solution[i] = np.nonzero(assigned_slots == -1)[0][0]
    return solution