import numpy as np
import networkx as nx

def solve(schedule):
    """
    Your solution of the problem
    :param schedule: object describing the input
    :return: a list of tuples of the form (c,t) where c is a course and t a time slot.
    """

    constraints = nx.to_numpy_array(schedule.conflict_graph, dtype=np.uint8)

    n = len(constraints)
    solution = np.zeros(n, dtype=np.uint16)
    uncolored_graph = constraints.copy()
    uncolored_nodes = list(np.arange(n))
    colors = np.full((n,n+1),n)
    saturation = np.zeros(n, dtype=np.uint16)

    while uncolored_nodes:
        selected = np.array(uncolored_nodes)[np.argwhere(saturation[uncolored_nodes] == saturation[uncolored_nodes].max())[:, 0]]
        uncolored_counts = np.sum(uncolored_graph[selected, :], axis=1)
        subselected = selected[np.argwhere(uncolored_counts == uncolored_counts.max())[:, 0]]
        final = np.random.choice(subselected)
        uncolored_nodes.remove(final)

        k = 0
        while k in colors[final, :]:
            k+=1
        solution[final] = k

        uncolored_graph[final, :] = np.full(1, 0)
        uncolored_graph[:, final] = np.full(n, 0)

        neighbours = np.nonzero(np.squeeze(np.asarray(constraints[final, :])))[0]
        colors[neighbours, final] = np.full(len(neighbours), k)

        saturation = np.zeros(n, dtype=np.uint16)
        for i in range(n):
            saturation[i] = len(np.unique(colors[i, :])) - 1

    return dict(zip(schedule.course_list, solution.tolist()))