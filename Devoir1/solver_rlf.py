import numpy as np
import networkx as nx

def solve(schedule):
    """
    Your solution of the problem
    :param schedule: object describing the input
    :return: a list of tuples of the form (c,t) where c is a course and t a time slot.
    """

    constraints = nx.to_numpy_matrix(schedule.conflict_graph, dtype=np.uint8)

    n = len(constraints)
    solution = np.zeros(n, dtype=np.uint16)
    uncolored_graph = constraints.copy()
    uncolored_nodes = list(np.arange(n))
    k = 0
    while uncolored_nodes:
        initial = uncolored_nodes[np.argmax(np.sum(uncolored_graph[uncolored_nodes,:], axis=1))]
        color_nodes = [initial]
        open_nodes = uncolored_nodes.copy()
        open_graph = uncolored_graph.copy()
        adjacent_graph = np.zeros(constraints.shape, dtype=np.uint8)

        open_nodes.remove(initial)
        open_graph[initial, :] = np.full((1,n), 0)
        open_graph[:, initial] = np.full((n,1), 0)

        neighbours = np.nonzero(uncolored_graph[initial, :].transpose())[0]
        adjacent_graph[:, neighbours] = uncolored_graph[:, neighbours]

        for i in neighbours:
            open_nodes.remove(i)
        open_graph[neighbours, :] = np.full((len(neighbours), n), 0)
        open_graph[:, neighbours] = np.full((n, len(neighbours)), 0)

        while open_nodes:
            adjacent_counts = np.sum(adjacent_graph[open_nodes, :], axis=1)
            selected = np.array(open_nodes)[np.argwhere(adjacent_counts == adjacent_counts.max())[:,0]]
            open_counts = np.sum(open_graph[selected,:], axis=1)
            subselected = selected[np.argwhere(open_counts == open_counts.min())[:,0]]
            final = np.random.choice(subselected)
            color_nodes.append(final)

            neighbours = np.nonzero(open_graph[final, :].transpose())[0]
            adjacent_graph[:, neighbours] = open_graph[:, neighbours]

            for i in neighbours:
                open_nodes.remove(i)
            open_graph[neighbours, :] = np.full((len(neighbours), n), 0)
            open_graph[:, neighbours] = np.full((n, len(neighbours)), 0)

            open_nodes.remove(final)
            open_graph[final, :] = np.full((1,n), 0)
            open_graph[:, final] = np.full((n,1), 0)

        for i in color_nodes:
            uncolored_nodes.remove(i)

        uncolored_graph[color_nodes, :] = np.full((len(color_nodes),n), 0)
        uncolored_graph[:, color_nodes] = np.full((n,len(color_nodes)), 0)

        solution[color_nodes] = k

        k = k+1

    return dict(zip(schedule.course_list, solution.tolist()))