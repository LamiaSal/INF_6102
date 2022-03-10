import random

def solve(mother):
    """
    Random resolution of the problem
    :param mother: object describing the input
    :return: a list of integers of size n where the i-th element of the list is the component located in site i 
    """
    solution = list(range(mother.n_components))
    random.shuffle(solution)
    return solution