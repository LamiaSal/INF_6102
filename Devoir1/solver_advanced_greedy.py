
def solve(schedule):
    """
    Your solution of the problem
    :param schedule: object describing the input
    :return: a list of tuples of the form (c,t) where c is a course and t a time slot. 
    """

    # Add here your agent

    solution = dict()

    time_slot_idx = 1

    for c in schedule.course_list:

        assignation = time_slot_idx
        solution[c] = assignation
        time_slot_idx += 1


    raise Exception("Agent is not implemented")

def filter_solution(self, neighbors):
    """ filtrer toutes les solutions violant les contraintes
    """
    return [ n for n in neighbors if self._problem.feasable(n) ]
