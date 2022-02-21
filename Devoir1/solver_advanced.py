#python main.py --agent=advanced --infile=instances/horaire_C_169_3328.txt
import solver_advanced_choice3_greedy as sd #Choix de solveur retenu

def solve(schedule):
    """
    Your solution of the problem
    :param schedule: object describing the input
    :return: a list of tuples of the form (c,t) where c is a course and t a time slot.
    """

    return sd.solve(schedule)