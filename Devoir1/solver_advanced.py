# Choix de solveur retenu : solver_advanced_choice3_greedy.py
import solver_advanced_choice3_greedy as sd


def solve(schedule):
    """
    Your solution of the problem
    :param schedule: object describing the input
    :return: a list of tuples of the form (c,t) where c is a course and t a time slot.
    """

    # Appel du solveur dans un fichier externe
    return sd.solve(schedule)


# Instances de test :

# python main.py --agent=advanced --infile=instances/horaire_A_11_20.txt
# python main.py --agent=advanced --infile=instances/horaire_B_23_71.txt
# python main.py --agent=advanced --infile=instances/horaire_C_169_3328.txt
# python main.py --agent=advanced --infile=instances/horaire_D_184_3916.txt
# python main.py --agent=advanced --infile=instances/horaire_E_645_13979.txt
