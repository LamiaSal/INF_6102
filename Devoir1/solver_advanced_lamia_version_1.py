import time
import random
from collections import defaultdict 

def solve(schedule):
    """
    Your solution of the problem
    :param schedule: object describing the input
    :return: a list of tuples of the form (c,t) where c is a course and t a time slot. 
    """
    begin_time = time.time()
    current_time = begin_time 


    nb_courses =len(schedule.course_list)

    # TODO : piste d'amélioration
    #solution = generateInitialSolution(schedule)
    solution = generateInitialSolutionNaive(schedule)
    solution_star = solution.copy()


    # evaluation with choice 2 in course so we use simutanously kambe chaine
    evaluation_result_star, nb_creneaux_star,nb_conflicts_star = naive_evaluation(schedule,solution)
    #evaluation_result_star, nb_creneaux_star,nb_conflicts_star = advanced_evaluation_no_conflicts(schedule,solution)

    # evaluation with choice 3 in course but we should not use kambe chaine and hard constraint
    #evaluation_result_star, nb_creneaux_star,nb_conflicts_star = advanced_evaluation_with_conflicts(schedule,solution)
    #evaluation_result_star, nb_creneaux_star,nb_conflicts_star = advanced_evaluation_valid_and_notValid_sol(schedule,solution)

    
    while current_time - begin_time <= 120 :

        # evaluating the solution beforehand
        evaluation_result, nb_creneaux,nb_conflicts= naive_evaluation(schedule,solution)
        #evaluation_result,nb_creneaux,nb_conflicts = advanced_evaluation_no_conflicts(schedule,solution)
        #evaluation_result,nb_creneaux,nb_conflicts = advanced_evaluation_with_conflicts(schedule,solution)
        #evaluation_result,nb_creneaux,nb_conflicts = advanced_evaluation_valid_and_notValid_sol(schedule,solution)
        
        # we take the course that has the most conflicts
        # TODO : piste d'amélioration pour la séléction
        
        '''
        #TODO: ne pas considérer le cours qui a été considéré précédemment
        conflict_list_sum=defaultdict(int) 
        for (k,l) in list(schedule.conflict_list):
            conflict_list_sum[k] +=1
            conflict_list_sum[l] +=1
        
        selected_course = max(conflict_list_sum, key=lambda key: conflict_list_sum[key])
        print(selected_course)
        '''
        
        # we take a course randomly
        selected_course = random.sample(schedule.course_list.items(), 1)[0][0] #TODO

        
        for i in range(nb_courses):

            

            # fonction de voisinage
            #TODO: amélioorer avec la chaine de Kempe
            solution_neigbour_bis = naive_neighbourhood_fonction(solution,selected_course,i)
            #solution_neigbour_bis = kambe_neighbourhood_fonction(solution,selected_course,i)
        
            # evaluate the nieghbour solution
            evaluation_result_n,nb_creneaux_n,nb_conflicts_n = naive_evaluation(schedule,solution_neigbour_bis)
            #evaluation_result_n,nb_creneaux_n,nb_conflicts_n = advanced_evaluation_no_conflicts(schedule,solution_neigbour_bis)
            #evaluation_result_n,nb_creneaux_n,nb_conflicts_n = advanced_evaluation_with_conflicts(schedule,solution_neigbour_bis)
            #evaluation_result_n,nb_creneaux_n,nb_conflicts_n = advanced_evaluation_valid_and_notValid_sol(schedule,solution_neigbour_bis)


            if evaluation_result_n <= evaluation_result and nb_conflicts_n==0: 
                solution = solution_neigbour_bis.copy()
                evaluation_result = evaluation_result_n
                nb_conflicts = nb_conflicts_n
                nb_creneaux = nb_creneaux_n
        
        if evaluation_result < evaluation_result_star and nb_conflicts==0: 
            solution_star = solution.copy()
        
        
        current_time=time.time()
    
    return solution_star


def generateInitialSolution(schedule):

    nb_courses=len(schedule.course_list)
    solution = dict()

    for c in schedule.course_list:

        assignation = random.randint(0,nb_courses//2)
        solution[c] = assignation
    
    return solution

def generateInitialSolutionNaive(schedule):
    solution = dict()

    time_slot_idx = 1

    for c in schedule.course_list:

        assignation = time_slot_idx
        solution[c] = assignation
        time_slot_idx += 1
        
    return solution

def naive_evaluation(schedule,solution):
    nb_creneaux = schedule.get_n_creneaux(solution)
    nb_conflicts = sum(solution[a[0]] == solution[a[1]] for a in schedule.conflict_list)
    evaluation_result = nb_creneaux + nb_conflicts
    return evaluation_result,nb_creneaux,nb_conflicts

def advanced_evaluation_no_conflicts(schedule, solution):
    # TODO : minimiser les conflits (cf formule du cours - maximiser les groupes de cours à numéros de créneau unique
    nb_creneaux = schedule.get_n_creneaux(solution)
    nb_conflicts = sum(solution[a[0]] == solution[a[1]] for a in schedule.conflict_list)
    
    # compute evaluation result
    slots_list_sum = defaultdict(int) 
    for key, val in solution.items():
        slots_list_sum[val] += 1 

    evaluation_result= 0

    for val in slots_list_sum.values():
        evaluation_result += val**2

    return -evaluation_result, nb_creneaux,nb_conflicts

def advanced_evaluation_with_conflicts(schedule, solution):
    # TODO : minimiser les conflits (cf formule du cours - maximiser les groupes de cours à numéros de créneau unique
    nb_creneaux = schedule.get_n_creneaux(solution)
    nb_conflicts = sum(solution[a[0]] == solution[a[1]] for a in schedule.conflict_list)
    
    # compute evaluation result
    # sum of the squared of the number of courses by slot
    slots_list_sum = defaultdict(int) 
    for key, val in solution.items():
        slots_list_sum[val] += 1 

    evaluation_result_slot= 0
    for val in slots_list_sum.values():
        evaluation_result_slot += val**2

    # sum of the squared of edges connected to each courses 
    conflict_list_sum = defaultdict(int) 
    for a in schedule.conflict_list:
        if solution[a[0]] == solution[a[1]]:
            conflict_list_sum[a[0]] +=1
            conflict_list_sum[a[1]] +=1
    
    evaluation_result_conflicts= 0
    for val in conflict_list_sum.values():
        evaluation_result_conflicts += val**2

    return evaluation_result_conflicts - evaluation_result_slot, nb_creneaux,nb_conflicts

def advanced_evaluation_valid_and_notValid_sol(schedule, solution):
    # TODO : minimiser les conflits (cf formule du cours - maximiser les groupes de cours à numéros de créneau unique
    nb_creneaux = schedule.get_n_creneaux(solution)
    nb_conflicts = sum(solution[a[0]] == solution[a[1]] for a in schedule.conflict_list)
    
    # compute evaluation result

    # sum of the squared of the number of courses by slot
    slots_list_sum = defaultdict(int) 
    for key, val in solution.items():
        slots_list_sum[val] += 1 

    evaluation_result_slot= 0
    for val in slots_list_sum.values():
        evaluation_result_slot += val**2
    
    
    # sum of the squared of edges connected to each courses 
    conflict_list_sum = defaultdict(int) 
    for a in schedule.conflict_list:
        if solution[a[0]] == solution[a[1]]:
            conflict_list_sum[a[0]] +=1
            conflict_list_sum[a[1]] +=1
    
    evaluation_result_product= 0
    for val in list(schedule.course_list):
        evaluation_result_product += conflict_list_sum[val]*slots_list_sum[val]

    return evaluation_result_product - evaluation_result_slot, nb_creneaux,nb_conflicts

'''
def kambe_neighbourhood_fonction(solution_neigbour_bis):
    # TODO : amélioorer avec la chaine de Kempe
'''
def naive_neighbourhood_fonction(solution,selected_course,new_slot_value):
    # we change the slot of the considered course
    solution_neigbour_bis = solution.copy()
    solution_neigbour_bis[selected_course] = new_slot_value
    return solution_neigbour_bis
