
import time
import random

def solve(schedule):
    """
    Your solution of the problem
    :param schedule: object describing the input
    :return: a list of tuples of the form (c,t) where c is a course and t a time slot. 
    """
    print(list(schedule.conflict_list)[0])
    begin_time = time.time()
    current_time = begin_time 
    # Add here your agent

    nb_courses =len(schedule.course_list)

    # TODO : piste d'amélioration
    #solution = generateInitialSolution(schedule)
    solution = generateInitialSolutionNaive(schedule)

    
    solution_star = solution.copy()
    

    nb_creneaux_star = schedule.get_n_creneaux(solution)
    nb_conflicts_star = sum(solution_star[a[0]] == solution_star[a[1]] for a in schedule.conflict_list)
    evaluation_result_star = nb_creneaux_star + nb_conflicts_star


    
    while current_time - begin_time <= 100 :
        nb_creneaux = schedule.get_n_creneaux(solution)
        nb_conflicts = sum(solution[a[0]] == solution[a[1]] for a in schedule.conflict_list)
        evaluation_result = nb_creneaux + nb_conflicts

    
        # we take the course that has the most conflicts
        # TODO : piste d'amélioration pour la séléction
        '''
        conflict_list=dict()
        for (k,l) in list(schedule.conflict_list):
            if k in conflict_list.keys() :
                conflict_list[k] +=1
            else:
                conflict_list[k] =1

            if l in conflict_list.keys() :
                conflict_list[l] +=1
            else:
                conflict_list[l] =1
        
        selected_course = max(conflict_list, key=lambda key: conflict_list[key])
        '''
        # we take a course randomly
        selected_course = random.sample(schedule.course_list.items(), 1)[0][0] #TODO

        
        for i in range(nb_courses):

            # we change the slot of the considered course
            solution_neigbour_bis = solution.copy()

            # fonction de voisinage
            #TODO: amélioorer avec la chiaane de Kempe
            solution_neigbour_bis[selected_course] = i

            
        
            # A MODIFIER
            nb_creneaux_ = schedule.get_n_creneaux(solution_neigbour_bis)
            nb_conflicts_ = sum(solution_neigbour_bis[a[0]] == solution_neigbour_bis[a[1]] for a in schedule.conflict_list)
            evaluation_result_ = nb_creneaux_ + nb_conflicts_


            if evaluation_result_ <= evaluation_result and nb_conflicts_==0: 
                solution = solution_neigbour_bis.copy()
                evaluation_result = evaluation_result_
                nb_conflicts = nb_conflicts_
                nb_creneaux = nb_creneaux_
        
        if evaluation_result < evaluation_result_star and nb_conflicts==0: 
            solution_star = solution.copy()
        
        
        current_time=time.time()
        
        
    
    return solution_star
    #raise Exception("Agent is not implemented")


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
'''
def fonction_evaluation():
    # TODO : minimiser les conflits (cf formule du cours - maximiser les groupes de cours à numéros de créneau unique

def fonction_neighbourhood():
    # TODO : amélioorer avec la chaine de Kempe
'''