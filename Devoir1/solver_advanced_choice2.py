import time
import random
from collections import defaultdict 
import numpy as np
import networkx as nx

def solve(schedule):
    """
    Your solution of the problem
    :param schedule: object describing the input
    :return: a list of tuples of the form (c,t) where c is a course and t a time slot. 
    """
    begin_time = time.time()
    current_time = begin_time 
    nb_courses =len(schedule.course_list)
    
    # initialisation
    solution_start = generateInitialSolution(schedule)
    # solution saved after on restart
    solution_star = solution_start.copy()
    # solution saved overall
    finale_solution = solution_star.copy()
    
    evaluation_result_star, nb_creneaux_star,nb_conflicts_star = advanced_evaluation_no_conflicts(schedule,solution_star)
    
    solution =solution_start.copy()
    prev_courses_seen=[]
    restart = False
    minima_bool = False
    # nb of iteration where the naive local search didn't improve
    nb_iteration_naive_neighbour_not_improved=0
    # nb of iteration where the local search with kempe didn't improve
    nb_not_improved_result=0

    evaluation_finale_solution=evaluation_result_star
    nb_creneaux_finale = nb_creneaux_star
    restart_at_least_once=False
    while current_time - begin_time <= 1200 :

        
        if restart :
            restart = False
            restart_at_least_once=True
            solution_start=generateInitialSolution(schedule)
            solution = solution_start.copy()
            solution_star = solution_start.copy()
            evaluation_result_star, nb_creneaux_star,nb_conflicts_star = advanced_evaluation_no_conflicts(schedule,solution_star)
            minima_bool = False
            nb_iteration_naive_neighbour_not_improved=0
            nb_not_improved_result=0
            prev_courses_seen=[]
            print('current best:',nb_creneaux_finale)
        
            
        # evaluating the solution beforehand
        evaluation_result,nb_creneaux,nb_conflicts = advanced_evaluation_no_conflicts(schedule,solution)
        
                    
        # Selecting a node
        if not minima_bool:
            selected_course, lenghth_minimas = node_selection_advanced(schedule, solution)
            prev_selected_course=[selected_course]
            while selected_course in prev_courses_seen:
                selected_course, lenghth_minimas = node_selection_advanced(schedule, solution)
                
                # if we're in a minima with the kempe chain, we chose randomly a node and 
                # put minima_bool to True to switch of neighborhood functionn
                if len(prev_selected_course)==lenghth_minimas:
                    selected_course = node_selection_naive(schedule)
                    minima_bool = True
                    break
                if not selected_course in prev_selected_course:
                    prev_selected_course.append(selected_course)
        else:
            selected_course = node_selection_naive(schedule)
        
        
        
        
        # Update the list of seen courses
        if len(prev_courses_seen)>=10:
            del prev_courses_seen[0]
            prev_courses_seen.append(selected_course)
        else  :
            prev_courses_seen.append(selected_course)
        
        #selected_course = node_selection_naive(schedule)
        
        #list of the values that we consider for selected_course (which are the current slots present in the solution)
        list_creneau_selec_c = set( val for val in solution.values() if val!=solution[selected_course])
        for slot_candidate in list_creneau_selec_c:

            # fonction de voisinage
            if not minima_bool:
                solution_neigbour_bis = kempe_neighbourhood_fonction(schedule,solution,selected_course,slot_candidate,nb_courses)
            else:
                solution_neigbour_bis = naive_neighbourhood_fonction(solution,selected_course,slot_candidate)

        
            # evaluate the neighbour solution
            evaluation_result_n,nb_creneaux_n,nb_conflicts_n = advanced_evaluation_no_conflicts(schedule,solution_neigbour_bis)


            if evaluation_result_n < evaluation_result and nb_conflicts_n==0: 
                solution = solution_neigbour_bis.copy()
                evaluation_result = evaluation_result_n
                nb_conflicts = nb_conflicts_n
                nb_creneaux = nb_creneaux_n
                
        
        if (evaluation_result < evaluation_result_star and nb_conflicts==0) or (evaluation_result == evaluation_result_star and nb_creneaux < nb_creneaux_star and nb_conflicts==0): 
            solution_star = solution.copy()
            evaluation_result_star=evaluation_result
            nb_conflicts_star = nb_conflicts
            nb_creneaux_star = nb_creneaux
            if minima_bool:
                nb_iteration_naive_neighbour_not_improved=0
            else:
                nb_not_improved_result=0

        elif evaluation_result == evaluation_result_star and nb_creneaux == nb_creneaux_star and nb_conflicts==0:
            rdn_nb = random.randint(0, 2)
            if rdn_nb%2==0:
                solution_star = solution.copy()
                nb_creneaux_star = nb_creneaux
            if minima_bool:
                nb_iteration_naive_neighbour_not_improved+=1
            else:
                nb_not_improved_result+=1
        else:
            if minima_bool:
                nb_iteration_naive_neighbour_not_improved+=1
            else:
                nb_not_improved_result+=1
        
        if nb_not_improved_result >=100:
            minima_bool=True
            
        if nb_iteration_naive_neighbour_not_improved>=100:
            if nb_creneaux_star<nb_creneaux_finale:
                finale_solution = solution_star.copy()
                evaluation_finale_solution=evaluation_result_star
                nb_creneaux_finale = nb_creneaux_star
            
            restart=True
        
        current_time=time.time()
    
    if not restart_at_least_once:
        finale_solution = solution_star.copy()
        evaluation_finale_solution=evaluation_result_star
        nb_creneaux_finale = nb_creneaux_star
    
    return finale_solution

def node_selection_naive(schedule):
    # pick a node randomly
    selected_course = random.choice(list(schedule.course_list))
    return selected_course

def node_selection_advanced(schedule, solution):
    '''
    for kempe chain:
    we find the slot that has the most occurence and chose randomly a node that is assigned to this slot
    '''

    # dictionnaire of occurences per slot
    dict_slot_occurence=defaultdict(int)
    for val in solution.values():
        dict_slot_occurence[val]+=1
    
    min_nb_slot =  min(dict_slot_occurence.values())
    min_slot = random.choice([ key for (key, value) in dict_slot_occurence.items() if value == min_nb_slot])
    
    # list of nodes corresponding to the least occured slot
    list_min_node = [key for (key, value) in solution.items() if value == min_slot] 
    
    chosen_node = random.choice(list_min_node)
    return chosen_node , len(list_min_node)


def generateInitialSolution(schedule):
    '''
    we beginn with a certain proportion of slots
    then we pick a random node and assign it to one of those color.
    If there is a conflict we chose another color in the the given slots.
    If no slots gives 0 conflict we assign it to a new slot and add this 
    slot to our considered list of slots.
    '''
    nb_courses=len(schedule.course_list)
    solution = dict()
    nb_considered_slots=(nb_courses)//5
    seen_courses=[]
    not_seen_course=list(schedule.course_list)
    while len(not_seen_course)>0:
        index_course = random.randint(0,len(not_seen_course)-1)

        c=not_seen_course[index_course]
        slot=0
        done=False
        seen_slot=[]
        not_seen_slot=list(range(0,nb_considered_slots))
        while len(not_seen_slot)>0 and not done:
            index_slot = random.randint(0,len(not_seen_slot)-1)
            slot=not_seen_slot[index_slot]
            done=True
            c_neigbors = schedule.conflict_graph.adj[c]
            
            for c_neigbor in c_neigbors:
                if c_neigbor in seen_courses and solution[c_neigbor]==slot:
                    done=False
                    break
            if done:
                solution[c] = slot
            seen_slot.append(slot)
            del not_seen_slot[index_slot]
        if not done :
            nb_considered_slots+=1
            solution[c] = nb_considered_slots
        seen_courses.append(c)
        del not_seen_course[index_course]
    
    return solution



def advanced_evaluation_no_conflicts(schedule, solution):
    # evaluation function to minise the number of cnflicts
    #  (cf formule on the report ) 
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


def naive_neighbourhood_fonction(solution,selected_course,new_slot_value):
    # we change the slot of the considered course
    solution_neigbour_bis = solution.copy()
    solution_neigbour_bis[selected_course] = new_slot_value
    return solution_neigbour_bis


def kempe_neighbourhood_fonction(schedule,solution,selected_course,slot_candidate,nb_courses):
    # kempe chaine

    solution_neigbour_bis = solution.copy()

    # STEP 0 : save the original slot of the selected course
    slot_original = solution_neigbour_bis[selected_course]
    
    # STEP 1 : we change the slot of the selected course with the slot candidate
    solution_neigbour_bis[selected_course] = slot_candidate

    # STEP 2 : consider every conflicts entailed by the changement
    neigbours = list(schedule.conflict_graph.neighbors(selected_course))
    direct_conflicts_list = []
    for neighbour in neigbours:
        if solution_neigbour_bis[neighbour] == slot_candidate:
            direct_conflicts_list.append(neighbour)
    
    # STEP 3, 4, 5 ... N : We define all the chaines alternating the slot_candidate and the slot_original
    chaine_global=[direct_conflicts_list]
    seen_neighbours = [selected_course] + direct_conflicts_list
    if direct_conflicts_list != []:
        chaine_global.append(direct_conflicts_list)
        i=0
        state=True
        while state and (i <= nb_courses - 2):
            chaine_rank_k = []
            if i%2==0:
                for neighbour in direct_conflicts_list:
                    if not (neighbour in seen_neighbours):
                        if solution_neigbour_bis[neighbour] == slot_original:
                            seen_neighbours.append(neighbour)
                            neigbours = list(schedule.conflict_graph.neighbors(neighbour))
                            if neigbours != []:
                                chaine_rank_k.append(neigbours)
            
            else:
                for neighbour in direct_conflicts_list:
                    if not (neighbour in seen_neighbours):
                        if solution_neigbour_bis[neighbour] == slot_candidate:
                            seen_neighbours.append(neighbour)
                            neigbours = list(schedule.conflict_graph.neighbors(neighbour))
                            if neigbours != []:
                                chaine_rank_k.append(neigbours)
            if chaine_rank_k == []:
                state=False
            else:
                chaine_global.append(chaine_rank_k)
                direct_conflicts_list = chaine_rank_k
            i+=1
    
    # once the whole path found we alternate the tw colours along the path
    j=0
    for list_i in chaine_global:
        for node in list_i:
            if j%2 == 0:
                solution_neigbour_bis[node]=slot_original
            else:
                solution_neigbour_bis[node]=slot_candidate
        j+=1
    
    return solution_neigbour_bis
