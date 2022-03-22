import random as r
import time as t
import numpy as np
import networkx as nx
import solver_iterated_tabu
import solver_grasp as sg

rng = np.random.default_rng()

def solve(mother):
    """s
    Your solution of the problem
    :param mother: object describing the input
    :return: a list of integers of size n where the i-th element of the list is the component located in site i 
    """
    MAX_TIME = 1200
    
    t0 = t.time()
    n_restarts = 0

    s_star = generate_individual(mother.n_components)
    cost_star = f_eval(mother, s_star)
    while n_restarts < 1000 and t.time()-t0 < MAX_TIME:
        solution,cost_sol = genetic_search(mother,max_time=MAX_TIME)  

        # Update if amelioration
        if cost_sol < cost_star:
            cost_star = cost_sol
            s_star = solution
        n_restarts += 1
        print("RESTART numéro", n_restarts, "best cost:",cost_star)
    
    return s_star

def genetic_search(mother, max_time):
    # Initialisation
    N_COMPONENTS = mother.n_components
    N_TAILLE_POPULATION = N_COMPONENTS*20
    NB_GENERATION= 1000
    TOURNAMENT_SIZE = int(N_TAILLE_POPULATION*0.4)
    MUTATION_RATE = 0.5 # 0.5 with glouton initialisation
    CROSS_RATE = 0.95 #0.95
    PARENTS_RATE = 0.2 # 0.2 if we use glouton initialisation

    # On représente le graphe par ses matrices de flux et de distance
    flows = nx.to_numpy_array(mother.graph, dtype=np.int32, weight="flow")
    dists = nx.to_numpy_array(mother.graph, dtype=np.int32, weight="dist")
    

    #p_k = generate_population(N_COMPONENTS,N_TAILLE_POPULATION) # TODO: A compléter A tester GLOUTON !!!!
    #p_k = generate_population_mixte(N_COMPONENTS, flows, dists, N_TAILLE_POPULATION, mother)
    #p_k =generate_population_init2(N_COMPONENTS, flows, dists, N_TAILLE_POPULATION)
    #p_k= generate_population_greedy(N_COMPONENTS, flows, dists, N_TAILLE_POPULATION) # TODO: A compléter A tester GLOUTON !!!!
    #s_star= generate_solution(N_COMPONENTS) # TODO: A tester GLOUTON !!!!
    p_k = generate_population_grasp(N_COMPONENTS, flows, dists, N_TAILLE_POPULATION, mother, 60)


    
    p_k_eval=[]
    for s_k in p_k:
        #cost = f_eval(mother, s_k)
        cost = evaluation(np.array(s_k), flows, dists)
        p_k_eval.append(cost)
    
    #print(p_k_eval)
    index_min = np.argmin(np.array(p_k_eval))
    cost_star = p_k_eval[index_min]
    s_star = p_k[index_min]

    i=0
    #s_stars=[]
    #s_stars_eval=[]
    mutation_rate = MUTATION_RATE
    cross_rate = CROSS_RATE
    t0 = t.time()
    max_iter_without_improv=0
    sub_sample_size=0.6
    lambda_p=N_COMPONENTS//6
    restart=False

    #print(p_k)
    while (i<NB_GENERATION and t.time()-t0 < max_time and not restart):
        #print("p_k 11",p_k)
        #print("genereation:",i,"best cost",cost_star)
        # Selection (ranking ou tournoi)
        p_k_star= tournament(p_k, p_k_eval,mother, n_components=N_COMPONENTS,population_size = N_TAILLE_POPULATION, tournament_size=TOURNAMENT_SIZE)
        #p_k_star=roulette(p_k, p_k_eval,mother)
        '''
        if i==0:
            p_k_prev, p_k_eval_pred= p_k, p_k_eval
            p_k_star= tournament(p_k, p_k_eval,mother, n_components=N_COMPONENTS,population_size = N_TAILLE_POPULATION, tournament_size=TOURNAMENT_SIZE)
        elif i>=1:
            p_k_star= tournament(p_k, p_k_eval,p_k_prev, p_k_eval_pred, mother, n_components=N_COMPONENTS,population_size = N_TAILLE_POPULATION, tournament_size=TOURNAMENT_SIZE)
        '''
        #print("p_k_star",p_k_star)
        #print("len(p_k_star)",len(p_k_star))
        # Hybridation
        c_k= crossing(p_k_star,n_components = N_COMPONENTS, cross_rate =cross_rate, probablity=0.5) # TODO : numpy, 2-point crossover (THEO)

        #print("len(c_k)",len(c_k))
        #Mutation
        #TODO: paramter tuning : mut_rate adaptatif et lambda
        m_k= permut_mutation(c_k,N_COMPONENTS, mut_rate=mutation_rate,lambda_p=lambda_p,sub_sample_size=sub_sample_size,poisson=True) # TODO : Poisson/swap/permutation (LAMIA)

        #print("len(m_k)",len(m_k))
        #print("m_k",m_k)
        #print("p_k 22",p_k)

        # Updating
        m_k_eval=[]
        for s_k in m_k:
            #cost = f_eval(mother, s_k)
            cost = evaluation(np.array(s_k), flows, dists)
            m_k_eval.append(cost)
        #m_k_eval_argsort = np.argsort(m_k_eval)


        # generate new population
        p_k, p_k_eval = generate_new_population(m_k, m_k_eval, p_k, p_k_eval, N_COMPONENTS,N_TAILLE_POPULATION,flows,dists,parents_rate=PARENTS_RATE) # TODO: point d'amélioration
        
        index_min=np.argmin(np.array(p_k_eval))
        best_cost = p_k_eval[index_min]
        best_s_star = p_k[index_min]
        
        if cost_star > best_cost :
            cost_star= best_cost
            s_star = best_s_star
            # paramters for mutation (controls intensification and variety)
            #mutation_rate= r.uniform(0.2, 0.4)
            #sub_sample_size=r.uniform(0.1, 0.5)
            lambda_p=N_COMPONENTS//6
            
            #print(lambda_p)
            #lambda_p = 3
            max_iter_without_improv=0
        else:
            max_iter_without_improv+=1
        if max_iter_without_improv >=10:
            # parameters for mutation (controls intensification and variety)
            #mutation_rate= r.uniform(0.2, 0.6)
            lambda_p = int(N_COMPONENTS*r.uniform(0.1, 0.3))
            #sub_sample_size=r.uniform(0.7, 0.9)
        if max_iter_without_improv >=60:
            #mutation_rate=0.99
            lambda_p = int(N_COMPONENTS*r.uniform(0.1, 0.6))
            #TOURNAMENT_SIZE = int(N_TAILLE_POPULATION*0.5)
            #mutation_rate=0.03
            #cross_rate = 0.98
            #print(p_k)
        if max_iter_without_improv>=100:
            restart = True
        if N_TAILLE_POPULATION <=300:
            diviseur=NB_GENERATION
            mutation_rate= 1-(i/diviseur)
            cross_rate = (i/diviseur)
            #mutation_rate= r.uniform(0.1, 0.5)
            #cross_rate = r.uniform(0.8, 0.999)
        else:
            mutation_rate= (i/NB_GENERATION)
            cross_rate = 1 - (i/NB_GENERATION)
        
        '''print(mutation_rate)
        print(cross_rate)
        print(lambda_p)'''
        i+=1
    
    return s_star, cost_star


'''
def generate_solution(n_components):
    solution = list(range(n_components))
    random.shuffle(solution)
    return solution
'''
#####################################################################################################################################
##################################################### initialisation functions ######################################################
#####################################################################################################################################
def generate_individual(n_components):
    try:
        individual = r.sample(range(0, n_components), n_components)
    except ValueError:
        print('Sample size exceeded population size.')
    return individual

def greedy_init1(n, flows, dists):
    """
    Init greedy. On choisit le noeud avec le plus de flow sortant et on lui attribue le slot avec les plus petites distances sortantes
    On a 3 manières intéressantes d'ordonner les choix des noeuds:
        - random
		- max total flow, random tie break (total assigned flow tie break ?)
		- max assigned flow, max unassigned flow tie break
	Ici on utilise la deuxième
    :param n: nombre de machine/slots
    :param flows: matrice des fluxs
    :param dists: matrice des distances
    :return res: solution
    """
    solution = np.zeros(n, dtype=np.uint8)
    sum_flows = np.sum(flows, axis=1)
    sum_dists = np.sum(dists, axis=1)
    open_machines = list(np.arange(n))
    open_slots = list(np.arange(n))
    for i in range(n):
        open_flows = sum_flows[open_machines]
        machine = open_machines[np.random.choice(np.argwhere(open_flows == open_flows.max())[:, 0])]
        open_dists = sum_dists[open_slots]
        slot = open_slots[np.random.choice(np.argwhere(open_dists == open_dists.min())[:, 0])]
        solution[slot] = machine
        open_machines.remove(machine)
        open_slots.remove(slot)

    return solution

def greedy_init2(n, flows, dists):
    """
    Init greedy.
    On utilise un critère qui synthétise les informations sur les flows et distance pour faire des assignation directes
    2 critères semblent intéressants:
        - minimiser le ratio somme des distances sortantes/ somme des flows sortants poru chaque couple slot/machine
        - minimiser le produit
    Ici on minimise le ratio
    :param n: nombre de machine/slots
    :param flows: matrice des fluxs
    :param dists: matrice des distances
    :return res: solution
    """
    solution = np.zeros(n, dtype=np.uint8)
    sum_flows = np.sum(flows, axis=1)
    sum_dists = np.sum(dists, axis=1)
    ratio = np.expand_dims(sum_flows, axis=1) @ np.expand_dims(sum_dists, axis=0)
    open_machines = list(np.arange(n))
    open_slots = list(np.arange(n))
    for i in range(n):
        open_ratio = ratio[open_machines, open_slots]
        best = np.argwhere(open_ratio == open_ratio.min())
        machine, slot = best[r.randint(0,len(best))]
        solution[slot] = machine
        open_machines.remove(machine)
        open_slots.remove(slot)

    return solution

def greedy_init4(n, flows, dists, random_degree=1):
    """
    Init greedy. On choisit le noeuds avec le plus de flow sortant et on lui attribue le slot qui minimise le coût par rapport aux slots déjà attribués
    On a 3 manières intéressantes d'ordonner les choix des noeuds:
        - random
        - max total flow, random tie break (total assigned flow tie break ?)
        - max assigned flow, max unassigned flow tie break
    L'ordonnancement et les tie break dépendent du degré d'aléatoire voulu. 1 est le plus performant en général sur 1000 générations.
    :param n: nombre de machine/slots
    :param flows: matrice des fluxs
    :param dists: matrice des distances
    :param random_degree: degré d'aléatoire dans l'initialisation, à 0 on est pareil que greedy3, à 1 on random tie break sur le slot assigné, à 2 on random tie break sur la machine choisie et à 3 on choisi la machine aléatoirement
    :return res: solution
    """
    solution = np.zeros(n, dtype=np.uint8)
    sum_flows = np.sum(flows, axis=1)
    sum_dists = np.sum(dists, axis=1)
    open_machines = list(np.arange(n))
    open_slots = list(np.arange(n))
    assigned_slots = []
    for i in range(n):
        if random_degree < 2:
            assigned_machines = solution[assigned_slots]
            open_flows = sum_flows[open_machines]
            assigned_flows = np.sum(flows[open_machines, :][:, assigned_machines], axis=1)
            selected = np.argwhere(assigned_flows == assigned_flows.max())[:, 0]
            unassigned_flows = open_flows[selected] - assigned_flows[selected]
            subselected = np.array(open_machines)[selected[np.argwhere(unassigned_flows == unassigned_flows.max())[:, 0]]]
            machine = np.random.choice(subselected)
        elif random_degree == 3:
            assigned_machines = solution[assigned_slots]
            assigned_flows = np.sum(flows[open_machines, :][:, assigned_machines], axis=1)
            selected = np.array(open_machines)[np.argwhere(assigned_flows == assigned_flows.max())[:, 0]]
            machine = np.random.choice(selected)
        else:
            machine = np.random.choice(open_machines)

        assigned_costs = np.sum(flows[solution[machine], :][solution[assigned_slots]] * dists[:, assigned_slots], axis=1)[open_slots]

        #Tie braek aléatoire
        if random_degree < 2:
            slot = np.array(open_slots)[np.random.choice(np.argwhere(assigned_costs == assigned_costs.min())[:, 0])]
        #Tie Break par min distance unassigned
        elif random_degree == 0:
            open_dists = sum_dists[open_slots]
            assigned_dists = np.sum(dists[open_slots, :][:, assigned_slots], axis=1)
            selected = np.argwhere(assigned_costs == assigned_costs.min())[:, 0]
            unassigned_dists = open_dists[selected] - assigned_dists[selected]
            subselected = np.array(open_slots)[selected[np.argwhere(unassigned_dists == unassigned_dists.max())[:, 0]]]
            slot = np.random.choice(subselected)

        solution[slot] = machine
        open_machines.remove(machine)
        open_slots.remove(slot)
        assigned_slots.append(slot)

    return solution

def greedy_init3(n, flows, dists):
    """
    Init greedy. On choisit le noeuds avec le plus de flow sortant et on lui attribue le slot qui minimise le coût par rapport aux slots déjà attribués
    On a 3 manières intéressantes d'ordonner les choix des noeuds:
        - random
        - max total flow, random tie break (total assigned flow tie break ?)
        - max assigned flow, max unassigned flow tie break
    Ici on utilise la troisième
    :param n: nombre de machine/slots
    :param flows: matrice des fluxs
    :param dists: matrice des distances
    :return res: solution
    """
    solution = np.zeros(n, dtype=np.uint8)
    sum_flows = np.sum(flows, axis=1)
    sum_dists = np.sum(dists, axis=1)
    open_machines = list(np.arange(n))
    open_slots = list(np.arange(n))
    assigned_slots = []
    for i in range(n):
        assigned_machines = solution[assigned_slots]
        open_flows = sum_flows[open_machines]
        assigned_flows = np.sum(flows[open_machines, :][:, assigned_machines], axis=1)
        selected = np.argwhere(assigned_flows == assigned_flows.max())[:, 0]
        unassigned_flows = open_flows[selected] - assigned_flows[selected]
        subselected = np.array(open_machines)[selected[np.argwhere(unassigned_flows == unassigned_flows.max())[:, 0]]]
        machine = np.random.choice(subselected)

        assigned_costs = np.sum(flows[solution[machine], :][solution[assigned_slots]] * dists[:, assigned_slots], axis=1)[open_slots]

        #Tie braek aléatoire
        #slot = open_slots[np.random.choice(np.argwhere(assigned_costs == assigned_costs.min())[:, 0])]

        #Tie Break par min distance unassigned
        open_dists = sum_dists[open_slots]
        assigned_dists = np.sum(dists[open_slots, :][:, assigned_slots], axis=1)
        selected = np.argwhere(assigned_costs == assigned_costs.min())[:, 0]
        unassigned_dists = open_dists[selected] - assigned_dists[selected]
        subselected = np.array(open_slots)[selected[np.argwhere(unassigned_dists == unassigned_dists.max())[:, 0]]]
        slot = np.random.choice(subselected)

        solution[slot] = machine
        open_machines.remove(machine)
        open_slots.remove(slot)
        assigned_slots.append(slot)

    return solution

def generate_population(n_components, n_taille_population):
    population = [generate_individual(n_components) for _ in range(n_taille_population)]
    return population

def generate_population_greedy(n_components, flows, dists, n_taille_population):
    population = [greedy_init1(n_components, flows, dists) for _ in range(n_taille_population)]
    return population

def generate_population_mixte(n_components, flows, dists, n_taille_population,mother):
    population_1 = [greedy_init1(n_components, flows, dists) for _ in range(n_taille_population//2)]
    population_2 = [generate_individual(n_components) for _ in range(n_taille_population//2)]
    #print(n_taille_population//40)
    #population_3 = [solver_tabu.solve(mother) for _ in range(n_taille_population//40)]
    #population_3 = [greedy_init3(n_components, flows, dists) for _ in range(n_taille_population//2)]
    #population_4 = [greedy_init4(n_components, flows, dists, random_degree=1) for _ in range(n_taille_population//4)]
    population = population_1 + population_2 #population_2 + population_1 + population_3 + population_4
    r.shuffle(population)
    return population

def generate_population_grasp(n_components, flows, dists, n_taille_population,mother,time_init):
    population = np.zeros((n_taille_population, n_components), dtype=np.int32)
    n = n_taille_population//4
    pop1 = sg.init(mother, n, time_init)
    n = len(pop1)
    population[:n] = pop1
    for i in range(n, n_taille_population):
        population[i] = rng.permutation(n_components)
    r.shuffle(population)
    return population


#####################################################################################################################################
##################################################### population selection ##########################################################
#####################################################################################################################################
def roulette(population, population_eval,mother):

    assert len(population) % 2 == 0, "population size should be an even number"
    
    roulette_list=population.copy()
    roul_population_eval = population_eval.copy()
    selected=[]
    for i in range(len(population)):
    #while len(roulette_list)!=0:

        # cumulative probabilities of the solution 
        sol_probas = []
        for j in range(len(roulette_list)):
            Pr_j=roul_population_eval[j]/sum(roul_population_eval)
            sol_probas.append(Pr_j)
        
        # random jet
        ceil = r.random()
        k = 0
        while sol_probas[i] < ceil:
            k += 1
        
        # add selected solution
        selected.append(roulette_list.pop(k))
        roul_population_eval.pop(k)

    return selected

def tournament(population, population_eval, mother,n_components,population_size, tournament_size):

    # verification of some connditions
    if tournament_size%2!=0:
        tournament_size+=1
    assert population_size == len(population), "population size should be equal to len(population)"
    assert population_size % 2 == 0, "population size should be an even number"
    assert population_size >= tournament_size, "population size should be higher than the tournament size"
    
    selected=[]
    size_subsets = population_size//tournament_size
    population_eval_arr=np.array(population_eval)

    for i in range(population_size):
        select_index = r.sample(range(0, len(population)), tournament_size)
        index_min_sol = np.argmin(population_eval_arr[select_index])
        selected.append(list((np.array(population)[select_index])[index_min_sol]))
    return selected

#####################################################################################################################################
##################################################### Hybridation functions #########################################################
#####################################################################################################################################
def crossing(population,n_components, cross_rate=0.95, probablity=0.5):
    
    assert len(population)%2 == 0, "Population de taille paire requise"
    #r.shuffle(population)

    zeros_arr = np.full((n_components),-1, dtype=int)
    children = []
    for i in range(len(population)//2):
        # Initialisation of parents
        p1 = np.array(population[2*i])
        p2 = np.array(population[2*i+1])
        
        Pr_mutation = r.random() #probability to mutate
        
        if Pr_mutation < cross_rate:

            # Creation of the mask for the Uniform crossing vector
            mask = np.random.binomial(n=1, p=0.5, size=[len(p1)])
            mask_bool = (mask == 0)

            # Initialisation of children and saving th position of the values that will be changed
            c1 = np.where(mask_bool, p1, zeros_arr)
            c2 = np.where(mask_bool, p2, zeros_arr)
            index_1 = np.argwhere(c1==-1)
            index_2 = np.argwhere(c2==-1)

            # Completing the children using the same order of apparition of the sites than in the other parent
            for j in range(len(p1)):
                if p1[j] not in c2:
                    c2[index_2[0]] = p1[j]
                    index_2 = np.delete(index_2,0)
                if p2[j] not in c1:
                    c1[index_1[0]] = p2[j]
                    index_1 = np.delete(index_1,0)

            children.append(list(c1))
            children.append(list(c2))
        else:
            children.append(list(p1))
            children.append(list(p2))
    
    return children

#####################################################################################################################################
######################################################## Mutation functions #########################################################
#####################################################################################################################################

# augmenter taux de pumtation = diversification, si y a des clones il manque de diversification
def permut_mutation(c_k,n_components, mut_rate=0.3,lambda_p=1,sub_sample_size=0.6,poisson=True):
    if poisson and lambda_p==1 :
        mut_rate=1/n_components
    

    m_k=[]
    for sol in c_k:
        if poisson:
            
            nb_permut = min(n_components,max(2,np.random.poisson(lambda_p)))
        else:
            nb_permut =  min(n_components,max(2,int(n_components*sub_sample_size)))
            #print(nb_permut)
        #print(nb_permut)
        Pr_mutation = r.random() #probability to mutate

        if Pr_mutation < mut_rate:
            # permutations of a subsample of the solution
            sub_sample = r.sample(list(range(n_components)),nb_permut)
            save_first = sol[sub_sample[0]]
            for i in range(len(sub_sample)-1):
                sol[sub_sample[i]] = sol[sub_sample[i+1]]
            sol[sub_sample[-1]]=save_first
        m_k.append(sol)
    return m_k


#####################################################################################################################################
####################################################### Evaluation function #########################################################
#####################################################################################################################################
def f_eval(mother, solution):
    return mother.get_total_cost(solution)

def evaluation(solution, flows, dists):
    return np.sum(flows * dists[solution, :][:, solution])


#####################################################################################################################################
########################################### New population generation functions #####################################################
#####################################################################################################################################
def generate_new_population(m_k, m_k_eval, p_k, p_k_eval, n_components,n_taille_population, flows, dists,parents_rate=0.5):
    # gardez 50% parents et 50% enfants
    #print(m_k)
    #print(p_k)
    nb_parents_kept = int(n_taille_population*(parents_rate))
    p_k_best_eval_index = (np.argsort(p_k_eval))[:nb_parents_kept]
    p_k_best = np.array(p_k)[p_k_best_eval_index]
    p_k_best_eval = np.array(p_k_eval)[p_k_best_eval_index]
    #print("np.min(p_k)",np.min(p_k_best_eval))
    m_k_best_eval_index = (np.argsort(m_k_eval))[:len(p_k)-nb_parents_kept]
    m_k_best = np.array(m_k)[m_k_best_eval_index]
    m_k_best_eval = np.array(m_k_eval)[m_k_best_eval_index]
    #print("np.min(m_k)",np.min(m_k_best_eval))
    
    new_p_k = list(p_k_best) + list(m_k_best)
    new_p_k_eval = list(p_k_best_eval) + list(m_k_best_eval)

    c = list(zip(new_p_k, new_p_k_eval))
    #print(c)
    #import itertools
    #c.sort()
    #new_p_k, new_p_k_eval = zip(*c)

    
    new_c = []
    for elem in c:
        if list(elem) not in c:
            new_c.append(elem)
    r.shuffle(new_c)
    new_p_k, new_p_k_eval = zip(*new_c)

    for i in range(n_taille_population-len(new_p_k)):
        new_p_k.append(generate_individual(n_components))
        #new_p_k_eval.append(f_eval(mother, new_p_k[-1]))
        new_p_k_eval.append(evaluation(np.array(new_p_k[-1]), flows, dists))
    
    
    return new_p_k, new_p_k_eval