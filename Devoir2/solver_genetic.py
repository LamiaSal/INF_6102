import random as r
import time as t
import numpy as np
import networkx as nx
import solver_iterated_tabu
import solver_grasp as sg

rng = np.random.default_rng()

def solve(mother):
    """
    Genetic Search with restart
    :param mother: object describing the input
    :return: a list of integers of size n where the i-th element of the list is the component located in site i 
    """

    # time maxed to 20 minutes
    MAX_TIME = 1200

    # initialisation of the population, either "random" or "grasp"
    INIT_POP = "grasp"
    
    t0 = t.time()
    n_restarts = 0

    # initialisation
    s_star = generate_individual(mother.n_components)
    cost_star = f_eval(mother, s_star)

    while n_restarts < 1000 and t.time()-t0 < MAX_TIME:
        # genetic search
        solution,cost_sol = genetic_search(mother,init_pop=INIT_POP,max_time=MAX_TIME-(t.time()-t0)) 
        print(MAX_TIME-(t.time()-t0)) 

        # Update if amelioration
        if cost_sol < cost_star:
            cost_star = cost_sol
            s_star = solution
        n_restarts += 1
        print("RESTART numéro", n_restarts, "best cost:",cost_star)
    
    return s_star

def genetic_search(mother, max_time,init_pop):
    """
    Genetic Search
    :param mother: object describing the input
    :max_time: maximum time for computation
    :init_pop: population initiale
    :return: a list of integers of size n where the i-th element of the list is the component located in site i 
    """
    # Initialisation
    N_COMPONENTS = mother.n_components
    N_TAILLE_POPULATION = N_COMPONENTS*20
    NB_GENERATION= 1000
    TOURNAMENT_SIZE = int(N_TAILLE_POPULATION*0.6)
    MUTATION_RATE = 1 
    CROSS_RATE = 0 
    PARENTS_RATE = 0.1 

    # On représente le graphe par ses matrices de flux et de distance
    flows = nx.to_numpy_array(mother.graph, dtype=np.int32, weight="flow")
    dists = nx.to_numpy_array(mother.graph, dtype=np.int32, weight="dist")

    if init_pop == "random":
        # initialisation with random variable
        p_k = generate_population(N_COMPONENTS,N_TAILLE_POPULATION)
    elif init_pop == "grasp":
        # initialisation with grasp
        p_k = generate_population_grasp(N_COMPONENTS, N_TAILLE_POPULATION, mother, 60)
    else: print("error :  wrong initialiations should be random or grasp")

    p_k_eval=[]
    for s_k in p_k:
        #cost = f_eval(mother, s_k)
        cost = evaluation(np.array(s_k), flows, dists)
        p_k_eval.append(cost)
    
    index_min = np.argmin(np.array(p_k_eval))
    cost_star = p_k_eval[index_min]
    s_star = p_k[index_min]


    # initialisation des hyperparamétres
    i=0
    mutation_rate = MUTATION_RATE
    cross_rate = CROSS_RATE
    t0 = t.time()
    max_iter_without_improv=0
    # if the mutation is not with poisson law we define the  number of elements to mutate in a solution
    sub_sample_size=0.6
    lambda_p = 3
    restart=False
    while (i<NB_GENERATION and t.time()-t0 < max_time and not restart):
        print("generation : ",i, " best score", cost_star)
        # Selection ranking
        p_k_star= tournament(p_k, p_k_eval,population_size = N_TAILLE_POPULATION, tournament_size=TOURNAMENT_SIZE)

        # Hybridation
        c_k= crossing(p_k_star,n_components = N_COMPONENTS, cross_rate =cross_rate) # TODO : numpy, 2-point crossover (THEO)

        #Mutation
        m_k= permut_mutation(c_k,N_COMPONENTS, mut_rate=mutation_rate,lambda_p=lambda_p,sub_sample_size=sub_sample_size,poisson=True) # TODO : Poisson/swap/permutation (LAMIA)

        # compute the children scores
        m_k_eval=[]
        for s_k in m_k:
            #cost = f_eval(mother, s_k)
            cost = evaluation(np.array(s_k), flows, dists)
            m_k_eval.append(cost)


        # generate new population
        p_k, p_k_eval = generate_new_population(m_k, m_k_eval, p_k, p_k_eval, N_COMPONENTS,N_TAILLE_POPULATION,flows,dists,parents_rate=PARENTS_RATE) # TODO: point d'amélioration
        
        # Updating the best result
        index_min=np.argmin(np.array(p_k_eval))
        best_cost = p_k_eval[index_min]
        best_s_star = p_k[index_min]
        
        
        # control of parameters if the score doesn't improve
        # (controls intensification and variety)
        if cost_star > best_cost :
            cost_star= best_cost
            s_star = best_s_star
            lambda_p = 3
            # number of iteration where the score doesn't improve
            max_iter_without_improv=0
        else:
            max_iter_without_improv+=1
        if max_iter_without_improv >=30:
            
            lambda_p = int(N_COMPONENTS*r.uniform(0.01, 0.2))
        if max_iter_without_improv >=60:
            lambda_p = int(N_COMPONENTS*r.uniform(0.01, 0.3))
        if max_iter_without_improv>=100:
            restart = True
        
        # update mutation and crossing rate
        mutation_rate= 1-(i/NB_GENERATION)
        cross_rate = (i/NB_GENERATION)
        i+=1
    
    return s_star, cost_star


#####################################################################################################################################
##################################################### initialisation functions ######################################################
#####################################################################################################################################
def generate_individual(n_components):
    '''
    generate a random solution
    :param n_components: nombre de machine/slots
    :return res: solution
    '''
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


def generate_population(n_components, n_taille_population):
    """
    generate a population of random solution
    """
    population = [generate_individual(n_components) for _ in range(n_taille_population)]
    return population


def generate_population_mixte(n_components, flows, dists, n_taille_population):
    """
    genere un population partiellement aléatoire et avec une initialisation greedy.
    :param n_components: nombre de machine/slots
    :param flows: matrice des fluxs
    :param dists: matrice des distances
    :n_taille_population: taille de la population
    :return res: population
    """
    population_1 = [greedy_init1(n_components, flows, dists) for _ in range(n_taille_population//2)]
    population_2 = [generate_individual(n_components) for _ in range(n_taille_population//2)]
    population = population_1 + population_2 
    r.shuffle(population)
    return population

def generate_population_grasp(n_components, n_taille_population,mother,time_init):
    """
    genere un population partiellement aléatoire et avec une initialisation grasp.
    :param n_components: nombre de machine/slots
    :n_taille_population: taille de la population
    :time_init: temps d'initialisation des solution GRASP
    :return res: population
    """
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

def tournament(population, population_eval,population_size, tournament_size):
    """
    tournament to select best candidates in population
    :population: population (list of solutions)
    :population_eval: list of the score for each solution in the population
    :population_size: size of population
    :tournament_size: size of the tournament
    :return res: new population
    """
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
        select_index = r.sample(range(0, population_size), tournament_size)
        index_min_sol = np.argmin(population_eval_arr[select_index])
        selected.append(list((np.array(population)[select_index])[index_min_sol]))
    return selected

#####################################################################################################################################
##################################################### Hybridation functions #########################################################
#####################################################################################################################################
def crossing(population,n_components, cross_rate=0.95):
    """
    crossing, creating new children by crossing the 2 parents at a time chosen with a probability equals to cross_rate
    :population: population (list of solutions)
    :n_compenents: number of compenents
    :cross_rate: rate of crossed parents
    :return res: new population
    """

    assert len(population)%2 == 0, "Population de taille paire requise"

    m_one_arr = np.full((n_components),-1, dtype=int)
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
            c1 = np.where(mask_bool, p1, m_one_arr)
            c2 = np.where(mask_bool, p2, m_one_arr)
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
    """
    tournament to select best candidates in population
    :population: population (list of solutions)
    :population_eval: list of the score for each solution in the population
    :population_size: size of population
    :tournament_size: size of the tournament
    :return res: new population
    """

    if poisson and lambda_p==1 :
        # optimal value
        mut_rate=1/n_components

    m_k=[]
    for sol in c_k:
        if poisson:
            # if the poisson law is used to select the number of elements to permute
            nb_permut = min(n_components,max(2,np.random.poisson(lambda_p)))
        else:
            nb_permut =  min(n_components,max(2,int(n_components*sub_sample_size)))
        
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
    """
    tournament to select best candidates in population
    :m_k: population of children solution
    :m_k_eval: score of the children population
    :p_k: population of parent solution
    :p_k_eval: score of the parent population
    :n_components: number of compenents
    :n_taille_population: size of population
    :param flows: flow matrix
    :param dists: matrix of distances
    :parents_rate: the rate of best parents that will be kept
    :return res: new population
    """
    # keep parents_rate of the parent population
    nb_parents_kept = int(n_taille_population*(parents_rate))
    p_k_best_eval_index = (np.argsort(p_k_eval))[:nb_parents_kept]
    p_k_best = np.array(p_k)[p_k_best_eval_index]
    p_k_best_eval = np.array(p_k_eval)[p_k_best_eval_index]

    # keep (1-parents_rate) of the children population
    m_k_best_eval_index = (np.argsort(m_k_eval))[:len(p_k)-nb_parents_kept]
    m_k_best = np.array(m_k)[m_k_best_eval_index]
    m_k_best_eval = np.array(m_k_eval)[m_k_best_eval_index]
    
    new_p_k = list(p_k_best) + list(m_k_best)
    new_p_k_eval = list(p_k_best_eval) + list(m_k_best_eval)

    # suppress cloones and shuffle in a zip file
    c = list(zip(new_p_k, new_p_k_eval))
    new_c = []
    for elem in c:
        if list(elem) not in c:
            new_c.append(elem)
    r.shuffle(new_c)
    new_p_k, new_p_k_eval = zip(*new_c)

    # add random solutions to complet de population
    for i in range(n_taille_population-len(new_p_k)):
        new_p_k.append(generate_individual(n_components))
        new_p_k_eval.append(evaluation(np.array(new_p_k[-1]), flows, dists))
    
    
    return new_p_k, new_p_k_eval