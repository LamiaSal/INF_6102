import random as r
import time as t
import numpy as np
import networkx as nx

rng = np.random.default_rng()

def solve(mother):
    """s
    Your solution of the problem
    :param mother: object describing the input
    :return: a list of integers of size n where the i-th element of the list is the component located in site i 
    """
    MAX_TIME = 1200
    cost_star = 10000
    t0 = t.time()
    n_restarts = 0

    while n_restarts < 1000 and t.time()-t0 < MAX_TIME:
        solution,cost_sol = genetic_search(mother,max_time=MAX_TIME)  

        # Update if amelioration
        if cost_sol < cost_star:
            cost_star = cost_sol
            s_star = solution
        n_restarts += 1
        print("RESTART numéro", n_restarts, "best cost:",cost_star)
    
    return s_star
    genetic_search(mother)
def genetic_search(mother, max_time):
    # Initialisation
    N_COMPONENTS = mother.n_components
    N_TAILLE_POPULATION = N_COMPONENTS*10
    NB_GENERATION= 1000 
    TOURNAMENT_SIZE = N_TAILLE_POPULATION//10
    MUTATION_RATE = 0.6 # 0.9 with glouton initialisation
    PARENTS_RATE = 0.5 # 0.2 if we use glouton initialisation

    # On représente le graphe par ses matrices de flux et de distance
    dtype = np.dtype([("flow", np.uint8), ("dist", np.uint8)])
    graph_dict = nx.to_numpy_array(mother.graph, dtype=dtype, weight=None)
    flows, dists = graph_dict["flow"], graph_dict["dist"]
    

    #p_k = generate_population(N_COMPONENTS,N_TAILLE_POPULATION) # TODO: A compléter A tester GLOUTON !!!!
    #p_k =generate_population_init2(N_COMPONENTS, flows, dists, N_TAILLE_POPULATION)
    p_k= generate_population_greedy(N_COMPONENTS, flows, dists, N_TAILLE_POPULATION) # TODO: A compléter A tester GLOUTON !!!!
    #s_star= generate_solution(N_COMPONENTS) # TODO: A tester GLOUTON !!!!


    
    p_k_eval=[]
    for s_k in p_k:
        cost = f_eval(mother, s_k)
        #cost = evaluation(s_k, flows, dists)
        p_k_eval.append(cost)
    
    #print(p_k_eval)
    index_min = np.argmin(np.array(p_k_eval))
    cost_star = p_k_eval[index_min]
    s_star = p_k[index_min]

    i=0
    #s_stars=[]
    #s_stars_eval=[]
    mutation_rate = MUTATION_RATE
    t0 = t.time()
    max_iter_without_improv=0
    sub_sample_size=0.6
    lambda_p=3
    restart=False
    while (i<NB_GENERATION and t.time()-t0 < max_time and not restart):
        print("genereation:",i,"best cost",cost_star)
        # Selection (ranking ou tournoi)
        p_k_star= tournament(p_k, p_k_eval,mother, n_components=N_COMPONENTS,population_size = N_TAILLE_POPULATION, tournament_size=TOURNAMENT_SIZE)

        #print("len(p_k_star)",len(p_k_star))
        # Hybridation
        c_k= crossing(p_k_star,n_components = N_COMPONENTS, probablity=0.5) # TODO : numpy, 2-point crossover (THEO)

        #print("len(c_k)",len(c_k))
        #Mutation
        #TODO: paramter tuning : mut_rate adaptatif et lambda
        m_k= permut_mutation(c_k,N_COMPONENTS, mut_rate=mutation_rate,lambda_p=lambda_p,sub_sample_size=sub_sample_size,poisson=True) # TODO : Poisson/swap/permutation (LAMIA)

        #print("len(m_k)",len(m_k))
        # Updating
        
        m_k_eval=[]
        for s_k in m_k:
            cost = f_eval(mother, s_k)
            m_k_eval.append(cost)
        #m_k_eval_argsort = np.argsort(m_k_eval)


        # generate new population
        p_k, p_k_eval = generate_new_population(m_k, m_k_eval, p_k, p_k_eval, N_COMPONENTS,N_TAILLE_POPULATION,parents_rate=PARENTS_RATE) # TODO: point d'amélioration
        
        index_min=np.argmin(np.array(p_k_eval))
        best_cost = p_k_eval[index_min]
        best_s_star = p_k[index_min]
        if cost_star > best_cost :
            cost_star= best_cost
            s_star = best_s_star
            # paramters for mutation (controls intensification and variety)
            mutation_rate=r.uniform(0.4, 0.8)
            sub_sample_size=r.uniform(0.1, 0.5)
            lambda_p=3
            #lambda_p = 3
            max_iter_without_improv=0
        else:
            max_iter_without_improv+=1
        if max_iter_without_improv >=10:
            # parameters for mutation (controls intensification and variety)
            mutation_rate=0.99
            lambda_p = 8
            sub_sample_size=r.uniform(0.7, 0.9)
        if max_iter_without_improv>=50:
            restart = True
        #print(restart)
        #print(max_iter_without_improv)
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


def generate_population(n_components, n_taille_population):
    population = [generate_individual(n_components) for _ in range(n_taille_population)]
    return population

def generate_population_greedy(n_components, flows, dists, n_taille_population):
    population = [greedy_init1(n_components, flows, dists) for _ in range(n_taille_population)]
    return population

def generate_population_init2(n_components, flows, dists, n_taille_population):
    population = [greedy_init2(n_components, flows, dists) for _ in range(n_taille_population)]
    return population
#####################################################################################################################################
##################################################### population selection ##########################################################
#####################################################################################################################################
def roulette(population, mother):

    assert len(population) % 2 == 0, "population size should be an even number"
    
    # costs for each solution of the population
    sol_costs=np.array([])
    for sol in population:
        sol_costs.append(f_eval(mother, sol))
    #print(sol_costs)
    
    roulette_list=population.copy()
    selected=[]
    for i in range(len(population)):
    #while len(roulette_list)!=0:

        # cumulative probabilities of the solution 
        sol_probas = []
        for j in range(len(roulette_list)):
            Pr_j=sol_costs[j]/sum(sol_costs)
            sol_probas.append(Pr_j)
        
        # random jet
        ceil = r.random()
        k = 0
        while sol_probas[i] < ceil:
            k += 1
        
        # add selected solution
        selected.append(roulette_list.pop(k))
        sol_costs.pop(k)

    return selected

def tournament(population, population_eval, mother,n_components,population_size, tournament_size):

    # verification of some connditions
    if tournament_size%2!=0:
        tournament_size+=1
    assert population_size == len(population), "population size should be equal to len(population)"
    assert population_size % 2 == 0, "population size should be an even number"
    assert population_size >= tournament_size, "population size should be higher than the tournament size"

    #r.shuffle(population)
    sol_costs=np.array([])
    for sol in population:
        sol_costs = np.append(sol_costs,f_eval(mother, sol))
    
    selected=[]
    size_subsets = population_size//tournament_size
    population_eval_arr=np.array(population_eval)

    for i in range(population_size):
        select_index = r.sample(range(0, len(population)), tournament_size)
        index_min_sol = np.argmin(population_eval_arr[select_index])
        selected.append((np.array(population)[select_index])[index_min_sol])
    return selected

#####################################################################################################################################
##################################################### Hybridation functions #########################################################
#####################################################################################################################################
def crossing(population,n_components, probablity=0.5):
    
    assert len(population)%2 == 0, "Population de taille paire requise"
    #r.shuffle(population)

    zeros_arr = np.full((n_components),-1, dtype=int)
    children = []
    for i in range(len(population)//2):

        # Initialisation of parents
        p1 = np.array(population[2*i])
        p2 = np.array(population[2*i+1])

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
    return np.sum(flows[solution][:,solution] * dists)


#####################################################################################################################################
########################################### New population generation functions #####################################################
#####################################################################################################################################
def generate_new_population(m_k, m_k_eval, p_k, p_k_eval, N_COMPONENTS,N_TAILLE_POPULATION, parents_rate=0.5):
    # gardez 50% parents et 50% enfants

    nb_parents_kept = int(N_TAILLE_POPULATION*(parents_rate))
    p_k_best_eval_index = (np.argsort(p_k_eval))[:int(N_TAILLE_POPULATION*parents_rate)]
    p_k_best = np.array(p_k)[p_k_best_eval_index]
    p_k_best_eval = np.array(p_k_eval)[p_k_best_eval_index]
    print("np.min(p_k)",np.min(p_k_best_eval))
    m_k_best_eval_index = (np.argsort(m_k_eval))[:len(p_k)-nb_parents_kept]
    m_k_best = np.array(m_k)[m_k_best_eval_index]
    m_k_best_eval = np.array(m_k_eval)[m_k_best_eval_index]
    print("np.min(m_k)",np.min(m_k_best_eval))
    
    new_p_k = list(p_k_best) + list(m_k_best)
    new_p_k_eval = list(p_k_best_eval) + list(m_k_best_eval)

    c = list(zip(new_p_k, new_p_k_eval))
    r.shuffle(c)
    new_p_k, new_p_k_eval = zip(*c)
    #new_p_k_eval = m_k_eval[:N_TAILLE_POPULATION//2] + p_k_eval[:N_TAILLE_POPULATION//2]
    #print(len(new_p_k))

    
    return new_p_k, new_p_k_eval