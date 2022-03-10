from random import r
import time as t
import numpy as np

def solve(mother):
    """
    Your solution of the problem
    :param mother: object describing the input
    :return: a list of integers of size n where the i-th element of the list is the component located in site i 
    """

    # Initialisation

    # Add here your agent
    N_COMPONENTS = mother.n_component
    MAX_TIME = 10
    N_TAILLE_POPULATION=5 # TODO: A DEFINIR SELON LE PB
    NB_GENERATION= 5 # TODO: A DEFINIR



    p_k= generate_population(N_COMPONENTS,N_TAILLE_POPULATION) # TODO: A compléter A tester GLOUTON !!!!
    #s_star= generate_solution(N_COMPONENTS) # TODO: A tester GLOUTON !!!!


    cost_star = 100000
    for s_k in p_k:
        cost = f_eval(mother, s_k)
        if cost < cost_star:
            cost_star = cost
            s_star = s_k

    i=0
    t0 = t.time()
    while i<NB_GENERATION and t.time()-t0 < MAX_TIME:

        # Selection
        p_k_star= roulette(p_k) # TODO: ranking ou tournoi (LAMIA)
        #p_k_star= ranking(p_k)

        # Hybridation
        c_k= crossing(p_k_star) # TODO : numpy, 2-point crossover (THEO)

        #Mutation
        #TODO: paramter tuning : mut_rate adaptatif et lambda
        m_k= permut_mutation(c_k,mother, mut_rate=0.7,lambda_p=1,poisson=True) # TODO : Poisson/swap/permutation (LAMIA)

        # Updating
        for s_k in m_k:
            cost = f_eval(mother, s_k)
            if cost < best_cost:
                best_cost = cost
                s_star= s_k
        
        # generate new population
        p_k = generate_new_population(m_k,p_k) # TODO: point d'amélioration

    
    return s_star


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

def generate_population(n_components,n_zones, n_taille_population):
    population = [generate_individual(n_components,n_zones) for _ in range(n_taille_population)]
    return population

#####################################################################################################################################
##################################################### population selection ##########################################################
#####################################################################################################################################
def roulette(population, mother):

    assert len(population) % 2 == 0, "populationn size should be an even number"
    
    # costs for each solution of the population
    sol_costs=[]
    for sol in population:
        sol_costs.append(f_eval(mother, sol))

    
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

def ranking(p_k):
    return None

#####################################################################################################################################
##################################################### Hybridation functions #########################################################
#####################################################################################################################################
def crossing(p_k_star):
    return None

#####################################################################################################################################
##################################################### Mutation functions #########################################################
#####################################################################################################################################

# augmenter taux de pumtation = diversification, si y a des clones il manque de diversification
def permut_mutation(c_k,mother, mut_rate=0.7,lambda_p=1,poisson=True):
    if poisson and lambda_p==1 :
        mut_rate=1/mother.n_component
    
    nb_permut = int(mother.n_component*0.1)
    m_k=[]
    for sol in c_k:
        if poisson:
            nb_permut = np.random.poisson(lambda_p)
        Pr_mutation = r.random() #probability to mutate

        if Pr_mutation < mut_rate:
            # permutations of a subsample of the solution
            sub_sample = r.sample(list(range(mother.n_component)),nb_permut)
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


#####################################################################################################################################
########################################### New population generation functions #####################################################
#####################################################################################################################################
def generate_new_population(m_k,p_k):
    return m_k