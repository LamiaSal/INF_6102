from random import random
import time as t

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


    best_cost = 10**10
    for s_k in p_k:
        cost = f_eval(s_k)
        if cost < best_cost:
            best_cost = cost
            s_star = s_k

    i=0
    t0 = t.time()
    while i<NB_GENERATION and t.time()-t0 < MAX_TIME:

        # Selection
        p_k_star= roulette(p_k) # TODO: A tester d'autres méthodes

        # Hybridation
        c_k= generate_children(p_k_star)

        #Mutation
        m_k= mutation(c_k)


        # Updating
        for s_k in m_k:
            cost = f_eval(s_k)
            if cost < best_cost:
                best_cost = cost
                s_star= s_k
        
        # generate new population
        p_k = generate_population(m_k,p_k)

    
    return s_star


'''
def generate_solution(n_components):
    solution = list(range(n_components))
    random.shuffle(solution)
    return solution
'''

def generate_individual(n_components):
    try:
        individual = random.sample(range(0, n_components), n_components)
    except ValueError:
        print('Sample size exceeded population size.')
    return individual

def generate_population(n_components,n_zones, n_taille_population):
    population = [generate_individual(n_components,n_zones) for _ in range(n_taille_population)]
    return population

def roulette(p_k):
    return None

def generate_children(p_k_star):
    return None

def mutation(c_k):
    return None

def f_eval():
    return None

#def generate_population(p_k_1,m_k):
#    return None