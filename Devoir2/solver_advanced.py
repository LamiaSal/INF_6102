import random as r
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
    N_COMPONENTS = mother.n_components
    MAX_TIME = 30
    N_TAILLE_POPULATION= 30 # TODO: A DEFINIR SELON LE PB
    NB_GENERATION= 2 # TODO: A DEFINIR



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
        print("genereation:",i,"best cost",cost_star)
        # Selection
        #p_k_star= roulette(p_k) # TODO: ranking ou tournoi (LAMIA)
        p_k_star= tournament(p_k, mother, tournament_size=10)

        print("len(p_k_star)",len(p_k_star))
        # Hybridation
        c_k= crossing(p_k_star,mother) # TODO : numpy, 2-point crossover (THEO)

        print("len(c_k)",len(c_k))
        #Mutation
        #TODO: paramter tuning : mut_rate adaptatif et lambda
        m_k= permut_mutation(c_k,mother, mut_rate=0.3,lambda_p=1,sub_sample_size=0.4,poisson=True) # TODO : Poisson/swap/permutation (LAMIA)

        print("len(m_k)",len(m_k))
        # Updating
        for s_k in m_k:
            cost = f_eval(mother, s_k)
            if cost < cost_star:
                cost_star = cost
                s_star= s_k
        
        # generate new population
        p_k = generate_new_population(m_k,p_k) # TODO: point d'amélioration

        i+=1
    
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

def generate_population(n_components, n_taille_population):
    population = [generate_individual(n_components) for _ in range(n_taille_population)]
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

def tournament(population, mother, tournament_size):

    # costs for each solution of the population

    r.shuffle(population)
    sol_costs=[]
    for sol in population:
        sol_costs.append(f_eval(mother, sol))
    
    selected=[]
    size_subsets = len(population)//tournament_size
    #f size_subsets%2!=0:
    #    size_subsets-=1
    
    #print("pop len",len(population))
    #print("tournament size",tournament_size)
    #print("subset size",size_subsets)
    low_born=0
    high_born=size_subsets
    for i in range(tournament_size):
        #print(i)
        #print(low_born,"::",high_born)
        if i==tournament_size-1:
            k = np.argmax(sol_costs[low_born:])
            selected.append(population[low_born:][k])
            low_born=high_born
            high_born+=size_subsets
            
        else:
            k = np.argmax(sol_costs[low_born:high_born])
            # add selected solution
            selected.append(population[low_born:high_born][k])
            low_born=high_born
            high_born+=size_subsets

    return selected

#####################################################################################################################################
##################################################### Hybridation functions #########################################################
#####################################################################################################################################
def crossing(population,mother):
    
    assert len(population)%2 == 0, "Population de taille paire requise"
    r.shuffle(population)

    zeros_arr = np.full((mother.n_components),-1, dtype=int)
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

        population.append(list(c1))
        population.append(list(c2))
    
    return population

#####################################################################################################################################
######################################################## Mutation functions #########################################################
#####################################################################################################################################

# augmenter taux de pumtation = diversification, si y a des clones il manque de diversification
def permut_mutation(c_k,mother, mut_rate=0.3,lambda_p=1,sub_sample_size=0.4,poisson=True):
    if poisson and lambda_p==1 :
        mut_rate=1/mother.n_components
    
    nb_permut =  max(2,int(mother.n_components*0.4))

    m_k=[]
    for sol in c_k:
        if poisson:
            nb_permut = max(2,np.random.poisson(lambda_p))
        Pr_mutation = r.random() #probability to mutate

        if Pr_mutation < mut_rate:
            # permutations of a subsample of the solution
            sub_sample = r.sample(list(range(mother.n_components)),nb_permut)
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