import copy
import random as r
import time as t
import numpy as np

def solve(factory):
    """
    A random feasible solution of the problem
    :param factory: object describing the input
    
    :return: a dictionnary where the keys are the machines and the values are the ordered list of tuples (job, operation, start, end) 
             treated by the machines
    """
    
    # time maxed to 20 minutes
    MAX_TIME = 1200
    k_max=2
    MAX_ITER_WITHOUT_IMP = 10

    solution = init_encoded(factory)

    for i in range(100):
        s = init_encoded(factory)
        if f(factory, decode_sol(factory, s)) < f(factory, decode_sol(factory, solution)):
            solution  = copy.deepcopy(s)
    
    s_star = copy.deepcopy(solution)
    score_star= f(factory, decode_sol(factory, s_star))
    #print("score de départ", score_star)
    
    t0 = t.time()
    nb_iter_restart = 0
    while t.time()-t0 < MAX_TIME:
        
        solution = GVNS(factory, solution,k_max,t0,MAX_TIME)
        solution_score = f(factory, decode_sol(factory, solution))
        if solution_score < score_star:
            s_star = copy.deepcopy(solution)
            score_star = solution_score
            nb_iter_restart = 0
        else:
            nb_iter_restart += 1
        if nb_iter_restart >= MAX_ITER_WITHOUT_IMP:
            solution = init_encoded(factory)
            print("RESTART numéro", nb_iter_restart, "best cost:",score_star)
            nb_iter_restart = 0

    return decode_sol(factory, s_star)

def init_encoded(factory):
    '''
    create random chromosome
    input : 
    :param factory: object describing the input

    output :
    param solution: list of integers correponding to the operations cf rapport for more information.
    '''
    solution =[]
    for i in range(1,factory.n_jobs+1):
        solution += [ i ]*factory.n_ope[i]
    r.shuffle(solution)
    assert len(solution) == sum(factory.n_ope.values())
    return solution

def decode_sol(factory, sol_encoded):
    '''
    function to decode the chromosome into an actual planning
    input : 
    :param factory: object describing the input
    :sol_encoded: chromosome, list of integers correponding to the operations cf rapport for more information.

    output :
    param solution: a dictionnary where the keys are the machines and the values are the ordered list of tuples (job, operation, start, end) 
    treated by the machines
    '''

    ope_job = {i:0 for i in range(1, factory.n_jobs+1)}
    solution = {m:[] for m in range(1,factory.n_mach+1)}

    time_job = {i+1:0 for i in range(factory.n_jobs)}  # Times until which each job is currently being done
    time_mach = {i+1:0 for i in range(factory.n_mach)} # Times until which each machine is currently working

    for job in sol_encoded:
        # ASSIGNER DE FAÇON PLUS SMART les jobs au machines !!
        machines = [m for m in range(1,factory.n_mach+1) if (job,ope_job[job]+1,m) in factory.p]
        
        mach = machines[0]
        time_m=time_mach[mach]
        for m in range(1, len(machines)):
            if time_mach[machines[m]] <= time_m:
                mach = machines[m]
                time_m = time_mach[mach]
        
        # Computing the start and end dates
        te = max(time_job[job],time_mach[mach])
        ts = te + factory.p[(job,ope_job[job]+1,mach)]

        # Updating
        ope_job[job]+=1
        solution[mach].append((job,ope_job[job],te,ts))
        time_job[job] = ts
        time_mach[mach] = ts
    return solution 



def GVNS(factory,s,k_max, t0, max_time):
    '''
    GVNS : general variable neighbor search
    input : 
    :param factory: object describing the input
    :s: chromosome, list of integers correponding to the operations cf rapport for more information.
    :k_max: number of nieghborhood considered
    :t0: time chen the search begun
    :max_time: maximum time for the execution (20 minutes)

    output :
    :s: chromosome, list of integers correponding to the operations cf rapport for more information.
    '''
    k=1
    while k<=k_max:
        s_prime = shake(factory, s,k)
        s_prime_prime = VND(factory,s_prime,k_max,t0, max_time)
        s, k = NeighborhoodChange(factory, s, s_prime_prime, k)
        if t.time()-t0 >= max_time :
            return s
    return s

def VND(factory,s,k_max,t0, max_time):
    '''
    VND : variable neighbor descent
    input : 
    :param factory: object describing the input
    :s: chromosome, list of integers correponding to the operations cf rapport for more information.
    :k_max: number of nieghborhood considered
    :t0: time chen the search begun
    :max_time: maximum time for the execution (20 minutes)

    output :
    :s: chromosome, list of integers correponding to the operations cf rapport for more information.
    '''
    k=1
    while k<=k_max:
        s_prime = BestImprovement(factory, s, k, t0, max_time)
        s, k = NeighborhoodChange(factory, s, s_prime, k)
        if t.time()-t0 >= max_time :
            return s
    return s

def NeighborhoodChange(factory, s, s_prime, k):
    '''
    Function to decide if a move should be made or not
    input : 
    :param factory: object describing the input
    :s: previou chromosome
    :s_prime: chroomosome with the move made
    :k: cureent neighborhood

    output :
    :s: chromosome after move or not
    :k: next neighborhood 
    '''
    if f(factory,  decode_sol(factory, s_prime))<f(factory, decode_sol(factory, s)):
        s=copy.deepcopy(s_prime) # make a move
        k=1 # initial neighborhood
    else:
        k+=1 # next neighborhood
    return s, k

def shake(factory, solution, k):
    '''
    compute the best neighbor of the neighborhood k
    input : 
    :param factory: object describing the input
    :solution: chromosome, list of integers correponding to the operations cf rapport for more information.
    :k: current neighbor
    
    output :
    :s_prime: a neighbor of the neighborhood k
    '''
    s_prime = copy.deepcopy(solution)
    p = r.sample(range(1,len(s_prime)-1), 2)
    if k == 2:
        s_prime = swap_single(s_prime, p[0])
    elif k == 1:
        s_prime = interchange_single(s_prime, p[0], p[1])
    elif k == 3:
        s_prime = insertion_single(s_prime, p[0], p[1])
    else :
        raise Exception("number of beinghbor exceeded")
    
    return s_prime


def BestImprovement(factory,solution, k, t0, max_time):
    '''
    compute the best neighbor of the neighborhood k
    input : 
    :param factory: object describing the input
    :solution: chromosome, list of integers correponding to the operations cf rapport for more information.
    :k: current neighbor
    :t0: time when the search begun
    :max_time: maximum time for the execution (20 minutes)
    

    output :
    :s_prime: the best neighbor k solution
    '''
    if k == 2:
        s_prime = swap(factory, solution, t0, max_time)
    elif k == 1:
        s_prime = interchange(factory, solution, t0, max_time)
    elif k == 3:
        s_prime = insertion(factory, solution, t0, max_time)
    else :
        raise Exception("number of beinghbor exceeded")
    return s_prime


############ swap ############
def swap(factory, solution, t0, max_time):
    '''
    swap neighborhood
    input : 
    :param factory: object describing the input
    :solution: chromosome, list of integers correponding to the operations cf rapport for more information.
    :t0: time when the search begun
    :max_time: maximum time for the execution (20 minutes)

    output :
    :solution: the best neighbor solution after swap
    '''
    sol_score = f(factory, decode_sol(factory, solution))
    for i in range(len(solution)-1):
        new_sol = swap_single(solution, i)
        new_sol_score = f(factory, decode_sol(factory, new_sol))
        if new_sol_score < sol_score :
            sol_score=new_sol_score
            solution = copy.deepcopy(new_sol)
        if t.time()-t0 >= max_time :
            return solution
    return solution
    
def swap_single(solution, i):
    '''
    interchange, interchnage the i ème element of the solution and with the i+1 eme element
    input : 
    :solution: chromosome, list of integers correponding to the operations cf rapport for more information.
    :i: integer

    output :
    :solution: the neighbor solution
    '''
    new_sol = copy.deepcopy(solution)
    new_sol[i], new_sol[i+1] = solution[i+1], solution[i]
    return new_sol

########## interchange ##########
def interchange(factory, solution, t0, max_time):
    '''
    interchange neighborhood
    input : 
    :param factory: object describing the input
    :solution: chromosome, list of integers correponding to the operations cf rapport for more information.
    :t0: time when the search begun
    :max_time: maximum time for the execution (20 minutes)

    output :
    :solution: the best neighbor solution after innterchange
    '''
    sol_score = f(factory, decode_sol(factory, solution))
    for i in range(len(solution)-1):
        for j in range(len(solution)-1):
            if i!=j:
                new_sol = interchange_single(solution, i, j)
                new_sol_score = f(factory, decode_sol(factory, new_sol))
                if new_sol_score < sol_score :
                    sol_score=new_sol_score
                    solution = copy.deepcopy(new_sol)
                if t.time()-t0 >= max_time :
                    return solution
    return solution

def interchange_single(solution, i, j):
    '''
    interchange, interchnage the i ème element of the solution and with the j eme element
    input : 
    :solution: chromosome, list of integers correponding to the operations cf rapport for more information.
    :i: integer
    :j: integer

    output :
    :solution: the neighbor solution
    '''
    new_sol = copy.deepcopy(solution)
    new_sol[i], new_sol[j] = solution[j], solution[i]
    return new_sol

########## insertion ##########
def insertion(factory, solution, t0, max_time):
    '''
    insertion neighborhood
    input : 
    :param factory: object describing the input
    :solution: chromosome, list of integers correponding to the operations cf rapport for more information.
    :t0: time when the search begun
    :max_time: maximum time for the execution (20 minutes)

    output :
    :solution: the best neighbor solution after insertion
    '''
    sol_score = f(factory, decode_sol(factory, solution))
    for i in range(len(solution)-1):
        for j in range(len(solution)-1):
            if i!=j:
                new_sol = insertion_single(solution, i, j)
                #new_sol[i], new_sol[j] = solution[j], solution[i]
                new_sol_score = f(factory, decode_sol(factory, new_sol))
                if new_sol_score < sol_score :
                    sol_score=new_sol_score
                    solution = copy.deepcopy(new_sol)
                    return solution
                if t.time()-t0 >= max_time :
                    return solution
    return solution
    
def insertion_single(solution, i, j):
    '''
    insertion, take the i ème element of the solution and put it in front of the j eme element
    input : 
    :solution: chromosome, list of integers correponding to the operations cf rapport for more information.
    :i: integer
    :j: integer

    output :
    :solution: the neighbor solution
    '''
    new_sol = copy.deepcopy(solution)
    val = new_sol.pop(i)
    new_sol.insert(j, val)
    return new_sol


########## Evaluation function ##########
def f(factory, solution):
    return factory.get_total_cost(solution)
