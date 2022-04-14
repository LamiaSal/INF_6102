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
    MAX_TIME =1200
    k_max=2
    MAX_ITER_WITHOUT_IMP = 10

    solution = init_encoded(factory)

    for i in range(100):
        s = init_encoded(factory)
        if f(factory, decode_sol(factory, s)) < f(factory, decode_sol(factory, solution)):
            solution  = copy.deepcopy(s)
    
    s_star = copy.deepcopy(solution)
    score_star= f(factory, decode_sol(factory, s_star))
    print("score de départ", score_star)
    
    t0 = t.time()
    nb_iter_restart = 0
    while t.time()-t0 < MAX_TIME:
        
        solution = GVNS(factory, solution,k_max,t0,MAX_TIME)
        solution_score = f(factory, decode_sol(factory, solution))
        if solution_score < score_star:
            s_star = copy.deepcopy(solution)
            score_star = solution_score
            print("score_star",score_star)
            nb_iter_restart = 0
        else:
            nb_iter_restart += 1
        if nb_iter_restart >= MAX_ITER_WITHOUT_IMP:
            solution = init_encoded(factory)
            print("RESTART numéro", nb_iter_restart, "best cost:",score_star)
            nb_iter_restart = 0

    return decode_sol(factory, s_star)

def init_encoded(factory):
        solution =[]
        for i in range(1,factory.n_jobs+1):
            solution += [ i ]*factory.n_ope[i]
        r.shuffle(solution)
        assert len(solution) == sum(factory.n_ope.values())
        return solution

def decode_sol(factory, sol_encoded):

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


# TODO: considérer les voisinage du plus petit au plus grand
def GVNS(factory,s,k_max, t0, max_time):
    k=1
    while k<=k_max:
        s_prime = shake(factory, s,k)
        s_prime_prime = VND(factory,s_prime,k_max,t0, max_time) #BestImprovement(factory,k, s_prime)
        s, k = NeighborhoodChange(factory, s, s_prime_prime, k)
        #print(k)
        if t.time()-t0 >= max_time :
            print("done")
            return s
    return s

def VND(factory,s,k_max,t0, max_time):
    k=1
    while k<=k_max:
        s_prime = BestImprovement(factory, s, k, t0, max_time)
        s, k = NeighborhoodChange(factory, s, s_prime, k)
        if t.time()-t0 >= max_time :
            print("done")
            return s
    return s

def NeighborhoodChange(factory, s, s_prime, k):
    '''
    TODO: description
    '''
    if f(factory,  decode_sol(factory, s_prime))<f(factory, decode_sol(factory, s)):
        s=copy.deepcopy(s_prime) # make a move
        k=1 # initial neighborhood
        #print("improved", f(factory,  decode_sol(factory, s_prime)) )
    else:
        k+=1 # next neighborhood
    return s, k

def shake(factory, solution, k):
    # Exchange +Insert+ Exchange
    s_prime = copy.deepcopy(solution)
    '''
    l = r.sample(range(1,len(s_prime)-1), 6)
    s_prime = interchange_single(s_prime, l[0], l[1])
    s_prime = insertion_single(s_prime, l[2], l[3])
    s_prime = interchange_single(s_prime, l[4], l[5])
    '''
    p = r.sample(range(1,len(s_prime)-1), 2)
    if k == 1:
        s_prime = swap_single(s_prime, p[0])
    elif k == 2:
        s_prime = interchange_single(s_prime, p[0], p[1])
    elif k == 3:
        s_prime = insertion_single(s_prime, p[0], p[1])
    else :
        raise Exception("number of beinghbor exceeded")
    
    return s_prime

# TODO: les voisinages doivent respecter les contraintes dures
def BestImprovement(factory,solution, k, t0, max_time):
    if k == 1:
        # voisinage 1 : permuter les opérations côte à côte entre elles (2-swap)
        s_prime = swap(factory, solution, t0, max_time)
        #print("swap")
    elif k == 2:
        s_prime = interchange(factory, solution, t0, max_time)
        #print("interchange")
    elif k == 3:
        s_prime = insertion(factory, solution, t0, max_time)
        #print("insertion")
    # autre idée exchange mais que les 2 trucs à côté
    else :
        raise Exception("number of beinghbor exceeded")
    return s_prime

def swap(factory, solution, t0, max_time):
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
    new_sol = copy.deepcopy(solution)
    new_sol[i], new_sol[i+1] = solution[i+1], solution[i]
    return new_sol

def interchange(factory, solution, t0, max_time):
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
    new_sol = copy.deepcopy(solution)
    new_sol[i], new_sol[j] = solution[j], solution[i]
    return new_sol

def insertion(factory, solution, t0, max_time):
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
                if t.time()-t0 >= max_time :
                    return solution
    return solution
    
def insertion_single(solution, i, j):
    new_sol = copy.deepcopy(solution)
    val = new_sol.pop(i)
    new_sol.insert(j, val)
    return new_sol



def f(factory, solution):
    return factory.get_total_cost(solution)

def sol_init_NEH(factory):
    return None