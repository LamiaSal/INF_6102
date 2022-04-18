import time as t
import numpy as np
import threading

def solve(factory):
    """
    A random feasible solution of the problem
    :param factory: object describing the input
    
    :return: a dictionnary where the keys are the machines and the values are the ordered list of tuples (job, operation, start, end) 
             treated by the machines
    """

    n_jobs, n_machines, n_ope, n_total_ope, durations = transform(factory)

    # time maxed to 20 minutes
    MAX_TIME = 1200
    k_max = 2
    MAX_ITER_WITHOUT_IMP = 10  # 5*n_jobs ? n_total_ope ? 10 ?
    global interpret_mode
    interpret_mode = 5

    solution, solution_score = init_random(n_jobs, n_ope, n_machines, n_total_ope, durations, factory, 1000)

    s_star = solution.copy()
    score_star = solution_score
    print("score de départ", score_star)
    
    t0 = t.time()
    nb_iter_restart = 0
    n_tot_iter = 0
    n_restart = 0
    while t.time()-t0 < MAX_TIME:
        
        solution, solution_score, n_iter = GVNS(n_jobs, n_machines, n_ope, n_total_ope, durations, factory, solution, solution_score, k_max, t0, MAX_TIME)
        n_tot_iter += n_iter
        if solution_score < score_star:
            s_star = solution.copy()
            score_star = solution_score
            #print("score_star",score_star)
            nb_iter_restart = 0
        else:
            nb_iter_restart += 1

        if nb_iter_restart >= MAX_ITER_WITHOUT_IMP:
            n_restart += 1
            solution, solution_score = init_random(n_jobs, n_ope, n_machines, n_total_ope, durations, factory, 1000)
            #print("RESTART numéro", nb_iter_restart, "best cost:",score_star)
            nb_iter_restart = 0

    print("Nb de restart : ", n_restart)
    print("Nb total d'itérations : ", n_tot_iter)

    if interpret_mode == 4:
        complete_solution = interpret_python_greedy(factory, s_star)
        greedy_score = evaluation(complete_solution)
        if greedy_score < score_star:
            return complete_solution

    return decode(interpret(s_star, n_jobs, n_machines, n_ope, n_total_ope, durations, factory))


#################################
# Transformation des données de base
#################################
def transform(factory):
    n_jobs = factory.n_jobs
    n_machines = factory.n_mach
    n_ope = np.zeros(n_jobs, dtype=np.int)
    for k,v in factory.n_ope.items():
        n_ope[k-1] = v
    n_total_ope = sum(n_ope)
    durations = np.full((n_jobs,max(n_ope),n_machines),-1, dtype=np.int)
    for (j,o,m),v in factory.p.items():
        durations[j-1,o-1,m-1] = v
    return n_jobs, n_machines, n_ope, n_total_ope, durations


#################################
# Décodeurs
#################################
def interpret(solution, n_jobs, n_machines, n_ope, n_total_ope, durations, factory):
    if interpret_mode == 0:
        return interpret_greedy(solution, n_jobs, n_machines, n_ope, n_total_ope, durations)
    elif interpret_mode == 1:
        return interpret_min(solution, n_jobs, n_machines, n_ope, n_total_ope, durations)
    elif interpret_mode == 2:
        return interpret_base(solution, n_jobs, n_machines, n_ope, n_total_ope, durations)
    elif interpret_mode == 3:
        return interpret_base_exact(solution, n_jobs, n_machines, n_ope, n_total_ope, durations)
    elif interpret_mode == 4:
        return interpret_python(factory, solution)
    elif interpret_mode == 5:
        return interpret_python_greedy(factory, solution)


def interpret_greedy(solution, n_jobs, n_machines, n_ope, n_total_ope, durations):
    complete_solution = np.full((n_machines,int(max(n_ope)*2),4),-1, dtype=np.int)
    complete_solution[:,:,2:] = np.zeros((n_machines,int(max(n_ope)*2),2), dtype=np.int)
    current_ope = np.zeros(n_jobs, dtype=np.int)
    current_finish_time = np.zeros(n_jobs, dtype=np.int)
    last_slot_active = np.full(n_machines, -1, dtype=np.int)
    max_time = 0
    max_time_per_machine = np.zeros(n_machines, dtype=np.int)
    for j in solution:
        max_slot = max(0,max(last_slot_active))
        ope = current_ope[j]
        mask = durations[j,ope,:] != -1
        possible_machines = np.nonzero(mask)[0]
        costs = np.reshape(durations[j, ope, :][mask], (len(possible_machines),1))

        sol_view = complete_solution[possible_machines,:max_slot+1]
        mask2 = (np.maximum(sol_view[:,:max_slot,3],current_finish_time[j]) + costs <= sol_view[:,1:,2])
        internal_slots = np.argwhere(mask2)
        internal_slots[:,0] = possible_machines[internal_slots[:,0]]
        internal_timespan = np.full(len(internal_slots), max_time, dtype=np.int)

        external_slots = np.stack((possible_machines, last_slot_active[possible_machines]), axis=-1)
        external_timespan = np.maximum(max_time_per_machine[possible_machines] + costs[:,0], max_time)

        possible_slots = np.concatenate((internal_slots, external_slots), axis=0)
        possible_timespan = np.concatenate((internal_timespan, external_timespan))

        selected = np.nonzero(possible_timespan == possible_timespan.min())
        selected_slots = possible_slots[selected]
        endtimers = np.maximum(complete_solution[selected_slots[:,0],selected_slots[:,1],:][:,3],current_finish_time[j]) + durations[j,ope,:][selected_slots[:,0]]
        machine, last_slot = selected_slots[np.argmin(endtimers)]

        if last_slot < last_slot_active[machine]:
            complete_solution[machine,last_slot+2:last_slot_active[machine]+2,:] = complete_solution[machine,last_slot+1:last_slot_active[machine]+1,:]
        else:
            max_time_per_machine[machine] += durations[j,ope,machine]
        start = max(complete_solution[machine, last_slot, 3], current_finish_time[j])
        end = start + durations[j,ope,machine]
        complete_solution[machine, last_slot + 1] = np.array([j,ope,start,end])
        current_ope[j] += 1
        last_slot_active[machine] += 1
        max_time = max(max_time, end)
        current_finish_time[j] = end

    return complete_solution


def interpret_min(solution, n_jobs, n_machines, n_ope, n_total_ope, durations):
    complete_solution = np.full((n_machines,  int(max(n_ope)*2), 4), -1, dtype=np.int)
    complete_solution[:, :, 2:] = np.zeros((n_machines, int(max(n_ope)*2), 2), dtype=np.int)
    current_ope = np.zeros(n_jobs, dtype=np.int)
    current_finish_time = np.zeros(n_jobs, dtype=np.int)
    last_slot_active = np.full(n_machines, -1, dtype=np.int)
    max_time = 0
    max_time_per_machine = np.zeros(n_machines, dtype=np.int)
    for j in solution:
        ope = current_ope[j]
        mask = durations[j, ope, :] != -1
        possible_machines = np.nonzero(mask)[0]
        costs = durations[j, ope, :][mask]
        possible_slots = np.stack((possible_machines, last_slot_active[possible_machines]), axis=-1)
        possible_timespan = np.maximum(max_time_per_machine[possible_machines] + costs, max_time)
        selected = np.nonzero(possible_timespan == possible_timespan.min())
        selected_slots = possible_slots[selected]
        machine, last_slot = selected_slots[np.argmax(durations[j, ope, :][selected_slots[:, 0]])]

        max_time_per_machine[machine] += durations[j, ope, machine]
        start = max(complete_solution[machine, last_slot, 3], current_finish_time[j])
        end = start + durations[j, ope, machine]
        complete_solution[machine, last_slot + 1] = np.array([j, ope, start, end])
        current_ope[j] += 1
        last_slot_active[machine] += 1
        max_time = max(max_time, end)
        current_finish_time[j] = end

    return complete_solution


def interpret_base(solution, n_jobs, n_machines, n_ope, n_total_ope, durations):
    complete_solution = np.full((n_machines, int(max(n_ope)*2), 4), -1, dtype=np.int)
    complete_solution[:, :, 2:] = np.zeros((n_machines, int(max(n_ope)*2), 2), dtype=np.int)
    current_ope = np.zeros(n_jobs, dtype=np.int)
    current_finish_time = np.zeros(n_jobs, dtype=np.int)
    last_slot_active = np.full(n_machines, -1, dtype=np.int)
    max_time_per_machine = np.zeros(n_machines, dtype=np.int)
    for j in solution:
        ope = current_ope[j]
        mask = durations[j, ope, :] != -1
        possible_machines = np.nonzero(mask)[0]
        possible_slots = np.stack((possible_machines, last_slot_active[possible_machines]), axis=-1)
        selected = np.argmin(max_time_per_machine[possible_machines])
        machine, last_slot = possible_slots[selected]

        max_time_per_machine[machine] += durations[j, ope, machine]
        start = max(complete_solution[machine, last_slot, 3], current_finish_time[j])
        end = start + durations[j, ope, machine]
        complete_solution[machine, last_slot + 1] = np.array([j, ope, start, end])
        current_ope[j] += 1
        last_slot_active[machine] += 1
        current_finish_time[j] = end

    return complete_solution


def interpret_base_exact(solution, n_jobs, n_machines, n_ope, n_total_ope, durations):
    complete_solution = np.full((n_machines, int(max(n_ope)*2), 4), -1, dtype=np.int16)
    complete_solution[:, :, 2:] = np.zeros((n_machines, int(max(n_ope)*2), 2), dtype=np.int16)
    current_ope = np.zeros(n_jobs, dtype=np.int16)
    current_finish_time = np.zeros(n_jobs, dtype=np.int16)
    last_slot_active = np.full(n_machines, -1, dtype=np.int16)
    max_time_per_machine = np.zeros(n_machines, dtype=np.int16)
    for j in solution:
        ope = current_ope[j]
        mask = durations[j, ope, :] != -1
        possible_machines = np.nonzero(mask)[0]
        possible_slots = np.stack((possible_machines, last_slot_active[possible_machines]), axis=-1)
        selected = np.nonzero(max_time_per_machine[possible_machines] == max_time_per_machine[possible_machines].min())[0][-1]
        machine, last_slot = possible_slots[selected]

        max_time_per_machine[machine] += durations[j, ope, machine]
        start = max(complete_solution[machine, last_slot, 3], current_finish_time[j])
        end = start + durations[j, ope, machine]
        complete_solution[machine, last_slot + 1] = np.array([j, ope, start, end])
        current_ope[j] += 1
        last_slot_active[machine] += 1
        current_finish_time[j] = end

    return complete_solution


def interpret_python(factory, sol_encoded):
    sol = sol_encoded+1
    ope_job = {i: 0 for i in range(1, factory.n_jobs + 1)}
    solution = {m: [] for m in range(1, factory.n_mach + 1)}

    time_job = {i + 1: 0 for i in range(factory.n_jobs)}  # Times until which each job is currently being done
    time_mach = {i + 1: 0 for i in range(factory.n_mach)}  # Times until which each machine is currently working

    for job in sol:
        machines = [m for m in range(1, factory.n_mach + 1) if (job, ope_job[job] + 1, m) in factory.p]

        mach = machines[0]
        time_m = time_mach[mach]
        for m in range(1, len(machines)):
            if time_mach[machines[m]] <= time_m:
                mach = machines[m]
                time_m = time_mach[mach]

        # Computing the start and end dates
        te = max(time_job[job], time_mach[mach])
        ts = te + factory.p[(job, ope_job[job] + 1, mach)]

        # Updating
        ope_job[job] += 1
        solution[mach].append((job, ope_job[job], te, ts))
        time_job[job] = ts
        time_mach[mach] = ts
    return solution


def interpret_python_greedy(factory, sol_encoded):
    sol = sol_encoded + 1
    ope_job = {i: 0 for i in range(1,factory.n_jobs+1)}
    solution = {m: [] for m in range(1,factory.n_mach+1)}

    time_job = {i+1: 0 for i in range(factory.n_jobs)}  # Times until which each job is currently being done
    time_mach = {i+1: 0 for i in range(factory.n_mach)}  # Times until which each machine is currently working

    for job in sol:
        n_ope = ope_job[job] + 1
        durations = []
        for i in range(1,factory.n_mach+1):
            if (job, n_ope, i) in factory.p:
                durations.append(factory.p[(job, n_ope, i)])
            else:
                durations.append(-1)
        slots = [(m, time_mach[m], -1) for m in range(1,factory.n_mach+1) if(job, n_ope, m) in factory.p]
        for m, liste_ope in solution.items():
            if durations[m-1] != -1:
                slots += [(m, ts, i) for i, (_, _, _, ts) in enumerate(liste_ope[:len(liste_ope) - 1]) if liste_ope[i + 1][2] >= durations[m-1] + max(ts,time_job[job])]

        b_mach, b_ts, b_i = slots[0]
        for mach, ts, i in slots[1:]:
            if ts <= b_ts:
                b_ts = ts
                b_mach = mach
                b_i = i

        # Computing the start and end dates
        te = max(time_job[job], b_ts)
        ts = te + durations[b_mach-1]

        # Updating
        ope_job[job] += 1
        if b_i == -1:
            solution[b_mach].append((job, n_ope, te, ts))
        else:
            solution[b_mach].insert(b_i+1, (job, n_ope, te, ts))
        time_job[job] = ts
        time_mach[b_mach] = max(ts, time_mach[b_mach])
    return solution


#################################
# Evaluation
#################################
def evaluation(complete_solution):
    if interpret_mode <= 3:
        return complete_solution[:,:,3].max()
    else:
        return max([complete_solution[m][-1][-1] for m in range(1,len(complete_solution)+1) if len(complete_solution[m])>0])


#################################
# Restitution de la solution
#################################
def decode(complete_solution):
    if interpret_mode <= 3:
        result = dict()
        for m, list_ope in enumerate(complete_solution):
            a = []
            for j, o, s, e in list_ope:
                if j == -1:
                    break
                a.append((j+1, o+1, s, e))
            result[m+1] = a
        return result
    else:
        return complete_solution


#################################
# Initialisations
#################################
def init_base(n_jobs, n_ope):
    solution = []
    for i in range(n_jobs):
        solution += [i] * n_ope[i]
    return np.array(solution, dtype=np.int)


def init_random(n_jobs, n_ope, n_machines, n_total_ope, durations, factory, n):
    base_sol = init_base(n_jobs, n_ope)
    best_sol = base_sol.copy()
    np.random.shuffle(best_sol)
    best_score = evaluation(interpret(best_sol, n_jobs, n_machines, n_ope, n_total_ope, durations, factory))
    for i in range(n-1):
        sol = base_sol.copy()
        np.random.shuffle(sol)
        score = evaluation(interpret(sol, n_jobs, n_machines, n_ope, n_total_ope, durations, factory))
        if score < best_score:
            best_sol = sol
            best_score = score
    return best_sol, best_score


def init_ACO(n_jobs, n_ope, n_machines, n_total_ope, durations, factory, n, n_ants, t_max=3.0, t_min=1.0, rho=0.2, alpha=1, beta=1):
    best_score = 100000
    best_sol = None
    tau = np.full((n_total_ope, n_jobs), t_max, dtype=np.float)
    for i in range(n):
        current_best_sol = None
        current_best_score = 100000
        worst_score = 0
        for ant in range(n_ants):
            sol = np.zeros(n_total_ope, dtype=np.int)

            current_ope = np.zeros(n_jobs, dtype=np.int)
            time_job = np.zeros(n_jobs, dtype=np.int)
            time_machine = np.zeros(n_machines, dtype=np.int)

            for var in range(n_total_ope):
                open_jobs = np.nonzero(current_ope < n_ope)[0]
                heuristic = np.zeros(len(open_jobs), dtype=np.int)
                machine_delta = []
                for k, j in enumerate(open_jobs):
                    mask = durations[j, current_ope[j], :] != -1
                    possible_machines = np.nonzero(mask)[0]
                    costs = durations[j, current_ope[j], :][mask]
                    total_costs = costs + np.maximum(time_machine[possible_machines], time_job[j])
                    index = np.argmin(total_costs)
                    heuristic[k] = total_costs[index]
                    machine_delta.append((possible_machines[index], total_costs[index]))

                heuristic_val = (heuristic / max(heuristic)) ** beta
                ph = tau[var, open_jobs] ** alpha
                probas = ph * heuristic_val
                probas /= sum(probas)

                index = np.random.choice(len(open_jobs), 1, p=probas)
                job = open_jobs[index]
                sol[var] = job

                current_ope[job] += 1
                time_job[job] = heuristic[index]
                i,j = machine_delta[index[0]]
                time_machine[i] = j

            score = evaluation(interpret(sol, n_jobs, n_ope, n_machines, n_total_ope, durations, factory))
            if score > worst_score:
                worst_score = score

            if score < current_best_score:
                current_best_sol = sol.copy()
                current_best_score = score

        tau = (1 - rho) * tau
        for var,job in enumerate(current_best_sol):
            tau[var,job] = max(min(tau[var,job] + worst_score/current_best_score,t_max),t_min)

        if current_best_score < best_score:
            best_score = current_best_score
            best_sol = current_best_sol.copy()

    assert (np.bincount(best_sol) == n_ope).all(), "Not correct number of jobs : " + str(np.bincount(best_sol) ) + " et " + str(n_ope)
    return best_sol


#################################
# Fonctions de recherche
#################################
def GVNS(n_jobs, n_machines, n_ope, n_total_ope, durations, factory, s, score, k_max, t0, max_time):
    k = 1
    n_tot_iter = 0
    while k <= k_max:
        s_prime = shake(s, k)
        score_prime = evaluation(interpret(s_prime, n_jobs, n_machines, n_ope, n_total_ope, durations, factory))
        s_prime_prime, score_prime, n_iter = VND(n_jobs, n_machines, n_ope, n_total_ope, durations, factory, s_prime, score_prime, k_max, t0, max_time) #BestImprovement(factory,k, s_prime)
        n_tot_iter += n_iter
        s, score, k = neighborhood_change(s, score, s_prime_prime, score_prime, k)
        if t.time()-t0 >= max_time:
            return s, score, n_tot_iter
    return s, score, n_tot_iter


def VND(n_jobs, n_machines, n_ope, n_total_ope, durations, factory, s, score, k_max, t0, max_time):
    k = 1
    n_iter = 0
    while k<=k_max:
        n_iter += 1
        s_prime, score_prime = best_improvement(n_jobs, n_machines, n_ope, n_total_ope, durations, factory, s, score, k, t0, max_time)
        s, score, k = neighborhood_change(s, score, s_prime, score_prime, k)
        if t.time()-t0 >= max_time :
            return s, score, n_iter
    return s, score, n_iter


def neighborhood_change(s, score, s_prime, score_prime, k):
    if score_prime < score:
        s = s_prime # make a move
        score = score_prime
        k = 1 # initial neighborhood
    else:
        k += 1 # next neighborhood
    return s, score, k


def shake(solution, k):
    # Exchange +Insert+ Exchange
    s_prime = solution.copy()
    p = np.random.choice(len(s_prime), 2, replace=False)
    if k == 1:
        s_prime = swap_single(s_prime, min(p[0],len(s_prime)-2))
    elif k == 2:
        s_prime = interchange_single(s_prime, p[0], p[1])
    elif k == 3:
        s_prime = insertion_single(s_prime, p[0], p[1])
    else :
        raise Exception("number of beinghbor exceeded")
    
    return s_prime


def best_improvement(n_jobs, n_machines, n_ope, n_total_ope, durations, factory, solution, score, k, t0, max_time):
    if k == 1:
        # voisinage 1 : permuter les opérations côte à côte entre elles (2-swap)
        s_prime, score = swap(n_jobs, n_machines, n_ope, n_total_ope, durations, factory, solution, score, t0, max_time)
        #print("swap")
    elif k == 2:
        s_prime, score = interchange(n_jobs, n_machines, n_ope, n_total_ope, durations, factory, solution, score, t0, max_time)
        #print("interchange")
    elif k == 3:
        s_prime, score = insertion(n_jobs, n_machines, n_ope, n_total_ope, durations, factory, solution, score, t0, max_time)
        #print("insertion")
    # autre idée exchange mais que les 2 trucs à côté
    else :
        raise Exception("number of neighborhood exceeded")
    return s_prime, score

#################################
# Fonctions de mouvement et voisinage
#################################
def swap(n_jobs, n_machines, n_ope, n_total_ope, durations, factory, solution, score, t0, max_time):
    for i in range(len(solution)-1):
        new_sol = solution.copy()
        new_sol[i], new_sol[i + 1] = new_sol[i + 1], new_sol[i]
        new_sol_score = evaluation(interpret(new_sol, n_jobs, n_machines, n_ope, n_total_ope, durations, factory))
        if new_sol_score < score :
            score = new_sol_score
            solution = new_sol
        if t.time()-t0 >= max_time :
            return solution, score
    return solution, score


def swap_single(solution, i):
    solution[i], solution[i+1] = solution[i+1], solution[i]
    return solution


def interchange(n_jobs, n_machines, n_ope, n_total_ope, durations, factory, solution, score, t0, max_time):
    for i in range(len(solution)):
        for j in range(i):
            new_sol = solution.copy()
            new_sol[i], new_sol[j] = new_sol[j], new_sol[i]
            new_sol_score = evaluation(interpret(new_sol, n_jobs, n_machines, n_ope, n_total_ope, durations, factory))
            if new_sol_score < score:
                score = new_sol_score
                solution = new_sol
            if t.time()-t0 >= max_time:
                return solution, score
    return solution, score


def interchange_single(solution, i, j):
    solution[i], solution[j] = solution[j], solution[i]
    return solution


def insertion(n_jobs, n_machines, n_ope, n_total_ope, durations, factory, solution, score, t0, max_time):
    for i in range(len(solution)):
        for j in range(len(solution)):
            if i != j:
                new_sol = solution.copy()
                val = new_sol[i]
                new_sol = np.delete(new_sol, i)
                np.insert(new_sol, j, val)
                new_sol_score = evaluation(interpret(new_sol, n_jobs, n_machines, n_ope, n_total_ope, durations, factory))
                if new_sol_score < score:
                    score = new_sol_score
                    solution = new_sol
                if t.time()-t0 >= max_time:
                    return solution, score
    return solution, score


def insertion_single(solution, i, j):
    val = solution[i]
    solution = np.delete(solution, i)
    np.insert(solution, j, val)
    return solution
