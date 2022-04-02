import random as r

def solve(factory):
    """
    A random feasible solution of the problem
    :param factory: object describing the input
    
    :return: a dictionnary where the keys are the machines and the values are the ordered list of tuples (job, operation, start, end) 
             treated by the machines
    """

    ope_job = {i:0 for i in range(1, factory.n_jobs+1)}
    solution = {m:[] for m in range(1,factory.n_mach+1)}

    time_job = {i+1:0 for i in range(factory.n_jobs)}  # Times until which each job is currently being done
    time_mach = {i+1:0 for i in range(factory.n_mach)} # Times until which each machine is currently working

    while ope_job != factory.n_ope: 
        # Picking a random job to assign on a random machine
        job = r.sample([j for j in range(1,factory.n_jobs+1) if ope_job[j]<factory.n_ope[j]],1)[0]
        mach = r.sample([m for m in range(1,factory.n_mach+1) if (job,ope_job[job]+1,m) in factory.p],1)[0]
        
        # Computing the start and end dates
        te = max(time_job[job],time_mach[mach])
        ts = te + factory.p[(job,ope_job[job]+1,mach)]
        
        # Updating
        ope_job[job]+=1
        solution[mach].append((job,ope_job[job],te,ts))
        time_job[job] = ts
        time_mach[mach] = ts

    return solution