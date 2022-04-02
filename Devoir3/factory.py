import matplotlib
from matplotlib.axis import XAxis, YAxis
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import random as r

class Factory():

    def __init__(self,filename):
        """Creates an instance of the problem
        :param filename: The data file used to build the instance
        """
        self.f = filename

        with open(filename, 'r') as f:
            lines = f.readlines()
        
        line = lines[0].split()

        self.n_jobs = int(line[0])  # Number of jobs
        self.n_mach = int(line[1])  # Number of machines
        self.p = dict()             # Keys are tuple (job,operation,machine) and values are the corresponding processing times
        self.n_ope = dict()         # Keys are the jobs and values are the number of operations the have
        
        for job in range(1,len(lines[1:])+1):
            line = lines[job].split()
            curr=0
            self.n_ope[job]=int(line[0])
            for ope in range(1,int(line[0])+1):
                curr+=1
                for _ in range(int(line[curr])):
                    curr+=1
                    self.p[(job,ope,int(line[curr]))]=int(line[curr+1])
                    curr+=1

    def get_total_cost(self,solution):
        """Return the ending time of the schedule
        :param solution: a dictionnary where the keys are the machines and the values are the ordered list of tuples (job, operation, start, end) 
                         treated by the machines
        :return: the ending time"""

        return max([solution[m][-1][-1] for m in range(1,self.n_mach+1) if len(solution[m])>0])

    def verify_solution(self,solution):
        """Returns True if a solution is valid, thus 
        1) if all operations are treated and only by machines which actually can
        2) if there isn't any overlap
        3) if precedence constraints are respected
        4) Starting dates and ending dates are correct (start + processing time = end)
        5) The data has not been changed during the process
        :param solution: a dictionnary where the keys are the machines and the values are the ordered list of tuples (job, operation, start, end) 
                         treated by the machines
        :return: True if the solution is valid"""
        
        # Criterion 1
        ope = set()
        for m in solution:
            for (j,o,_,_) in solution[m]:
                assert (j,o,m) in self.p, "Opération traitée sur une machine incapable de le faire"
                ope.add((j,o,m))
        
        assert len(ope) == sum(self.n_ope.values()), "Opération(s) non assignée(s)"

        # Criterion 2
        for m in solution:
            for j in range(len(solution[m])-1):
                assert solution[m][j][3] <= solution[m][j+1][2], "Solution comportant un overlap"
        
        # Criterion 3
        for job in range(1,self.n_jobs+1):
            for ope in range(1,self.n_ope[job]):
                t1 = [solution[m][i][3] for m in solution if len(solution[m])>0 for i in range(len(solution[m])) if solution[m][i][0]==job and solution[m][i][1]==ope][0]
                t2 = [solution[m][i][2] for m in solution if len(solution[m])>0 for i in range(len(solution[m])) if solution[m][i][0]==job and solution[m][i][1]==ope+1][0]
                assert t1<=t2, "Contraintes de précédence non respectées"
        
        # Criterion 4
        for m,mach in solution.items():
            for ope in mach:
                assert ope[2] + self.p[(ope[0],ope[1],m)] == ope[3], "Dates de début et de fin incohérentes"
        
        # Criterion 5
        verif = Factory(self.f)
        assert self.n_jobs == verif.n_jobs and self.n_mach == verif.n_mach and self.n_ope == verif.n_ope and self.p == verif.p, "Données d'instance modifiées pendant le processus"

        return True

    def save_solution(self, solution, filename):
        """Saves the solution as a txt file.
        :param solution: a dictionnary where the keys are the machines and the values are the ordered list of tuples (job, operation, start, end) 
                         treated by the machines
        :param filename: the file in which to write the solution
        :param dates: Also saves the corresponding starting dates if this boolean is True."""

        with open(filename,'w') as f:
            f.write("%s %s\n%s\n\n" % (self.n_jobs,self.n_mach,self.get_total_cost(solution)))
            for m in range(1,self.n_mach+1):
                for ope in solution[m]:
                    f.write("(%s,%s,%s,%s) " % (ope[0],ope[1],ope[2],ope[3]))
                f.write("\n")

        return
        
    def display_solution(self, solution, filename):
        """Displays a solution as a png
        :param solution: a dictionnary where the keys are the machines and the values are the ordered list of tuples (job, operation, start, end) 
                         treated by the machines
        :param filename: the file in which to draw the solution"""
        tasks_df = pd.DataFrame()

        for m,mach in solution.items():
            for ope in mach:
                new_task = {"Ressource": "tâche %s" % ope[0],
                            "Start": ope[2],
                            "Finish": ope[3],
                            "Task": "M" + str(m)}
                tasks_df = tasks_df.append(new_task, ignore_index=True)
        
        fig = ff.create_gantt(tasks_df,
                    colors=[(r.random(),r.random(),r.random()) for _ in range(len(tasks_df))],
                    index_col='Ressource', 
                    reverse_colors=True,
                    show_colorbar=True,
                    group_tasks=True,
                    showgrid_x=True,
                    showgrid_y=True,
                    title="Date de complétion (makespan) : %s" % self.get_total_cost(solution),
                    width=1600)
        fig.update_layout(xaxis_type='linear', font_size=20)
        fig.update_traces(mode='lines', line_color='black', selector=dict(fill='toself'))
        #for trace in fig.data:
        #    trace.x += (trace.x[0], trace.x[0], trace.x[0])
        #    trace.y += (trace.y[-5], trace.y[0], trace.y[-1])
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

        fig.write_image(filename)
        return