import matplotlib.pyplot as plt
import networkx as nx

class MotherCard():

    def __init__(self,filename):

        self.graph = nx.MultiDiGraph()
        
        with open(filename,'r') as f:
            lines = f.readlines()
        
        n = int(lines[0])

        for i in range(n):
            l1 = lines[2+i].split()
            l2 = lines[3+n+i].split()

            for j in range(n):
                self.graph.add_edge(i,j, flow=int(l1[j]), dist=int(l2[j]))
    
        self.n_components = self.graph.number_of_nodes()

    def get_total_cost(self,solution):
        return sum(self.graph[solution[i]][solution[j]][0]['dist']*self.graph[i][j][0]['flow'] for i in range(self.graph.number_of_nodes()) for j in range(self.graph.number_of_nodes()))

    def verify_solution(self,solution):
        assert (i in solution for i in range(self.graph.number_of_nodes())), "Solution invalide"
        return True
    
    def save_solution(self, solution, filename):
        with open(filename,'w') as f:
            f.write("%s\n" % self.graph.number_of_nodes())
            for i in solution:
                f.write("%s " % i)
        return

    def lab(self,i,j,solution):
        if i==j:
            return ""
        else:
            return "%s*%s + %s*%s" % (self.graph[solution[i]][solution[j]][0]['dist'],self.graph[i][j][0]['flow'],self.graph[solution[j]][solution[i]][0]['dist'],self.graph[j][i][0]['flow'])

    def display_solution(self, solution, filename):
        nx.draw_networkx_edge_labels(self.graph, pos=nx.spring_layout(self.graph,seed=10), edge_labels={(i,j):self.lab(i,j,solution) for i in range(self.graph.number_of_nodes()) for j in range(self.graph.number_of_nodes())})
        nx.draw_networkx(self.graph, pos=nx.spring_layout(self.graph,seed=10), with_labels=True, labels={i:solution[i] for i in self.graph.nodes})
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, labelleft=False, left=False)
        plt.title('solution cost: ' + str(self.get_total_cost(solution)), size=15, color='red')
        plt.savefig(filename)
        return