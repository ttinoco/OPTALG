#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2017, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import time
import numpy as np
from numpy.linalg import norm
from .problem_ms import StochProblemMS

class Node:
    """
    Scenario tree node class.
    """

    def __init__(self,w,p,parent=None,id=0):
        """
        Scenario tree node class.

        Parameters
        ----------
        w : random vector
        p : probability
        parent : Node
        id : int
        """
 
        # Save
        self.w = w
        self.p = p
        self.children = []
        self.parent = parent
        self.id = id
        self.data = None

        # Check
        assert(0 <= p <= 1)

    def add_child(self,child):
        self.children.append(child)

    def get_p(self):
        return self.p

    def get_id(self):
        return self.id

    def get_data(self):
        return self.data

    def get_ancestors(self):
        if self.is_root():
            return []
        else:
            p = self.get_parent()
            return p.get_ancestors()+[p]

    def get_w(self):
        return self.w

    def get_parent(self):
        return self.parent

    def get_child(self,i):
        return self.children[i]

    def get_children(self):
        return self.children

    def get_num_children(self):
        return len(self.children)

    def get_leafs(self):
        if self.is_leaf():
            return [self]
        else:
            return sum([c.get_leafs() for c in self.children],[])

    def get_descendants(self):
        return self.children+sum([c.get_descendants() for c in self.children],[])

    def is_leaf(self):
        return not self.children

    def is_root(self):
        return self.parent is None

    def set_data(self,data):
        self.data = data

    def show(self):
        pass

class StochProblemMS_Tree:
    """
    Scenario tree class.
    """

    def __init__(self,problem,branching_factors,cluster=False,num_samples=1000,seed=None):
        """
        Scenario tree class.
        
        Parameters
        ----------
        problem : StochProblemMS
        branching_factors : list (length T-1)
        cluster : {True,False}
        num_samples : int
        seed : int
        """

        # Init data
        self.problem = problem
        self.branching_factors = branching_factors
        self.cluster = cluster
        self.num_samples = num_samples
        self.construction_time = 0.

        if seed is not None:
            np.random.seed(seed)
            
        if cluster:
            from sklearn.cluster import k_means
      
        T = problem.get_num_stages()
        
        # Check
        assert(isinstance(problem,StochProblemMS))
        assert(len(branching_factors) == T-1)
        assert(all([f > 0 for f in branching_factors]))

        # Construct
        t0 = time.time()
        self.root = Node(problem.sample_w(0,[]), # realization
                         1.,                     # probability
                         id=0)                   # id
        counter = 1
        nodes = [self.root]
        for t in range(1,T):
            new_nodes = []
            for node in nodes:
                observations = [n.get_w() for n in node.get_ancestors()+[node]]
                if cluster:
                    w_array = np.array([problem.sample_w(t,observations) for i in range(num_samples)])
                    assert(w_array.shape[0] == num_samples)
                    clusters = k_means(w_array,branching_factors[t-1])
                    assert(clusters[0].shape[0] == branching_factors[t-1])
                    assert(clusters[1].size == num_samples)
                for i in range(branching_factors[t-1]):
                    if cluster:
                        w = clusters[0][i,:]
                        p = float(np.sum(clusters[1] == i))/float(clusters[1].size)
                    else:
                        w = problem.sample_w(t,observations)
                        p = 1./float(branching_factors[t-1])
                    node.add_child(Node(w,p,parent=node,id=counter))
                    counter += 1
                assert(np.abs(sum(map(lambda n: n.get_p(),node.get_children()))-1.) < 1e-12)
                new_nodes += node.get_children()
            nodes = new_nodes

        # Check
        num_curr = 1
        num_nodes = 1
        for t in range(1,T):
            num_curr = branching_factors[t-1]*num_curr
            num_nodes += num_curr
        assert(num_nodes == counter)
        assert(num_nodes == len(self.get_nodes()))

        # Time
        self.construction_time = time.time()-t0

    def check_branch(self,branch):

        if len(branch) > 0:
            assert(not branch[0].get_parent())
            assert(all([branch[i+1] in branch[i].get_children() for i in range(len(branch)-1)]))
            assert(all([branch[i] is branch[i+1].get_parent() for i in range(len(branch)-1)]))
            if len(branch) == self.problem.get_num_stages():
                assert(not branch[-1].get_children())

    def get_nodes(self):
        
        return [self.root]+self.root.get_descendants()

    def get_stage_nodes(self,t):

        T = self.problem.get_num_stages()
        
        assert(0 <= t < T)
        nodes = [self.root]
        for i in range(t):
            new_nodes = []
            for n in nodes:
                new_nodes += n.get_children()
            nodes = new_nodes
        return nodes

    def get_root_node(self):
        
        return self.root

    def get_leaf_nodes(self):

        return self.root.get_leafs()
        
    def sample_branch(self,t):

        assert(0 <= t < self.problem.get_num_stages())
        
        node = self.root
        for tau in range(t):
            node = node.get_child(np.random.randint(0,node.get_num_children()))            
        branch = node.get_ancestors()+[node]
        assert(len(branch) == t+1)
        self.check_branch(branch)

        return branch

    def get_closest_branch(self,sample):

        assert(len(sample) <= self.problem.get_num_stages())
        
        branch = []
        nodes = [self.root]
        for t in range(len(sample)):
            branch.append(nodes[np.argmin([norm(sample[t]-n.get_w(),np.inf) for n in nodes])])
            nodes = branch[-1].get_children()
        assert(len(branch) == len(sample))
        self.check_branch(branch)
            
        return branch
 
    def show(self):
        
        print('\nScenario Tree')
        print('-------------')
        print('branching factors : %s' %self.branching_factors)
        print('cluster           : %r' %self.cluster)
        print('num samples       : %d' %self.num_samples)
        print('num scenarios     : %d' %len(self.get_leaf_nodes()))
        print('num nodes         : %d' %len(self.get_nodes()))
        print('construction time : %.2f min' %(self.construction_time/60.)) 

    def draw(self,node_size=40):

        if len(self.get_nodes()) > 10000:
            return

        import matplotlib.pyplot as plt
        import networkx as nx

        G = nx.DiGraph()
        for node in self.get_nodes():
            for child in node.get_children():
                G.add_edge(node.get_id(),child.get_id())
        plt.figure()
        pos = nx.graphviz_layout(G,prog='dot')
        nx.draw(G,pos,with_labels=False,arrows=False,node_size=node_size)
