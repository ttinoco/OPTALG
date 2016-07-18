#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2016, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import numpy as np
from numpy.linalg import norm

class Node:

    def __init__(self,w,parent=None,id=0):
        self.w = w
        self.children = []
        self.parent = parent
        self.id = id
        self.data = None

    def add_child(self,child):
        self.children.append(child)

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
            return sum(map(lambda c: c.get_leafs(),self.children),[])

    def get_descendants(self):
        return self.children+sum(map(lambda c: c.get_descendants(),self.children),[])

    def is_leaf(self):
        return not self.children

    def is_root(self):
        return self.parent is None

    def set_data(self,data):
        self.data = data

    def show(self):

        print '\nNode    :',self.id
        print 'Children:',map(lambda c: c.get_id(),self.children)

        map(lambda c: c.show(),self.children)

class StochProblemMS_Tree:

    def __init__(self,problem,branching_factor,branching_type,seed=None):
        """
        Creates scenario tree for multistage
        stochastic optimization problem.
        
        Parameters
        ----------
        problem : StochProblemMS
        branching_factor : int
        branching_type : {'uniform','decreasing'}
        seed : int
        """

        self.problem = problem
        self.branching_factor = branching_factor
        self.branching_type = branching_type

        if seed is not None:
            np.random.seed(seed)
      
        T = problem.get_num_stages()
       
        if branching_type == 'uniform':
            factor_list = [branching_factor for i in range(T-1)] 
        elif branching_type == 'decreasing':
            factor_list = [max([branching_factor-i,1]) for i in range(T-1)]
        else:
            raise ValueError('invalid branching type')
        assert(len(factor_list) == T-1)
        assert(all([factor_list[i] > 0 for i in range(len(factor_list))]))
 
        self.root = Node(problem.sample_w(0,[]),id=0)
        counter = 1
        nodes = [self.root]
        for t in range(1,T):
            new_nodes = []
            for node in nodes:
                for i in range(factor_list[t-1]):
                    w = problem.sample_w(t,map(lambda n: n.get_w(),node.get_ancestors()+[node]))
                    node.add_child(Node(w,node,id=counter))
                    counter += 1
                new_nodes += node.get_children()
            nodes = new_nodes
        num_nodes = 1
        num_curr = 1
        for t in range(1,T):
            num_curr = factor_list[t-1]*num_curr
            num_nodes += num_curr
        assert(num_nodes == counter)
        assert(num_nodes == len(self.get_nodes()))

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
        
        """
        t = len(sample)-1
        nodes = self.get_stage_nodes(t)
        branches = []
        for n in nodes:
            branch = n.get_ancestors()+[n]
            assert(len(branch) == len(sample))
            branches.append(branch)
        sample_vec = np.hstack(sample)
        branches_vec = map(lambda b: np.hstack(map(lambda n: n.get_w(),b)),branches)
        return branches[np.argmin(map(lambda b: norm(b-sample_vec),branches_vec))]
        """

        nodes = [self.root]
        branch = []
        for t in range(len(sample)):
            branch.append(nodes[np.argmin(map(lambda n: norm(sample[t]-n.get_w(),np.inf),nodes))])
            nodes = branch[-1].get_children()
        assert(len(branch) == len(sample))
        self.check_branch(branch)
            
        return branch
 
    def show(self):
        
        self.root.show()
        
        print '\nLeafs:',map(lambda n: n.get_id(),self.get_leaf_nodes())
        print '\nScenarios:'
        for node in self.get_leaf_nodes():
            print map(lambda n: n.get_id(),node.get_ancestors()+[node])

    def draw(self):

        if len(self.get_nodes()) > 1000:
            return

        import matplotlib.pyplot as plt
        import networkx as nx

        G = nx.DiGraph()
        for node in self.get_nodes():
            for child in node.get_children():
                G.add_edge(node.get_id(),child.get_id())
        plt.figure()
        pos = nx.graphviz_layout(G,prog='dot')
        nx.draw(G,pos,with_labels=False,arrows=False,node_size=10.)
