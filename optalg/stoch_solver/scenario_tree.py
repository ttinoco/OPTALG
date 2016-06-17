#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2016, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import numpy as np

class Node:

    def __init__(self,w,parent=None,id=0):
        self.w = w
        self.children = []
        self.parent = parent
        self.id = id

    def add_child(self,child):
        self.children.append(child)

    def get_id(self):
        return self.id

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

    def show(self):

        print '\nNode    :',self.id
        print 'Children:',map(lambda c: c.get_id(),self.children)

        map(lambda c: c.show(),self.children)

class ScenarioTree:

    def __init__(self,problem,branching_factor,seed=None):
        """
        Creates scenario tree for multistage
        stochastic optimization problem.
        
        Parameters
        ----------
        problem : 
        branching_factor :
        seed :
        """

        self.problem = problem
        self.branching_factor = branching_factor

        if seed is not None:
            np.random.seed(seed)
      
        T = problem.get_num_stages()
        
        self.root = Node(problem.sample_w(0,[]),id=0)
        counter = 1
        nodes = [self.root]
        for t in range(1,T):
            new_nodes = []
            for node in nodes:
                for i in range(branching_factor):
                    w = problem.sample_w(t,map(lambda n: n.get_w(),node.get_ancestors()+[node]))
                    node.add_child(Node(w,node,id=counter))
                    counter += 1
                new_nodes += node.get_children()
            nodes = new_nodes
        num_nodes = sum([branching_factor**tau for tau in range(T)])
        print num_nodes,len(self.get_nodes())
        assert(num_nodes == counter)
        assert(num_nodes == len(self.get_nodes()))

    def get_nodes(self):
        
        return [self.root]+self.root.get_descendants()

    def get_root_node(self):
        
        return self.root

    def get_leaf_nodes(self):

        return self.root.get_leafs()
        
    def sample_branch(self,t):

        assert(0 <= t < self.problem.get_num_stages())
        
        node = self.root
        for tau in range(t):
            node = node.get_child(np.random.randint(0,node.get_num_children()))            
        return node.get_ancestors()+[node]
            
    def show(self):
        
        self.root.show()
        
        print '\nLeafs:',map(lambda n: n.get_id(),self.get_leaf_nodes())
        print '\nScenarios:'
        for node in self.get_leaf_nodes():
            print map(lambda n: n.get_id(),node.get_ancestors()+[node])
        
        
