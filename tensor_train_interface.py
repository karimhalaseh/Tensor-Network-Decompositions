#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np
import numpy.linalg as lg
import numpy.random as rm
from tensorly.tenalg import multi_mode_dot
from scipy.stats import ortho_group

# Possible TODOs: 
# - change functionality so you can make A orthogonal, B not etc.
# - use tensorly for everything
# - generalize to m > 3
# - change it so that rank is found, rather than inputted (should be pretty easy,
#   just take number of non-zero eigenvalues)

class TTInterface(metaclass = abc.ABCMeta):
    """
    Interface for tensor train decompositions
    """
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'create_tensor') and  callable(subclass.create_tensor)
                and hasattr(subclass, 'decompose') and callable(subclass.decompose)
                and hasattr(subclass, 'sum_of_slices') and callable(subclass.sum_of_slices)
                and hasattr(subclass, 'verify_decomp') and callable(subclass.verify_decomp)
                and hasattr(subclass, 'test') and callable(subclass.test))
    
    
    @abc.abstractmethod
    def create_tensor(self):
        raise NotImplementedError
        
        
    @abc.abstractmethod
    def decompose(self):
        raise NotImplementedError
    
    
    @abc.abstractmethod
    def verify_decomp(self):
        raise NotImplementedError
        
        
    @abc.abstractmethod
    def test(self):
        raise NotImplementedError
        
        
    def sum_of_slices(self, T):
        """
        helper function taking arbitrary weighted sums of slices of T 
        for modes associated with L the "left-end" tensor in the train, and R 
        the "right-end" tensor in the train respectively
    
        Parameters: tensor to take sums of slices from T
        
        Returns: Weighted sum of slices S_L from first two modes of T, weighted
        sum of slices S_R from other two modes of T 
        """
        n_A = T.shape[0]
        n_B = T.shape[1]
        n_C = T.shape[2]
        n_D = T.shape[3]
        
        weights_L = rm.rand(n_C, n_D) # sampled from Unif[0,1] distribution
        weights_R = rm.rand(n_A, n_B)
        
        S_L = np.zeros((n_A, n_B))
        for i_3 in range(n_C):
            for i_4 in range(n_D):
                S_L += weights_L[i_3, i_4] * T[:, :, i_3, i_4]
        
        S_R = np.zeros((n_C, n_D))
        for i_1 in range(n_A):
            for i_2 in range(n_B):
                S_R += weights_R[i_1, i_2] * T[i_1, i_2, :, :]
                
        return S_L, S_R