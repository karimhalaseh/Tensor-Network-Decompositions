#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np
import numpy.random as rm

# Possible TODOs: 
# - change it so that rank is found, rather than inputted (just take number of non-zero eigenvalues)

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
        n_1 = T.shape[0]
        n_2 = T.shape[1]
        n_3 = T.shape[2]
        n_4 = T.shape[3]
        
        weights_L = rm.rand(n_3, n_4) # sampled from Unif[0,1] distribution
        weights_R = rm.rand(n_1, n_2)
        
        S_L = np.zeros((n_1, n_2))
        for i_3 in range(n_3):
            for i_4 in range(n_4):
                S_L += weights_L[i_3, i_4] * T[:, :, i_3, i_4]
        
        S_R = np.zeros((n_3, n_4))
        for i_1 in range(n_1):
            for i_2 in range(n_2):
                S_R += weights_R[i_1, i_2] * T[i_1, i_2, :, :]
                
        return S_L, S_R