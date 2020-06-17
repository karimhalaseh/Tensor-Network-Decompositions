#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as lg
from scipy.stats import ortho_group
from tensor_train_interface import TTInterface

class Orthog_Decomp(TTInterface):
    def create_tensor(self, lammy, ABC, mu, DEF):
        """
        artificially creates 4-way tensor T that decomposes into two odeco 
        3-way tensors L and R.
        
        Parameters: lists of matrices ABC and DEF, where ABC has matrices A, B, C 
        with orthogonal columns, of column size r_L, and R has matrices D, E, F
        with orthogonal columns, of column size r_R. For example, ABC[1][:,0] 
        represents the vector b_0, and must be orthogonal to all other b_i for 
        i in range(1, r_L). Last matrix in each list (i.e. C and F) will be 
        contracted to create 4-way tensor T, and so must have the same row size
        as each other. 
        
        i.e. L = lammy[0]*(a_0)x(b_0)x(c_0) + ...
                    + lammy[r_L - 1]*(a_{r_L - 1})x(b_{r_L - 1})x(c_{r_L - 1})
        
        Returns: the L and R tensors, and their contraction T
        """
        # create odeco tensors
        rank_L = lammy.shape[0]
        
        if (rank_L != ABC[0].shape[1] or rank_L != ABC[1].shape[1] 
            or rank_L != ABC[2].shape[1]):
            print("Rank of L inconsistent")
            return 
        
        L = np.zeros((ABC[0].shape[0], ABC[1].shape[0], ABC[2].shape[0]))
        
        for i in range(rank_L):
            L_i = np.tensordot(ABC[0][:,i], ABC[1][:,i], axes = 0)
            L_i = np.tensordot(L_i, ABC[2][:,i], axes = 0)
            L_i *= lammy[i]
            L += L_i
        
        rank_R = mu.shape[0]
        
        if (rank_R != DEF[0].shape[1] or rank_R != DEF[1].shape[1] 
            or rank_R != DEF[2].shape[1]):
            print("Rank of R inconsistent")
            return 
        
        R = np.zeros((DEF[0].shape[0], DEF[1].shape[0], DEF[2].shape[0]))
        
        for j in range(rank_R):
            R_j = np.tensordot(DEF[0][:,j], DEF[1][:,j], axes = 0)
            R_j = np.tensordot(R_j, DEF[2][:,j], axes = 0)
            R_j *= mu[j]
            R += R_j
        
        if (L.shape[2] != R.shape[2]):
            print("Dimensions of C and F must be equal")
            return
        
        # contract along one dimension to produce 4-way T
        T = np.tensordot(L, R, axes = (2, 2))
        
        return T, L, R


    def sinkhorn(self, X, tolerance = 10e-16, max_iter = 10000):
        """
        function that performs adapted sinkhorn algorithm on r_Lxr_R X. 
        
        Parameters: matrix with diag-orth-diag decomposition X, tolerance 
        sets when convergence is deemed to have occurred.
        
        Returns positive diagonal r_Lxr_L lammy, r_Rxr_R mu, and r_Lxr_R 
        Q with orthogonal rows and columns
        """
        rank_L = X.shape[0]
        rank_R = X.shape[1]
        
        ones_L = np.ones(rank_L)
        ones_R = np.ones(rank_R)
        
        lammy = ones_L
        mu = ones_R
        Q = X
        
        for i in range(max_iter):
            col_sum = np.sum(Q, axis = 0)
            row_sum = np.sum(Q, axis = 1)
            
            if (np.all(np.abs(col_sum - ones_R) < tolerance)
                      and np.all(np.abs(row_sum - ones_L) < tolerance)):
                break
            
            elif max_iter - i <= 1:
                print("Sinkhorn convergence not reached")
                print("\n")
                return
            
            Q = Q / col_sum
            mu = mu / col_sum

            Q = Q / row_sum[:, None]
            lammy = lammy / row_sum
        
        return lammy, Q, mu
        

    def decompose(self, T, rank_L, rank_R):
        """
        uses SVD of sum of slices and Sinkhorn's theorem to compute decomposition 
        of T into two orthogonal tensors L and R.

        Parameters: tensor to decompose T, ranks of L and R.
        
        Returns: weight values and orthonormal vectors of decomposition
        """
        # take weighted sums of slices of L's dimensions and R's dimensions in T
        S_L, S_R = self.sum_of_slices(T)
        
        #perform SVD on sum of slices to obtain A, B, E, F
        SVD_AB = lg.svd(S_L)
        A = SVD_AB[0][:, :rank_L]
        B = SVD_AB[2][:rank_L, :].T
        
        SVD_EF = lg.svd(S_R)
        D = SVD_EF[0][:, :rank_R]
        E = SVD_EF[2][:rank_R, :].T
        
        # contract to obtain lammy <c, f> mu matrix (dod for diag-orth-diag)
        dod = np.empty(shape = (rank_L, rank_R)) 
        
        for i in range(rank_L):
            for j in range(rank_R):
                dod_ij = np.tensordot(T, A[:, i], axes=(0, 0))
                dod_ij = np.tensordot(dod_ij, B[:, i], axes=(0, 0))
                dod_ij = np.tensordot(dod_ij, D[:, j], axes=(0, 0))
                dod[i,j] = np.tensordot(dod_ij, E[:, j], axes=(0, 0))
        
        lammy, Q, mu = self.sinkhorn(dod**2)
        
        Q = np.sign(dod) * (Q**0.5)
        lammy = lammy**0.5
        mu = mu**0.5
        
        n = max(rank_L, rank_R)
        
        if n == rank_R:
            F = np.eye(n)
            C = Q.T
        
        else:
            C = np.eye(n)
            F = Q
            
        ABC = (A, B, C)
        DEF = (D, E, F)
        
        return lammy, ABC, mu, DEF


    def verify_decomp(self, lammy, ABC, mu, DEF, T, threshold = 10e-10):
        """
        verifies decomposition is correct by reconstructing the tensor T from the
        decomposition and measuring relative error.
        
        Parameters: lammy and ABC represent the weights and list of vector matrices
        A, B and C of the symmetric decomposition of L, and similarly, mu and DEF 
        represent this for R. T is the original tensor. threshold measures the 
        maximum relative error such that we return true.
        
        Returns: True if the constructed tensor is equal to T within the relative
        error threshold, False otherwise.
        """     
        T_ver, L_ver, R_ver = self.create_tensor(lammy, ABC, mu, DEF)
        
        # measure relative error and return binary result
        rel_error = lg.norm(T - T_ver) / lg.norm(T)
        print("Relative error = ")
        print(rel_error)
        print("\n")
        if(rel_error < threshold):
            return True
        
        return False

        
    def test(self):
        pass


# RUN TESTS
lammy = np.ones(2)
A = ortho_group.rvs(2)
B = ortho_group.rvs(2)
C = ortho_group.rvs(2)

mu = np.ones(2)
D = ortho_group.rvs(2)
E = ortho_group.rvs(2)
F = ortho_group.rvs(2)

test2 = Orthog_Decomp()
T, L, R = test2.create_tensor(lammy, (A, B, C), mu, (D, E, F))
lammy, ABC, mu, DEF = test2.decompose(T, 2, 2)
print(test2.verify_decomp(lammy, ABC, mu, DEF, T))