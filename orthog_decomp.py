#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as lg
import numpy.random as rm
from scipy.stats import ortho_group
from tensor_train_interface import TTInterface
from dod import DOD, sinkhorn, tandem_procrustes

class Orthog_Decomp(TTInterface):
    """
    Class for Orthogonal Tensor Train Decompositions of Length 2
    
    WARNING: Diagonal-Orthogonal-Diagonal decomposition integral to 
    orthogonal tensor train decomposition only currently working for the 
    square case. Thus n_CF = rank_L = rank_R only works
    """
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
        
        Throws: ValueError if any inputs have incompatible shapes.
        """
        # create odeco tensors
        rank_L = lammy.shape[0]
        
        if (rank_L != ABC[0].shape[1] or rank_L != ABC[1].shape[1] 
            or rank_L != ABC[2].shape[1]):
            raise ValueError("Rank of L inconsistent")
        
        L = np.zeros((ABC[0].shape[0], ABC[1].shape[0], ABC[2].shape[0]))
        
        for i in range(rank_L):
            L_i = np.tensordot(ABC[0][:,i], ABC[1][:,i], axes = 0)
            L_i = np.tensordot(L_i, ABC[2][:,i], axes = 0)
            L_i *= lammy[i]
            L += L_i
        
        rank_R = mu.shape[0]
        
        if (rank_R != DEF[0].shape[1] or rank_R != DEF[1].shape[1] 
            or rank_R != DEF[2].shape[1]):
            raise ValueError("Rank of R inconsistent")
        
        R = np.zeros((DEF[0].shape[0], DEF[1].shape[0], DEF[2].shape[0]))
        
        for j in range(rank_R):
            R_j = np.tensordot(DEF[0][:,j], DEF[1][:,j], axes = 0)
            R_j = np.tensordot(R_j, DEF[2][:,j], axes = 0)
            R_j *= mu[j]
            R += R_j
        
        if (L.shape[2] != R.shape[2]):
            raise ValueError("Dimensions of C and F must be equal")
        
        # contract along one dimension to produce 4-way T
        T = np.tensordot(L, R, axes = (2, 2))
        
        return T, L, R
    

    def decompose(self, T, rank_L, rank_R, dod_alg = "sinkhorn", max_iter = 1000):
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
        X = np.empty(shape = (rank_L, rank_R)) 
        
        for i in range(rank_L):
            for j in range(rank_R):
                X_ij = np.tensordot(T, A[:, i], axes=(0, 0))
                X_ij = np.tensordot(X_ij, B[:, i], axes=(0, 0))
                X_ij = np.tensordot(X_ij, D[:, j], axes=(0, 0))
                X[i,j] = np.tensordot(X_ij, E[:, j], axes=(0, 0))
                
        X_dod = DOD()
        L, C, F, M, iters, conv = DOD.decompose(X_dod, X, method = dod_alg, max_iter = max_iter)
        
        lammy = np.diag(L)
        mu = np.diag(M)
        
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

        
    def test(self, n_A, n_B, n_CF, n_D, n_E, rank_L, rank_R, iters = 1, threshold = 10e-10, dod_alg = "sinkhorn"):
        """
        tests orthogonal decomposition algorithm
        
        Parameters: Dimensions of A, B, C & F (which must be equal), D, E, the ranks
        of the left and right 3-tensors (which must be equal to n_CF for the algorithm
        to work currently), iters sets number of tests to generate, threshold sets 
        maximum relative error such that a solution is declared successful
        """
        success = 0
        
        for i in range(iters):
            # set up test
            lammy = rm.randint(low = 1, high = 20, size = rank_L)
            A = ortho_group.rvs(n_A)[:, :rank_L]
            B = ortho_group.rvs(n_B)[:, :rank_L]
            C = ortho_group.rvs(n_CF)[:, :rank_L]
            
            mu = rm.randint(low = 1, high = 20, size = rank_R)
            D = ortho_group.rvs(n_A)[:, :rank_R]
            E = ortho_group.rvs(n_B)[:, :rank_R]
            F = ortho_group.rvs(n_CF)[:, :rank_R]
            
            T, L, R = self.create_tensor(lammy, (A, B, C), mu, (D, E, F))
            
            # decompose
            lammy_sol, ABC_sol, mu_sol, DEF_sol = self.decompose(T, 2, 2, dod_alg = dod_alg)
            
            if self.verify_decomp(lammy_sol, ABC_sol, mu_sol, DEF_sol, T, threshold):
                success += 1
            
        print("Number of successes: %d" % success)
        
        
# RUN TESTS
test1 = Orthog_Decomp()
test1.test(n_A = 2, n_B = 2, n_CF = 2, n_D = 2, n_E = 2, rank_L = 2, rank_R = 2, iters = 1)

# # set up test
# lammy = rm.randint(low = 1, high = 20, size = 2)
# A = ortho_group.rvs(2)
# B = ortho_group.rvs(2)
# C = ortho_group.rvs(2)

# mu = rm.randint(low = 1, high = 20, size = 2)
# D = ortho_group.rvs(2)
# E = ortho_group.rvs(2)
# F = ortho_group.rvs(2)

# print("Lambda: ")
# print(lammy)
# print("\n")
            
# print("mu: ")
# print(mu)
# print("\n")
            
# print("lambda * mu: ")
# print(np.diag(lammy) @ np.diag(mu))
# print("\n")

# print("A: ")
# print(A)
# print("\n")

# print("B: ")
# print(B)
# print("\n")
    
# print("C: ")
# print(C)
# print("\n")

# print("D: ")
# print(D)
# print("\n")

# print("E: ")
# print(E)
# print("\n")

# print("F: ")
# print(F)
# print("\n")

# print("C * F: ")
# print(C @ F)
# print("\n")

# T, L, R = test1.create_tensor(lammy, (A, B, C), mu, (D, E, F))

# # decompose
# lammy_sol, ABC_sol, mu_sol, DEF_sol = test1.decompose(T, 2, 2)
# A_sol, B_sol, C_sol = ABC_sol
# D_sol, E_sol, F_sol = DEF_sol

# print("lammy_sol: ")
# print(lammy_sol)
# print("\n")

# print("mu_sol: ")
# print(mu_sol)
# print("\n")        

# print("lammy_sol * mu_sol: ")
# print(np.diag(lammy_sol) @ np.diag(mu_sol))
# print("\n")

# print("A_sol: ")
# print(A_sol)
# print("\n")
    
# print("B_sol: ")
# print(B_sol)
# print("\n")
    
# print("C_sol: ")
# print(C_sol)
# print("\n")

# print("D_sol: ")
# print(D_sol)
# print("\n")
    
# print("E_sol: ")
# print(E_sol)
# print("\n")
    
# print("F_sol: ")
# print(F_sol)
# print("\n")
            
# print("C_sol * F_sol: ")
# print(C_sol @ F_sol)
# print("\n")

# # print("C found orthogonal? ")
# # print(C_sol @ C_sol.T)
# # print("\n")

# # print("F found orthogonal? ")
# # print(F_sol.T @ F_sol)
# # print("\n")

# test1.verify_decomp(lammy_sol, ABC_sol, mu_sol, DEF_sol, T)