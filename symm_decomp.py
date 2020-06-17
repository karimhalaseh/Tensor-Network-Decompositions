#!/usr/bin/env python3
# -*- coding: utf-8 -*-'

import numpy as np
import numpy.linalg as lg
import numpy.random as rm
from tensorly.tenalg import multi_mode_dot
from scipy.stats import ortho_group
from tensor_train_interface import TTInterface

class Symm_Decomp(TTInterface):
    def create_tensor(self, lammy, U, mu, V):
        """
        artificially creates nxnxnxn tensor T that decomposes into two symmetric 
        nxnxn tensors A and B.
        
        Parameters: pass (orthogonal or full rank with normalized columns)
        nxr_A and nxr_B matrices U and V respectively, such that U[:,i] = u_i 
        (V[:,i] = v_i) in the decomposition of rank-r_A (r_B) tensor A (B).
        lammy represents weight vector of A, mu represents weight vector of B,
        i.e. A = lammy[0]*(u_0)**3 + ... + lammy[r_A - 1]*(u_{r_A - 1})**3 
        (B = mu[0]*(v_0)**3 + ... + mu[r_B - 1]*(v_{r_B - 1})**3).
        
        Returns: the A and B tensors, and their contraction along a dimension T.
        """
        # create symmetric 3-way tensors
        n = U.shape[0]
        
        if n != V.shape[0]:
            print("Error: symmetric dimensions of A and B must match.")
            return
        
        rank_A = lammy.shape[0]
        
        if rank_A != U.shape[1]:
            print("Error: number of eigenvalues of A must equal number of eigenvectors")
            return       
        
        A = np.zeros((n, n, n))
        
        for u in range(rank_A):
            A_i = np.tensordot(U[:,u], U[:,u], axes = 0)
            A_i = np.tensordot(A_i, U[:,u], axes = 0)
            A_i *= lammy[u]
            A += A_i
        
        rank_B = mu.shape[0]
        
        if rank_B != V.shape[1]:
            print("Error: number of eigenvalues of B must equal number of eigenvectors")
            return 
        
        B = np.zeros((n, n, n))
        
        for v in range(rank_B):
            B_i = np.tensordot(V[:,v], V[:,v], axes = 0)
            B_i = np.tensordot(B_i, V[:,v], axes = 0)
            B_i *= mu[v]
            B += B_i
        
        # contract along one dimension to produce 4-way T
        T = np.tensordot(A, B, axes = (0, 2))
        
        return T, A, B

        
    def decompose(self, T, rank_A, rank_B, orthogonal = True, max_iter = 200, 
                  W_A = np.eye(2), W_B = np.eye(2), whitened = False): 
        # DO NOT USE W_A, W_B, whiten parameters in public call
        # W_A, W_B, whiten only used within whitening algorithm in recursive call
        """
        uses method analogous to Kolda's to compute decomposition of T into two 
        symmetric tensors A and B.
        
        if orthogonal = False, whitening is applied (assuming full rank decomposition).
        
        Parameters: tensor to decompose T, ranks of A and B, default assumes 
        orthogonal decomposition exists. If not, max_iter sets maximum iterations 
        until algorithm fails in finding positive semi-definite sum of slices when 
        whitening. 
        
        **  Internal Parameters: W_A and W_B are the whitening matrices, 
        **  whitened = True means a whitened tensor has been passed in and 
        **  dewhitening will take place
        
        Returns: "eigenvalues" and "eigenvectors" of decomposition
        """
        # take weighted sums of slices of A's dimensions and B's dimensions in T
        S_A, S_B = self.sum_of_slices(T)
        
        if orthogonal == True:
            # compute eigendecomposition of weighted sums of slices
            vals_A, U = lg.eigh(S_A)
            idx = np.abs(vals_A).argsort() # find index of sorted absolute eigvals
            U = (U[:, idx])[:, -rank_A:] # take rank_A highest corresp. eigenvectors
    
            vals_B, V = lg.eigh(S_B)
            idx = np.abs(vals_B).argsort()
            V = (V[:, idx])[:, -rank_B:] # take rank_B highest corresp. eigenvectors
            
            # dewhiten if necessary
            if whitened == True: # only true for internal recursive call
                U_og = lg.pinv(W_A) @ U 
                V_og = lg.pinv(W_B) @ V
                
            else:
                U_og = U
                V_og = V
            
            # obtain matrix of products of"eigenvalues" by contracting with eigenvectors
            eig_products = np.empty(shape = (rank_A, rank_B)) 
            
            for u in range(rank_A):
                for v in range(rank_B):
                    C = np.tensordot(T, U[:,u], axes=(0, 0))
                    C = np.tensordot(C, U[:,u], axes=(0, 0))
                    C = np.tensordot(C, V[:,v], axes=(0, 0))
                    C = np.tensordot(C, V[:,v], axes=(0, 0))
                    eig_products[u,v] = C / (U_og[:,u] @ V_og[:,v])
            
            # obtain "eigenvalues" by performing SVD on this rank-1 products matrix
            SVD = lg.svd(eig_products)
            
            # obtained up to scaling; absorb singular values into lambda WLOG
            lammy = SVD[0][:,0] * SVD[1][0]
            mu = SVD[2][0,:]
            
            return lammy, U_og, mu, V_og
            
            
        else: # apply whitening
            # Take weighted sums of slices of each side until psd matrices of ranks A and B are found
            for k in range(max_iter):
                if (np.all(lg.eigvalsh(S_A) > -10e-10) and (lg.matrix_rank(S_A) == rank_A)
                    and np.all(lg.eigvalsh(S_B) > -10e-10) and (lg.matrix_rank(S_B) == rank_B)):
                    break
                
                elif max_iter - k <= 1:
                    print("Max iterations reached for psd sum of slices associated with A," 
                          " or rank of sum of slices not equal to rank of A")
                    print("\n")
                    return
                
                S_A, S_B = self.__sum_of_slices(T)
                
            # take "skinny" eigendecompositions of these psd matrices        
            D_A, U_A = lg.eigh(S_A) 
            U_A = np.flip(U_A[:, -rank_A:], axis = 1) # rank_A highest corresp. eigenvectors
            D_A = np.diag(np.flip(D_A[-rank_A:])) # diagonal of rank_A highest eigenvalues
            
            D_B, U_B = lg.eigh(S_B)
            U_B = np.flip(U_B[:, -rank_B:], axis = 1) # rank_B highest corresp. eigenvectors
            D_B = np.diag(np.flip(D_B[-rank_B:])) # diagonal of rank_B highest eigenvalues   
            
            # produce whitening matrices 
            W_A = lg.inv(D_A**0.5) @ U_A.T
            W_B = lg.inv(D_B**0.5) @ U_B.T
            
            # take the tensor-matrix product of W_A along A's dimensions and W_B along B's dimensions
            T_bar = multi_mode_dot(T, (W_A, W_A, W_B, W_B), modes = (0, 1, 2, 3))
            
            # apply orthogonal decomposition to whitened tensor T_bar
            return self.decompose(T_bar, rank_A, rank_B, W_A = W_A, W_B = W_B, whitened = True)
    
    
    def verify_decomp(self, lammy, U, mu, V, T, threshold = 10e-10):  
        """
        verifies decomposition is correct by reconstructing the tensor T from the
        decomposition and measuring relative error.
        
        Parameters: lammy and U represent the "eigenvalues" and "eigenvectors" 
        of the symmetric decomposition of A, and similarly, mu and V represent 
        this for B. T is the original tensor. threshold measures the maximum relative
        error such that we return true.
        
        Returns: True if the constructed tensor is equal to T within the relative
        error threshold, False otherwise.
        """     
        T_ver, A_ver, B_ver = self.create_tensor(lammy, U, mu, V)
        
        # measure relative error and return binary result
        rel_error = lg.norm(T - T_ver) / lg.norm(T)
        print("Relative error = ")
        print(rel_error)
        print("\n")
        if(rel_error < threshold):
            return True
        
        return False
    
    
    def test(self, n, r_A, r_B, orthogonal = True, iters = 100, threshold = 10e-10):
        """
        tests decomposition algorithm, by generating iters number of tensors, 
        running decompose (with whitening if orthogonal = False), and printing
        relative error for each iteration as well as overall number of successes,
        number of failures, and number that failed to find psd sums of slices if 
        whitening was applied.
    
        Parameters: dimension size of tensors to generate n, ranks of A and B 
        to generate r_A, r_B, orthogonal toggles whether orthogonal matrices are
        generated, or simply full rank matrices where whitening is applied. 
        Iters controls number of tensors to generate, threshold controls max 
        relative error which is still a success. 
        """
        success = 0
        fail = 0
        psd_not_found = 0
        
        if orthogonal == True: # generate orthogonal matrices to create tensors from
            for i in range(iters):
                
                T, A, B = self.create_tensor(
                            rm.randint(low = 1, high = 20, size = r_A), # randomly sample eigenvalues from integers from 1-20
                            ortho_group.rvs(n)[:, :r_A], # randomly sample orthogonal normalized vectors
                            rm.randint(low = 1, high = 20, size = r_B), 
                            ortho_group.rvs(n)[:, :r_B])
                
                lammy, U, mu, V = self.decompose(T, r_A, r_B)
                
                if self.verify_decomp(lammy, U, mu, V, T, threshold):
                    success += 1
                    
                else:
                    fail += 1
        
        else: # generate full rank matrices to create tensors where whitening is called
            for i in range(iters):
                # generate matrices from Unif[-1, 1]
                X_A = rm.uniform(low = -1, high = 5, size = (n, r_A))
                X_B = rm.uniform(low = -1, high = 5, size = (n, r_B))
                
                # ensure these matrices are full rank
                while (lg.matrix_rank(X_A) != r_A):
                    X_A = rm.uniform(low = -1, high = 5, size = (n, r_A))
                    
                while (lg.matrix_rank(X_B) != r_B):
                    X_B = rm.uniform(low = -1, high = 5, size = (n, r_B))
                
                # normalize matrices
                X_A = X_A / lg.norm(X_A, axis = 0)
                X_B = X_B / lg.norm(X_B, axis = 0)
                    
                T, A, B = self.create_tensor(
                                    rm.randint(low = 1, high = 20, size = r_A), X_A, 
                                    rm.randint(low = 1, high = 20, size = r_B), X_B)
                
                # try-catch block to catch when psd sums of slices not found
                try:
                    lammy, U, mu, V = self.decompose(T, r_A, r_B, 
                                                       orthogonal = False)
                
                except(TypeError):
                    psd_not_found += 1
                    continue
                
                if self.verify_decomp(lammy, U, mu, V, T, threshold):
                    success += 1
                    
                else:
                    fail += 1
        
        print("Number of successes: %d" % success)
        print("Number of failures: %d" % fail)
        print("Number of psd not found = %d (only for whitening)" % psd_not_found)


# RUN TESTS
test1 = Symm_Decomp()
test1.test(n = 25, r_A = 4, r_B = 7, orthogonal = False) 
print(issubclass(Symm_Decomp, DecompInterface))
print(issubclass(Orthog_Decomp, DecompInterface))