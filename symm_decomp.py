import numpy as np
import numpy.linalg as lg
import numpy.random as rm
from tensorly.tenalg import multi_mode_dot
from scipy.stats import ortho_group
from tensor_train_interface import TT2Interface
import warnings

class Symm_Decomp(TT2Interface):
    """
    Class for symmetric orthogonal tensor train decompositions of length 2, as
    in "Orthogonal Tensor Network Decompositions" (Halaseh, Muller, Robeva),
    Section 3
    """
    def create_tensor(self, lammy, U, mu, V):
        """
        artificially creates nxnxnxn tensor T that decomposes into two symmetric 
        nxnxn tensors A and B. U, V should have orthonormal of full-rank
        normalized columns.

        Parameters
        ----------
        lammy : 1d array
            weight vector of A of size rank_A
        U : 2d array
            n x rank_A matrix such that each column is vector in decomp. of A, 
            i.e. A = lammy[0]*(u_0)**3 + ... + lammy[r_A - 1]*(u_{r_A - 1})**3 
        mu : 1d array
            weight vector of B of size rank_B
        V : 2d array
            n x rank_B matrix such that each column is vector in decomp. of B,
            i.e. B = mu[0]*(v_0)**3 + ... + mu[r_B - 1]*(v_{r_B - 1})**3

        Raises
        ------
        ValueError
            If inputs are of incompatible dimensions

        Returns
        -------
        T : 4d array
            4-way partially symmetric tensor 
        A : 3d array
            3-way symmetric tensor as left core of T
        B : 3d array
            3-way symmetric tensor as right core of T
        """
        # create symmetric 3-way tensors
        n = U.shape[0]
        
        if n != V.shape[0]:
            raise ValueError("Error: symmetric dimensions of A and B must match.")
        
        rank_A = lammy.shape[0]
        
        if rank_A != U.shape[1]:
            raise ValueError("""Error: number of eigenvalues of A must equal 
                             number of eigenvectors""")
        
        A = np.zeros((n, n, n))
        
        for u in range(rank_A):
            A_i = np.tensordot(U[:,u], U[:,u], axes = 0)
            A_i = np.tensordot(A_i, U[:,u], axes = 0)
            A_i *= lammy[u]
            A += A_i
        
        rank_B = mu.shape[0]
        
        if rank_B != V.shape[1]:
            raise ValueError("""Error: number of eigenvalues of B must equal 
                             number of eigenvectors""")
                             
        if (rank_A > n) or (rank_B > n):
            warnings.warn("""Ranks should not be greater than dimension for symmetric
                          orthogonal decomposition""", stacklevel = 2)
        
        B = np.zeros((n, n, n))
        
        for v in range(rank_B):
            B_i = np.tensordot(V[:,v], V[:,v], axes = 0)
            B_i = np.tensordot(B_i, V[:,v], axes = 0)
            B_i *= mu[v]
            B += B_i
        
        # contract along one dimension to produce 4-way T
        T = np.tensordot(A, B, axes = (0, 2))
        
        return T, A, B

        
    def decompose(self, T, orthogonal = True, max_iter = 200, 
                  W_A = np.eye(2), W_B = np.eye(2), whitened = False): 
        """
        uses method analogous to Kolda's to compute decomposition of T into two 
        symmetric tensors A and B.

        Parameters
        ----------
        T : 4d array
            Tensor to decompose
        orthogonal : bool, optional
            if True, T should be an odeco tensor train. The default is True.
            Whitening applied if false.
        max_iter : int, optional
            Number of iterations whitening algorithm executes to find psd 
            matrices before declaring failure. The default is 200.

        Raises
        ------
        ValueError
            If T is not a 4-way tensor of equal dimensions

        Returns
        -------
        lammy : 1d array
            weight vector of 3-way symmetric left core of T
        U : 2d array
            columns make up symmetric vectors in decomposition of left core of T.
        mu : 1d array
            weight vector of 3-way symmetric left core of T
        V : 2d array
            columns make up symmetric vectors in decomposition of right core of T.
        """
        if (whitened == True):
            warnings.warn("""W_A, W_B, whitened parameters should only be used in 
                          internal recursive call""", stacklevel = 2)
                          
        # check T is a 4-way partially symmetric tensor
        if len(T.shape) != 4:
            raise ValueError("T must be a 4-way tensor")
        
        if (T.shape[0] != T.shape[1]) or (T.shape[2] != T.shape[3]):
            raise ValueError("T must be a partially symmetric tensor")
            
        if (whitened == False) and (T.shape[0] != T.shape[2]):
            raise ValueError("T must be a partially symmetric tensor")
            
        # take weighted sums of slices of A's dimensions and B's dimensions in T
        S_A, S_B = self.sum_of_slices(T)
        
        if orthogonal == True:
            # compute eigendecomposition of weighted sums of slices
            vals_A, U = lg.eigh(S_A)
            absvals_A = np.abs(vals_A)
            idx = np.argsort(absvals_A) # find index of sorted absolute eigvals
            # no. nonzero eigvals = rank_A
            rank_A = np.shape(S_A)[0] - np.searchsorted(absvals_A[idx], 10e-10) 
            U = (U[:, idx])[:, -rank_A:] # take rank_A highest corresp. eigenvectors
    
            vals_B, V = lg.eigh(S_B)
            absvals_B = np.abs(vals_B)
            idx = np.argsort(absvals_B)
            rank_B = np.shape(S_B)[0] - np.searchsorted(absvals_B[idx], 10e-10)
            V = (V[:, idx])[:, -rank_B:] # take rank_B highest corresp. eigenvectors
            
            # dewhiten if necessary
            if whitened == True: # only true for internal recursive call
                U_og = lg.pinv(W_A) @ U 
                V_og = lg.pinv(W_B) @ V
                
            else:
                U_og = U
                V_og = V
            
            # obtain matrix of products of "eigenvalues" by contracting with eigenvectors
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
            # Take sums of slices until psd matrices of ranks A and B are found
            for k in range(max_iter):
                if (np.all(lg.eigvalsh(S_A) > -10e-10) and 
                    np.all(lg.eigvalsh(S_B) > -10e-10)):
                    break
                
                elif max_iter - k <= 1:
                    print(""""Max iterations reached for psd sum of slices 
                          associated with A\n""")
                    return
                
                S_A, S_B = self.sum_of_slices(T)
                
            # take "skinny" eigendecompositions of these psd matrices        
            D_A, U_A = lg.eigh(S_A)
            rank_A = np.shape(S_A)[0] - np.searchsorted(D_A, 10e-10) 
            U_A = np.flip(U_A[:, -rank_A:], axis = 1) # rank_A highest corresp. eigenvectors
            D_A = np.diag(np.flip(D_A[-rank_A:])) # diagonal of rank_A highest eigenvalues
            
            D_B, U_B = lg.eigh(S_B)
            rank_B = np.shape(S_B)[0] - np.searchsorted(D_B, 10e-10) 
            U_B = np.flip(U_B[:, -rank_B:], axis = 1) # rank_B highest corresp. eigenvectors
            D_B = np.diag(np.flip(D_B[-rank_B:])) # diagonal of rank_B highest eigenvalues   
            
            # produce whitening matrices 
            W_A = lg.inv(D_A**0.5) @ U_A.T
            W_B = lg.inv(D_B**0.5) @ U_B.T
            
            # take the tensor-matrix product of W_A along A's modes and W_B along B's modes
            T_bar = multi_mode_dot(T, (W_A, W_A, W_B, W_B), modes = (0, 1, 2, 3))
            
            # apply orthogonal decomposition to whitened tensor T_bar
            return self.decompose(T_bar, W_A = W_A, W_B = W_B, whitened = True)
    
    
    def verify_decomp(self, lammy, U, mu, V, T, threshold = 10e-10, verbose = False):  
        """
        verifies decomposition is correct by reconstructing the tensor T from the
        decomposition and measuring relative error.

        Parameters
        ----------
        lammy : 1d array
            "eigenvalues" of symmetric decomposition of left core of T
        U : 2d array
            columns are "eigenvectors" of symmetric decomposition of right core 
            of T
        mu : 1d array
            "eigenvalues" of symmetric decomposition of right core of T
        V : 2d array
            columns are "eigenvectors" of symmetric decomposition of right core 
            of T
        T : 4d array
            Tensor to compare with
        threshold : float, optional
            Max relative error for success. The default is 10e-10.
        verbose : bool, optional
            If True, prints out relative error. The default is False.

        Returns
        -------
        bool
            If True, decomposition accurate within relative error threshold 
        """
        T_ver, A_ver, B_ver = self.create_tensor(lammy, U, mu, V)
        
        # measure relative error and return binary result
        rel_error = lg.norm(T - T_ver) / lg.norm(T)
        
        if verbose:
            print("Relative error = ")
            print(rel_error)
            print("\n")
        
        return (rel_error < threshold)
    
    
    def test(self, n, r_A, r_B, orthogonal = True, iters = 100, threshold = 10e-10, 
             noise = 0, verbose = False):
        """
        tests decomposition algorithm. by generating iters number of tensors, 
        running decompose (with whitening if orthogonal = False), and printing
        relative error for each iteration as well as overall number of successes,
        number of failures, and number that failed to find psd sums of slices if 
        whitening was applied.
    
        Parameters: dimension size of tensors to generate n, ranks of A and B 
        to generate r_A, r_B, orthogonal toggles whether orthogonal matrices are
        generated, or simply full rank matrices where whitening is applied. 
        Iters controls number of tensors to generate, threshold controls max 
        relative error which is still a success, noise controls range of uniform
        distribution sampled from to add to tensor

        Parameters
        ----------
        n : int
            size of symmetric dimensions of tensor to generate
        r_A : int
            rank of left core of tensor
        r_B : int
            rank of right core of tensor
        orthogonal : bool, optional
            If True, generates tensor with symmetric orthogonal "eigenvectors". 
            Otherwise whitening applid to symmetric "eigenvectors". The default 
            is True.
        iters : int, optional
            Number of tests to run. The default is 100.
        threshold : float, optional
            Max relative error for successful test. The default is 10e-10.
        noise : float, optional
            noise parameter to scale normally distributed noise tensor added
            to original T. The default is 0.
        verbose : bool, optional
            If True, prints out relative error for each test. The default is 
            False.

        Returns
        -------
        None.

        """
        success = 0
        psd_not_found = 0
        
        if orthogonal == True: # generate orthogonal matrices to create tensors from
            for i in range(iters):
                
                T, A, B = self.create_tensor(
                            # randomly sample eigenvalues from integers from 1-20
                            rm.randint(low = 1, high = 20, size = r_A), 
                            # randomly sample orthogonal normalized vectors
                            ortho_group.rvs(n)[:, :r_A], 
                            rm.randint(low = 1, high = 20, size = r_B), 
                            ortho_group.rvs(n)[:, :r_B])
                
                # add noise tensor
                if noise > 0:
                    N = rm.normal(size = np.shape(T))
                    T += noise * (lg.norm(T) / lg.norm(N)) * N
                
                # run decomposition
                lammy, U, mu, V = self.decompose(T)
                
                # verify
                succ = self.verify_decomp(lammy, U, mu, V, T, threshold, verbose)
                
                if succ:
                    success += 1
        
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
                
                if noise > 0:
                    N = rm.normal(size = np.shape(T))
                    T += noise * (lg.norm(T) / lg.norm(N)) * N
                
                # try-catch block to catch when psd sums of slices not found
                try:
                    lammy, U, mu, V = self.decompose(T, orthogonal = False)
                
                except(TypeError):
                    psd_not_found += 1
                    continue
                
                succ = self.verify_decomp(lammy, U, mu, V, T, threshold, verbose)
                
                if succ:
                    success += 1
                    
            print("Number of psd not found = %d" % psd_not_found)
        
        print("Number of successes: %d" % success)