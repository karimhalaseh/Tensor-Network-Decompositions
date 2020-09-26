import numpy as np
import numpy.linalg as lg
import numpy.random as rm
from scipy.stats import ortho_group
from tensor_train_interface import TT2Interface
import dodd
import warnings

class Orthog_Decomp(TT2Interface):
    """
    Class for Orthogonal Tensor Train Decompositions of Length 2 as in 
    "Orthogonal Tensor Network Decompositions" (Halaseh, Muller, Robeva),
    Section 5
    """
    def create_tensor(self, lammy, ABC, mu, DEF):
        """
        creates 4-way tensor T that is decomposable into two odeco 3-way 
        tensors L and R.
        
        Parameters
        ----------
        lammy : 1d array
            lambda[i] is the weight of the i-th term of outer products taken 
            to produce the left tensor L. The size of lambda is rank_L.
        ABC : tuple of 3 2d arrays
            each array corresponds to matrices A, B, C in that order which 
            should have orthogonal columns. Number of columns should all be 
            equal (to the rank, rank_L). For example, ABC[1][:,0] represents 
            the vector b_0, and must be orthogonal to all other b_i for i from 
            1 to r_l inclusive. Last matrix C will be contracted with F from
            DEF to create 4-way tensor T, and so must have the same column size
            as F.
        mu : 1d array
            mu[j] is the weight of the j-th term of outer products taken 
            to produce the right tensor R. The size of mu is rank_R. 
        DEF : tuple of 3 2d arrays
            each array corresponds to matrices D, E, F in that order which 
            should have orthogonal columns. Number of columns should all be 
            equal (to the rank, rank_R). Last matrix F will be contracted with
            C to create 4-way tensor T, and so must have the same column size
            as F.

        Raises
        ------
        ValueError
            If any inputs have incompatible shapes as described above. 

        Returns
        -------
        T : 4d array
            Contraction of odeco tensors L and R, forming tensor train of length
            2.
        L : 3d array
            L = lammy[0]*(a_0)x(b_0)x(c_0) + ...
                    + lammy[r_L - 1]*(a_{r_L - 1})x(b_{r_L - 1})x(c_{r_L - 1})
        R : 3d array
            R = mu[0]*(d_0)x(e_0)x(f_0) + ...
                    + mu[r_R - 1]*(d_{r_R - 1})x(e_{r_R - 1})x(f_{r_R - 1})
        """
        # check input tuples ABC and DEF are in correct format
        if len(ABC) != 3 or len(DEF) != 3:
            raise ValueError("""Left and right tensors must be formed from 
                             tuple of 3 matrices ABC and DEF respectively""")
                             
        if (ABC[2].shape[0] != DEF[2].shape[0]):
            raise ValueError("Dimensions of C and F must be equal")

        rank_L = lammy.shape[0]
        
        # check rank_L is consistent
        if (rank_L != ABC[0].shape[1] or rank_L != ABC[1].shape[1] 
            or rank_L != ABC[2].shape[1]):
            raise ValueError("Rank of L inconsistent")
        
        # warn if rank_L > n for a dimension n, violation of orthogonality
        if (rank_L > ABC[0].shape[0] or rank_L > ABC[1].shape[0] 
            or rank_L > ABC[2].shape[0]):
            warnings.warn(""""Rank of L should not be greater than column size 
                          of A, B, or C for orthogonality""", stacklevel = 2)
        
        L = np.zeros((ABC[0].shape[0], ABC[1].shape[0], ABC[2].shape[0]))
        
        # create L
        for i in range(rank_L):
            L_i = np.tensordot(ABC[0][:,i], ABC[1][:,i], axes = 0)
            L_i = np.tensordot(L_i, ABC[2][:,i], axes = 0)
            L_i *= lammy[i]
            L += L_i
        
        rank_R = mu.shape[0]
        
        # check rank_R is consistent
        if (rank_R != DEF[0].shape[1] or rank_R != DEF[1].shape[1] 
            or rank_R != DEF[2].shape[1]):
            raise ValueError("Rank of R inconsistent")
            
        # warn if rank_R > n for a dimension n, violation of orthogonality
        if (rank_R > DEF[0].shape[0] or rank_R > DEF[1].shape[0] 
            or rank_R > DEF[2].shape[0]):
            warnings.warn(""""Rank of R should not be greater than column size 
                          of D, E, or F for orthogonality""", stacklevel = 2)
        
        R = np.zeros((DEF[0].shape[0], DEF[1].shape[0], DEF[2].shape[0]))
        
        # create R
        for j in range(rank_R):
            R_j = np.tensordot(DEF[0][:,j], DEF[1][:,j], axes = 0)
            R_j = np.tensordot(R_j, DEF[2][:,j], axes = 0)
            R_j *= mu[j]
            R += R_j
        
        # contract along dimension corresponding to C and F to produce 4-way T
        T = np.tensordot(L, R, axes = (2, 2))
        
        return T, L, R
    

    def decompose(self, T, d = 0, dodd_alg = "procrustes", max_iter = 1000, 
                  verbose = False):
        """
        uses SVD of sum of slices and DODD algorithm to compute decomposition 
        of T into two odeco tensors L and R.

        Parameters
        ----------
        T : 4-way tensor 
            Should be decomposable into an orthogonal length 2 train
        d : int, optional
            Hidden dimension of the contracted vectors. The default is 0, and 
            this is a flag for the DODD algorithm to choose a value of d itself 
            such that a solution exists.
        dodd_alg : string, optional
            DODD algorithm to use. The default is "procrustes".
        max_iter : int, optional
            Number of iterations of DODD algorithm before declaring failure. 
            The default is 1000.
        verbose : bool, optional
            if True, prints output of DODD alg as it runs.

        Raises
        ------
        ValueError
            If T is not a 4-way tensor

        Returns
        -------
        lammy : vector
            see below
        ABC : tuple of length 3 of matrices A, B, C
            columns of A, B, C are a_i, b_i, c_i such that
            L = lammy[0]*(a_0)x(b_0)x(c_0) + ...
                    + lammy[r_L - 1]*(a_{r_L - 1})x(b_{r_L - 1})x(c_{r_L - 1})
        mu : vector
            see below
        DEF : tuple of length 3 of matrices D, E, F
            columns of D, E, F are d_j, e_j, f_j such that
            R = mu[0]*(d_0)x(e_0)x(f_0) + ...
                    + mu[r_R - 1]*(d_{r_R - 1})x(e_{r_R - 1})x(f_{r_R - 1})
        """
        # check T is a 4-way tensor
        if len(T.shape) != 4:
            raise ValueError("T must be a 4-way tensor")
            
        # take weighted sums of slices of L's dimensions and R's dimensions in T
        S_L, S_R = self.sum_of_slices(T)
        
        #perform SVD on sum of slices to obtain A, B, E, F
        SVD_AB = lg.svd(S_L)
        rank_L = SVD_AB[1].size - np.searchsorted(np.flip(SVD_AB[1]), 10e-10)
        A = SVD_AB[0][:, :rank_L]
        B = SVD_AB[2][:rank_L, :].T
        
        SVD_DE = lg.svd(S_R)
        rank_R = SVD_DE[1].size - np.searchsorted(np.flip(SVD_DE[1]), 10e-10)
        D = SVD_DE[0][:, :rank_R]
        E = SVD_DE[2][:rank_R, :].T
        
        # contract to obtain lammy <c, f> mu matrix (dod for diag-orth-diag)
        X = np.empty(shape = (rank_L, rank_R)) 
        
        for i in range(rank_L):
            for j in range(rank_R):
                X_ij = np.tensordot(T, A[:, i], axes=(0, 0))
                X_ij = np.tensordot(X_ij, B[:, i], axes=(0, 0))
                X_ij = np.tensordot(X_ij, D[:, j], axes=(0, 0))
                X[i,j] = np.tensordot(X_ij, E[:, j], axes=(0, 0))
        
        # Call DODD to decompose LC^T FM matrix. Note that DODD will deal with 
        # invalid d by finding its own such that soln exists
        X_dodd = dodd.DODD()
        L, Q, M = dodd.DODD.decompose(X_dodd, X, d, method = dodd_alg, 
                                      max_iter = max_iter, verbose = verbose)
        
        # fix d to the one soln found in case it was not specified
        d = L.shape[0]
        
        # take lamdda and mu to nonzero portions of LQM matrix returned
        lammy = np.diag(L[:rank_L, :rank_L])
        mu = np.diag(M[:rank_R, :rank_R])
        
        # take C to be transpose of first rank_L orthogonal rows, F to be I_{d x rank_R}
        C = Q[:rank_L, :].T
        F = np.eye(d)[:, :rank_R]
        
        ABC = (A, B, C)
        DEF = (D, E, F)
        
        return lammy, ABC, mu, DEF


    def verify_decomp(self, lammy, ABC, mu, DEF, T, threshold = 10e-10, verbose = False):
        """
        verifies decomposition is correct by reconstructing the tensor T from the
        decomposition and measuring relative error.

        Parameters
        ----------
        lammy : 1d numpy array
            Weight vector lambda associated with left tensor L. Length == rank_L.
        ABC : tuple of length 3 of 2d numpy array's
            contains matrix A as its first element such that a_i vectors are 
            columns of A, then similarly for B as second element, and C as third.
        mu : 1d numpy array
            Weight vector mu associated with right tensor R. Length == rank_R.
        DEF : tuple of length 3 of 2d numpy array's
            contains matrix A as its first element such that a_i vectors are 
            columns of A, then similarly for B as second element, and C as third.
        T : TYPE
            DESCRIPTION.
        threshold : TYPE, optional
            DESCRIPTION. The default is 10e-10.
        verbose : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        bool
            True if decomposition accurate within threshold of relative 
            error, False otherwise.
        """
        T_ver, L_ver, R_ver = self.create_tensor(lammy, ABC, mu, DEF)
        
        # measure relative error and return binary result
        rel_error = lg.norm(T - T_ver) / lg.norm(T)
        if verbose:
            print("Relative error = ")
            print(rel_error)
            print("\n")
        
        if (rel_error < threshold):
            return True
        
        return False

        
    def test(self, n_A, n_B, d, n_D, n_E, rank_L, rank_R, iters = 100, 
             threshold = 10e-10, dodd_alg = "procrustes", find_d = False, 
             verbose = False):
        """
        tests orthogonal decomposition algorithm, printing out results.

        Parameters
        ----------
        n_A : int
            dimension spanned by orthogonal a_i vectors
        n_B : int
            dimension spanned by orthogonal b_i vectors
        d : int
            dimension spanned by orthogonal c_i and f_j vectors
        n_D : int
            dimension spanned by orthogonal d_j vectors
        n_E : int
            dimension spanned by orthogonal e_j vectors
        rank_L : int
            rank of left tensor i.e. L = \sum_{i=0}^rank_L lamda_i a_i x b_i x c_i
        rank_R : int
            rank of right tensor i.e. R = \sum_{j=0}^rank_R mu_j d_j x e_j x f_j
        iters : int, optional
            Number of tests to generate. The default is 100.
        threshold : float, optional
            Maximum relative error to declare success. The default is 10e-10.
        dod_alg : string, optional
            Method of DODD used. The default is "procrustes".
        find_d : boolean, optional
            specifies whether algorithm should find its own hidden dimension
            such that a solution exists, or input the d the decomposable 
            tensor has been constructed from. The default is False.
        verbose : boolean, optional
            if True, prints output of DODD alg as it runs.
        """
        success = 0
        
        for i in range(iters):
            # set up test
            lammy = rm.randint(low = 1, high = 20, size = rank_L)
            A = ortho_group.rvs(n_A)[:, :rank_L]
            B = ortho_group.rvs(n_B)[:, :rank_L]
            C = ortho_group.rvs(d)[:, :rank_L]
            
            mu = rm.randint(low = 1, high = 20, size = rank_R)
            D = ortho_group.rvs(n_D)[:, :rank_R]
            E = ortho_group.rvs(n_E)[:, :rank_R]
            F = ortho_group.rvs(d)[:, :rank_R]
            
            T, L, R = self.create_tensor(lammy, (A, B, C), mu, (D, E, F))
            
            # decompose
            if find_d == True:
                lammy_sol, ABC_sol, mu_sol, DEF_sol = self.decompose(T, 
                                                        dodd_alg = dodd_alg, 
                                                        verbose = verbose)
            
            else:
                lammy_sol, ABC_sol, mu_sol, DEF_sol = self.decompose(T, d, 
                                                        dodd_alg = dodd_alg,
                                                        verbose = verbose)
            
            if self.verify_decomp(lammy_sol, ABC_sol, mu_sol, DEF_sol, T, 
                                  threshold, verbose = True):
                success += 1
            
        print("Number of successes: %d" % success)