import numpy as np
import numpy.random as rm
import scipy.linalg as lg
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
import warnings

def tandem_procrustes(A, B, tol = 10e-28, verbose = False):
    """
    Finds V with orthonormal columns, diagonal D, such that ||A-VDB||^2 is
    minimized using tandem algorithm (Everson, 1997) with only one iteration. 
    If solution does not converge to solution with error less than tol, returns
    V and D from last iteration. 

    Parameters
    ----------
    A : 2d array
        target matrix of size mxn
    B : 2d array
        source matrix of size qxn
    tol : float, optional
        Maximum sum of squared error such that exact solution achieved. 
        The default is 10e-28.
    verbose : bool, optional
        If True, prints out final error and if convergence reached. 
        The default is False.

    Raises
    ------
    ValueError
        If A and B inputs are not of valid dimension

    Returns
    -------
    V : 2d array
        mxq matrix with orthonormal columns
    D : 2d array
        qxq diagonal matrix
    conv : bool
        True if error < tol, False otherwise
    """
    m = A.shape[0]
    n = A.shape[1]
    q = B.shape[0]
    
    if n != B.shape[1]:
        raise ValueError("Number of columns of A and B must match")
    
    if m < q:
        raise ValueError("Cannot have " + str(q) + " orthogonal columns in " 
              + str(m) + "-space")
    
    ABT = A @ B.T
    
    V = lg.polar(ABT @ np.eye(q))[0] # V = OPF(A B^T D)
    
    # d_k = dot prod of k-th columns of AB^T and V / norm^2 of k-th row of B
    D = np.diag(np.diag(ABT.T @ V) /  np.sum(B**2, axis = 1))
    
    error = np.sum((A - (V @ D @ B))**2)
    
    conv = error < tol
    
    if conv:
        if verbose:
            print("Converged successfully with error " + str(error))
    
    else:
        if verbose:
            print("Did not converge to within tolerance. Error is " + str(error))
        
    return V, D, conv


def sinkhorn(X, tol = 10e-28, max_iter = 1000, verbose = False):
    """
    sinkhorn algorithm implementation (Sinkhorn, Knopp, 1967). Returns positive 
    diagonal lammy, mu, and orthogonal Q such that Q = lammy @ X @ mu.

    Parameters
    ----------
    X : 2d array
        matrix should be square and have strictly positive elements
    tol : float, optional
        convergence threshold. The default is 10e-28.
    max_iter : int, optional
        Number of iterations before failure is declared. The default is 1000.
    verbose : bool, optional
        If True, prints out number of iterations takes and if convergence 
        reached. The default is False.

    Returns
    -------
    lammy : vector
        np.diag(lammy) will be positive diagonal
    Q : matrix
        orthogonal if convergence successful
    mu : vector
        np.diag(mu) will be positive diagonal
    conv: bool
        True if converged, False otherwise
    """
    if (np.any(X) < 0) or (X.shape[0] != X.shape[1]):
        warnings.warn("""Sinkhorn's theorem only applies to square matrices 
                          with positive entries""")
                          
    n = X.shape[0]
    ones = np.ones(n)
    
    lammy = np.ones(n)
    mu = np.ones(n)
    Q = np.copy(X)
    
    for i in range(max_iter):
        col_sum = np.sum(Q, axis = 0)
        Q = Q / col_sum
        mu = mu / col_sum
        
        row_sum = np.sum(Q, axis = 1)
        Q = Q / row_sum[:, None]
        lammy = lammy / row_sum
        
        error = np.sum((col_sum - ones)**2) + np.sum((row_sum - ones)**2)
        
        conv = error < tol

        if conv:
            if verbose:
                print("Sinkhorn converged successfully after " + str(i) + " iterations")
                
            return lammy, Q, mu, conv
        
    if verbose:    
        print("Sinkhorn did not converge after " + str(max_iter) + " iterations")
        
    return lammy, Q, mu, conv


class DODD():
    """
    Class for Diagonal-Orthogonal-Diagonal Decomposition as in "Orthogonal 
    Tensor Network Decompositions" (Halaseh, Muller, Robeva), Section 6
    """
    def create_matrix(self, m, n, d = 0, exact = True):
        """
        Creates matrix of size m x n. If exact = True, this matrix is created 
        to have an exact underlying DODD.

        Parameters
        ----------
        m : int
            Number of rows
        n : int
            Number of cols
        d : int
            Hidden dimension of contracted vectors, must be specified if 
            exact is True. Default = 0, when random mxn matrix sampled and d
            unused.
        exact : bool, optional
            Specifies if matrix with exact DODD is created. The default is True.
            
        Raises
        ------
        ValueError
            If d = 0 but exact = True.

        Returns
        -------
        X : 2d array
            Matrix created
        """
        if exact:
            if d == 0 or d < max(m, n):
                raise ValueError("If exact = True, d must be specified and valid")
            # sample X = LQM (remove zero paddings)
            L = np.diag(rm.randint(1, 11, m))
            M = np.diag(rm.randint(1, 11, n))
            Q = ortho_group.rvs(d)
        
            X = L @ Q[:m, :] @ np.eye(d)[:, :n] @ M
            
        else:
            X = 5 * rm.randn(m, n)
                
        return X
            
        
    def decompose(self, X, d = 0, method = "procrustes", max_iter = 1000, 
                  learning_rate = 2, verbose = False):
        """
        computes DODD.
        
        Parameters: matrix to decompose X, method of decomposition (one of 
        {"sinkhorn", "procrustes"}, with "sinkhorn" as default),
        learning rate for procrustes algorithm in the non-square case,
        max_iter sets number of iterations of method before declaring failure
        
        Returns: diagonal L, C with orthonormal rows, F with orthonormal 
        columns, diagonal M such that X = LCFM. Also returns boolean indicating 
        whether decomposition is exact (if False, convergence in algorithms 
        was not reached), and number of iterations for convergence (max_iter 
        returned if this was not achieved)
        
        Throws: ValueError if method not recognized

        Parameters
        ----------
        X : 2d array
            matrix to decompose of size mxn
        d : int, optional
            hidden dimension d. The default is 0, and if this is inputted or 
            a value of d less than max(m, n), algorithm proceeds with 
            d = max(m, n) + 5
        method : string, optional
            Algorithm for DODD. The default is "procrustes". The other option
            is "sinkhorn", and this only works for d = m = n
        max_iter : int, optional
            Number of iterations of DODD algorithm before failure declared. 
            The default is 1000.
        learning_rate : int, optional
            Number of executions of tandem procrustes algorithm within each
            iteration of procrustes-based DODD. The default is 2. Unused if 
            method = "sinkhorn".
        verbose : bool, optional
            prints out performance of algorithm as it is run if True. 
            The default is False.

        Raises
        ------
        ValueError
            if method = "sinkhorn" but !(d = m = n); or if method != 
            {"sinkhorn", "procrustes"}.

        Returns
        -------
        L : 2d array
            positive diagonal L of size dxd such that X = LQM (with padded zeros)
        Q : 2d array
            orthogonal  Q of size dxd such that X = LQM (with padded zeros)
        M : 2d array 
            positive diagonal M of size dxd such that X = LQM (with padded zeros)
        """
        m = X.shape[0]
        n = X.shape[1]
        
        if (d == 0) or (d < max(m, n)):
            d = max(m, n) + 5
            warnings.warn("d either unspecified or invalid. We will proceed with d = "
                          + str(d), stacklevel = 2)
                          
        I = np.eye(d)
        
        if method == "sinkhorn":
            # sinkhorn only works on square case
            if m != n or m != d:
                raise ValueError("Sinkhorn only works on the square case")
            
            # call sinkhorn on X.^2 as it solves for stochastic matrix
            L, Q, M, conv = sinkhorn(X**2, max_iter = max_iter, verbose = verbose)
            
            # Q will be elementwise root of decomposition adjusted for original sign
            Q = np.sign(X) * (Q**0.5)
            
            # lambda, mu will be sqrt inverse of diagonal matrices found from sinkhorn
            L = np.diag(1 / (L ** 0.5))
            M = np.diag(1 / (M ** 0.5))
            
            return L, Q, M
            
            
        elif method == "procrustes":
            # initialize L, C, M
            L = np.diag(np.concatenate((np.ones(m), np.zeros(d-m))))
            M = np.diag(np.concatenate((np.ones(n), np.zeros(d-n))))
            Q = np.concatenate((X, rm.rand(m, d-n)), axis = 1)
            Q = np.concatenate((Q, rm.rand(d-m, d)))
            
            for i in range(int(max_iter / learning_rate)):
                
                for j in range(learning_rate):                    
                    # call tandem procrustes on Q
                    V, D, conv = tandem_procrustes(Q, I, verbose = verbose)
                    D_inv = np.diag(1/np.diag(D))
                    
                    # multiply M by D, Q by D^(-1)
                    M = D @ M
                    Q = Q @ D_inv
                    
                    # call tandem procrustes on Q.T
                    V, D, conv = tandem_procrustes(Q.T, I, verbose = verbose)
                    D_inv = np.diag(1/np.diag(D))
                    
                    # multiply L by D, Q by D^(-1)
                    L = L @ D
                    Q = D_inv @ Q
                    
                    # check if converged
                    if conv:
                        if verbose:
                            print("Procrustes converged successfully after " + 
                                  str(i * (j+1)) + " iterations")
                        
                        return L, Q, M
                
                # update approximation
                if d-n > 0:
                    Q = np.concatenate((Q[:m, :n], V.T[:m, -(d-n):]), axis = 1)
                    
                    if d-m > 0:
                        Q = np.concatenate((Q, V.T[-(d-m):, :]))
                        
                elif d-m > 0:
                    Q = np.concatenate((Q[:m, :], V.T[-(d-m):, :]))
                
                if verbose:
                    print("\n")
                    print("Updating approximation...")
                    print("\n")
            
            if verbose:
                print("Procrustes did not converge after " + str(max_iter) 
                      + " iterations")
                
            return L, Q, M              
        
        else:
            raise ValueError("""Method not recognized. Method must be one of 
                             {"sinkhorn", "procrustes"}""")
            

    def test(self, m, n, d = 0, method = "procrustes", learning_rate = 2, 
             iters = 100, threshold = 10e-10, max_iter = 1000, 
             exact = True, verbose = False):
        """
        tests DODD algorithm.
        
        Parameters: m, n, d control size of matrices, method specifies which 
        DOD algorithm to run (only procrustes will run for the case 
        !(m = n = d)), iters controls how many tests to run, threshold specifies
        tolerance of solution, max_iter specifies number of iterations of each
        test before algorithm declares failure, exact tests if matrices with 
        exact DOD solutions are tested.

        Parameters
        ----------
        m : int
            Number of rows of matrix to decompose
        n : int
            Number of columns of matrix to decompose
        d : int, optional
            Hidden dimension of DODD of matrix. The default is 0, and must be 
            in the case where a exact = False i.e. no exact matrix is sampled
        method : string, optional
            DODD algorithm to use. The default is "procrustes".
        learning_rate : int, optional
            number of executions of procrustes algorithm within each iteration
            of procrustes-based algorithm. The default is 2. Unused if 
            method = "sinkhorn"
        iters : int, optional
            Number of tests to run. The default is 100.
        threshold : float, optional
            relative error maximum for successful test. The default is 10e-10.
        max_iter : int, optional
            Number of iterations algorithm runs before declaring failure. 
            The default is 1000.
        exact : bool, optional
            Samples matrix with exact DODD if True. The default is True.
        verbose : bool, optional
            Prints output of algorithm as it runs if True. The default is False.

        Returns
        -------
        None.
        """
        success = 0
        
        for i in range(iters):
            X = self.create_matrix(m, n, d, exact = exact)
            
            # run algorithm
            L_sol, Q_sol, M_sol = self.decompose(X, d, method = method, 
                                                 learning_rate = learning_rate,
                                                 max_iter = max_iter, 
                                                 verbose = verbose)
            
            # fix d to the one soln found in case it was not specified
            I_d = np.eye(Q_sol.shape[0])
            
            rel_error = lg.norm((Q_sol @ Q_sol.T) - I_d)/lg.norm(I_d)
            
            if (rel_error < threshold):
                success += 1
            
        print("Number of successes: %d" % success)