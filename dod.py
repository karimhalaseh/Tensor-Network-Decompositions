#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:57:21 2020

@author: karimhalaseh
"""

import numpy as np
import numpy.random as rm
import scipy.linalg as lg
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
import time

def tandem_procrustes(A, B, tol = 10e-22, max_iter = 100, verbose = False):
    """
    Finds V with orthonormal columns, diagonal D, such that ||A-VDB||^2 is
    minimized using tandem algorithm (Everson, 1997).
    
    A is of size mxn, B is of size qxn (n matching is checked). V returned is
    mxq (q orthonormal columns in m-space; m >= q is checked). D is qxq.
    
    If solution does not converge to solution with error less than tol, returns
    V and D from last iteration. Returns boolean indicating whether convergence 
    was successful. If verbose = True, prints final error if converged.
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
    D = np.eye(q)
    
    for i in range(max_iter):
        # Update V
        V = lg.polar(ABT @ D)[0] # V = OPF(A B^T D)
        
        # Update D
        # d_k = dot prod of k-th columns of AB^T and V / norm^2 of k-th row of B
        D = np.diag(np.diag(ABT.T @ V) /  np.sum(B**2, axis = 1))
        # print(D)
        # set zero rows of B to have corresponding d_k = 0
        np.nan_to_num(D, copy = False, nan = 0.0, posinf = 0.0, neginf = 0.0)
        
        error = np.sum((A - (V @ D @ B))**2)
        # print(error)
        
        if (error < tol):
            if verbose:
                print("Converged successfully with error " + str(error))
                
            return V, D, True
        
    return V, D, False


def sinkhorn(X, tol = 10e-22, max_iter = 1000, verbose = False):
    """
    sinkhorn algorithm implementation. 
    
    Parameters: matrix X, tol sets convergence threshold, max_iter sets number
    of iterations before failure is declared.
    
    Returns positive diagonal lammy, mu, and orthogonal Q such that 
    Q = lammy @ X @ mu, as well as boolean indicating if converged and number
    of iterations for convergence (max_iter returned if this was not achieved)
    """    
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
        
        if np.all(np.abs(col_sum - ones) < tol) and np.all(np.abs(row_sum - ones) < tol):
            if verbose:
                print("Sinkhorn converged successfully after " + str(i) + " iterations")
                
            return lammy, Q, mu, i, True
        
    if verbose:    
        print("Sinkhorn did not converge after " + str(max_iter) + " iterations")
        
    return lammy, Q, mu, max_iter, False



class DOD():
    """
    Class for Diagonal-Orthogonal-Decomposition (currently only compatible for 
    the square case)
    """
    
    def create_matrix(self, n, exact = True, verbose = False):
        """
        Creates matrix of size n. If exact = True, this matrix is created to 
        have an exact underlying DOD decomposition. If verbose = True, all
        underlying parameters printed.
        
        Returns matrix created X.
        """
        if exact:
            # sample X = LCFM
            L = np.diag(rm.randint(1, 11, n))
            M = np.diag(rm.randint(1, 11, n))
            C = ortho_group.rvs(n)
            F = ortho_group.rvs(n)
        
            X = L @ C @ F @ M
            
            if verbose:
                self.print_LCFM(L, C, F, M, sol = False)
            
        else:
            X = 5 * rm.randn(n, n)
            
            if verbose:
                print("X: ")
                print(X)
                print("\n")
                
        return X
            
        
    def decompose(self, X, method = "sinkhorn", max_iter = 1000, verbose = False):
        """
        computes DOD decomposition.
        
        Parameters: matrix to decompose X, method of decomposition (one of 
        {"sinkhorn", "procrustes1", "procrustes2"}, with "sinkhorn" as default),
        max_iter sets number of iterations of method before declaring failure
        
        Returns: diagonal L, C with orthonormal rows, F with orthonormal 
        columns, diagonal M such that X = LCFM. Also returns boolean indicating 
        whether decomposition is exact (if False, convergence in algorithms 
        was not reached), and number of iterations for convergence (max_iter 
        returned if this was not achieved)
        
        Throws: ValueError if method not recognized
        """
        I = np.eye(X.shape[0])
        
        if method == "sinkhorn":
            # call sinkhorn on X^2 as it solves for stochastic matrix
            L, C, M, num_iters, conv = sinkhorn(X**2, max_iter = max_iter, verbose = verbose)
            
            # C will be elementwise root of decomposition adjusted for original sign
            C = np.sign(X) * (C**0.5)
            
            # lambda, mu will be sqrt inverse of diagonal matrices found from sinkhorn
            L = np.diag(1 / (L ** 0.5))
            M = np.diag(1 / (M ** 0.5))
            
            return L, C, I, M, num_iters, conv
            
            
        elif method == "procrustes1":
            # initialize L, C, M
            L = np.copy(I)
            C = X
            M = np.copy(I)
            
            for i in range(max_iter):
                # call tandem procrustes on C.T
                V, D, convL = tandem_procrustes(C.T, I, verbose = verbose)
                D_inv = np.diag(1/np.diag(D))
                
                # multiply L by D, C by D^(-1)
                L = L @ D
                C = D_inv @ C 
                
                # call tandem procrustes on C
                V, D, convR = tandem_procrustes(C, I, verbose = verbose)
                D_inv = np.diag(1/np.diag(D))
                
                # multiply M by D, C by D^(-1)
                M = D @ M
                C = C @ D_inv
                
                # check if converged
                if np.all((convL, convR)):
                    if verbose:
                        print("Procrustes1 converged successfully after " + str(i) + " iterations")
                        
                    return L, C, I, M, i, True
            
            if verbose:
                print("Procrustes1 did not converge after " + str(max_iter) + " iterations")
                
            return L, C, I, M, max_iter, False
        
        
        elif method == "procrustes2": 
            L = np.copy(I)
            C = X
            F = np.copy(I)
            M = np.copy(I)
            E = np.copy(I) # error matrix
            
            for i in range(max_iter):
                # call tandem procrustes on (LCE)^T
                V, D, convL = tandem_procrustes((L @ C @ E).T, I, verbose = verbose)
                D_inv = np.diag(1/np.diag(D))
                
                # update error matrix, assign new L and C
                E = V @ D_inv @ L @ C @ E
                L = D
                C = V.T
                
                # call tandem procrustes on EFM 
                V, D, convR = tandem_procrustes(E @ F @ M, I, verbose = verbose)
                D_inv = np.diag(1/np.diag(D))
                
                # update error matrix, assign new F and M
                E = E @ F @ M @ D_inv @ V.T
                M = D
                F = V
                
                # check if converged
                if np.all((convL, convR)):
                    if verbose:
                        print("Procrustes2 converged successfully after " + str(i) + " iterations")
                        
                    return L, C, F, M, i, True
            
            if verbose:
                print("Procrustes2 did not converge after " + str(max_iter) + " iterations")
                
            return L, C, F, M, max_iter, False                
            
        
        else:
            raise ValueError('Method not recognized. Method must be one of {"sinkhorn", "procrustes1", "procrustes2"}')
    
    
    def print_LCFM(self, L, C, F, M, conv = False, sol = False):
        """
        Prints outs L, C, F, M, LM, CF in decomposition, as well as result of 
        convergence and orthogonality check if this is a solution to inspect
        """
        if sol == False:
            print("L: ")
            print(L)
            print("\n")
            
            print("M: ")
            print(M)
            print("\n")
            
            print("lambda * mu: ")
            print(L @ M)
            print("\n")
    
            print("C: ")
            print(C)
            print("\n")
            
            print("F: ")
            print(F)
            print("\n")
            
            print("C * F: ")
            print(C @ F)
            print("\n")
            
            print("X: ")
            print(L @ C @ F @ M)
            print("\n")
            
        else:
            print("Converged? ")
            print(conv)
            print("\n")
            
            print("X_sol: ")
            print(L @ C @ F @ M)
            print("\n")
            
            print("L_sol: ")
            print(L)
            print("\n")
    
            print("M_sol: ")
            print(M)
            print("\n")        
            
            print("L_sol * M_sol: ")
            print(L @ M)
            print("\n")
    
            print("C_sol: ")
            print(C)
            print("\n")
    
            print("F_sol: ")
            print(F)
            print("\n")
            
            print("C_sol * F_sol: ")
            print(C @ F)
            print("\n")
            
            print("C found orthogonal? ")
            print(C @ C.T)
            print("\n")
            
            print("F found orthogonal? ")
            print(F.T @ F)
            print("\n")
            
        
    def compare_algs(self, n, iters = 100, threshold = 10e-10, max_iter = 1000, 
                     exact = True, plot = False, verbose = False):
        """
        compares three DOD algorithms for accuracy
        
        Parameters: size of matrices to generate n, number of tests to run iters,
        threshold of relative error for success, max_iter for each algorithm
        before declaring failure, exact sets if tests should have exact
        underlying decomposition (or decompositions will be approximations), 
        plot prints histograms of number of iterations for each algorithm
        until convergence if true, verbose prints out tests and results.
        """
        success_sh = 0
        conv_sh = 0
        iters_sh = []

        success_p1 = 0
        conv_p1 = 0
        iters_p1 = []
        
        success_p2 = 0
        conv_p2 = 0
        iters_p2 = []
        
        I = np.eye(n)
        
        for i in range(iters):
            X = self.create_matrix(n, exact, verbose)
            
            # run sinkhorn 
            L_sol, C_sol, F_sol, M_sol, num_iters, conv = self.decompose(X, method = "sinkhorn", 
                                                                         max_iter = max_iter, verbose = verbose)
            
            if verbose: 
                self.print_LCFM(L_sol, C_sol, F_sol, M_sol, conv, sol = True)
                
            if conv:
                conv_sh += 1
                
            if ((lg.norm((C_sol @ C_sol.T) - I)/lg.norm(I) < threshold) and 
                (lg.norm((F_sol.T @ F_sol) - I)/lg.norm(I) < threshold)):
                success_sh += 1
                iters_sh.append(num_iters)
                
                
            # run procrustes1
            L_sol, C_sol, F_sol, M_sol, num_iters, conv = self.decompose(X, method = "procrustes1", 
                                                                         max_iter = max_iter, verbose = verbose)
            
            if verbose: 
                self.print_LCFM(L_sol, C_sol, F_sol, M_sol, conv, sol = True)
                
            if conv:
                conv_p1 += 1
                
            if ((lg.norm((C_sol @ C_sol.T) - I)/lg.norm(I) < threshold) and 
                (lg.norm((F_sol.T @ F_sol) - I)/lg.norm(I) < threshold)):
                success_p1 += 1
                iters_p1.append(num_iters)
                
                
            # run procrustes2
            L_sol, C_sol, F_sol, M_sol, num_iters, conv = self.decompose(X, method = "procrustes2", 
                                                                         max_iter = max_iter, verbose = verbose)
            
            if verbose: 
                self.print_LCFM(L_sol, C_sol, F_sol, M_sol, conv, sol = True)
                
            if conv:
                conv_p2 += 1
                
            if (lg.norm((L_sol @ C_sol @ F_sol @ M_sol) - X) < threshold):
                success_p2 += 1
                iters_p2.append(num_iters)
                
        
        if plot:            
            plt.hist(iters_sh, bins = 20, range = (0, max_iter))
            plt.xlabel("Number of Iterations of Algorithm for Convergence")
            plt.ylabel("Frequency")
            # plt.ylim((0, 50))
            plt.title("Sinkhorn")
            plt.grid()
            plt.show()
            
            plt.hist(iters_p1, bins = 20, range = (0, max_iter))
            plt.xlabel("Number of Iterations of Algorithm for Convergence")
            plt.ylabel("Frequency")
            # plt.ylim((0, 50))
            plt.title("Procrustes1")
            plt.grid()
            plt.show()
            
            plt.hist(iters_p2, bins = 20, range = (0, max_iter))
            plt.xlabel("Number of Iterations of Algorithm for Convergence")
            plt.ylabel("Frequency")
            # plt.ylim((0, 50))
            plt.title("Procrustes2")
            plt.grid()
            plt.show()
            
            
        print("Sinkhorn: " + str(conv_sh) + " converged and " + str(success_sh) 
              + " successful out of " + str(iters) + " runs")
        print("Procrustes1: " + str(conv_p1) + " converged and " + str(success_p1) 
              + " successful out of " + str(iters) + " runs")
        print("Procrustes2: " + str(conv_p2) + " converged and " + str(success_p2) 
              + " successful out of " + str(iters) + " runs")
        
        
    def compare_runtimes(self, max_iter = 1000, verbose = False):
        """
        Compares runtimes of the three DOD algorithms, producing plots of 
        runtime as n varies for each.
        """
        sh_time = []
        p1_time = []
        p2_time = []
        
        for n in range(4, 104, 4):
            sh_sum = 0
            p1_sum = 0
            p2_sum = 0
            
            for i in range(5):
                X = self.create_matrix(n, verbose = verbose)
                
                # time sinkhorn
                start = time.time()
                self.decompose(X, method = "sinkhorn", max_iter = max_iter, verbose = verbose)
                end = time.time()
                sh_sum += end - start
                
                # time procrustes1
                start = time.time()
                self.decompose(X, method = "procrustes1", max_iter = max_iter, verbose = verbose)
                end = time.time()
                p1_sum += end - start
                
                # time procrustes2
                start = time.time()
                self.decompose(X, method = "procrustes2", max_iter = max_iter, verbose = verbose)
                end = time.time()
                p2_sum += end - start
                
            sh_time.append(sh_sum / 5)
            p1_time.append(p1_sum / 5)
            p2_time.append(p2_sum / 5)
        
        x = np.arange(start = 4, stop = 104, step = 4)
        plt.figure()
        plt.plot(x, sh_time, color = 'b', label = 'Sinkhorn')
        plt.plot(x, p1_time, color = 'r', label = 'Procrustes1')
        plt.plot(x, p2_time, color = 'g', label = 'Procrustes2')
        plt.xlabel("n")
        plt.ylabel("Runtime (s)")
        plt.grid()
        plt.legend()
        plt.show()
                
            
        
# RUN TESTS
lcfm = DOD()
# lcfm.compare_algs(10, iters = 100, plot = True, verbose = True)
lcfm.compare_runtimes(verbose = True)
