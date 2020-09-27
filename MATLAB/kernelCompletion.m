function B_partial = kernelCompletion(T,A_partial)
% Given an nxn matrix T of the form T = A*Lambda*A'*B*M*B', where A is an
% nxn orthogonal matrix completing the orthonormal columns of A_partial, B
% is an nxn orthogonal matrix, Lambda and M are nxn diagonal matrices such
% that rank(Lambda) >= rank(M), this method finds the matrix B

% Inputs: nxn double: T, a matrix of the form described above
%         nx? double: A_partial, a matrix with orthonormal columns
% Outputs: nx? double: B_partial, a matrix with orthonormal columns,
%                                 corresponding to the rank(M) columns of
%                                 an nxn orthogonal matrix B satisfying the
%                                 above equation

n = size(T,1);
rank = size(A_partial,2);
tol_orth = 10^(-14); % tolerance for error in orthogonality

%% Symmetrization

A = completeOrthBasis(A_partial,rank,tol_orth);
J = A'*T*A;
J = J(1:rank,1:rank); % the top-left corner block of T that we will symmetrize

lambda_inv = ones(n,1);
if rank > 1
    L = zeros(nchoosek(rank,2),rank); % construct the matrix L
    m = 0;
    for j = 1:rank-1
        for i = j+1:rank
            m = m + 1;
            L(m,i) = J(i,j);
            L(m,j) = -J(j,i);
        end
    end
    [~,~,N] = svd(L);
    lambda_inv(1:rank) = N(:,end); % its nullspace is spanned by the vector whose entries
    % are the inverses of the lambda_i's, assuming generic conditions
end
S = diag(lambda_inv)*A'*T*A;
S(rank+1:end,1:rank) = S(1:rank,rank+1:end)'; % by symmetry, we know the first
% rankA rows and columns

%% Matrix completion

T = sym(S);
syms x;
for i = rank+1:n
    for j = i:n
        T(i,j) = x; % introduce a symbolic variable x into T
    end
    [~,U,P] = lu(T); % Gaussian elimination of T
    U = P'*U; % reshuffle U because MATLAB sometimes exchanges rows
    for j = i:n
        s = solve(U(i,j) == 0,x); % solve for the x which makes the entry 0 after Gaussian elimination
        T(i,j) = s; % insert this value of x in the entry and its mirror entry
        T(j,i) = s;
    end
end

%% Finding B

R = double(T);
[B,D] = eig(A*R*A');
[B_partial,~] = manageEigenvectors(B,D);

end

