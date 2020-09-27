function M_out = completeOrthBasis(M, rank, tol)
% Given an nxn matrix M whose first rank columns are orthonormal, this
% function completes the columns of M to an orthonormal basis using the
% Gram-Schmidt process, with orthogonality tolerance specified by tol

n = length(M);

for i=(rank+1):n
    nrm = 0;
    while nrm<tol
        vi = randn(n,1);
        vi = vi -  M(:,1:i-1)  * ( M(:,1:i-1).' * vi )  ;
        nrm = norm(vi);
    end
    M(:,i) = vi ./ nrm;
end

M_out = M;

end

