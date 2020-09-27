function [V_out,D_out] = manageEigenvectors(V,D)
% Extracts eigenvectors corresponding to non-zero eigenvalues from 
% eigendecomposition V*D*V^(-1), assuming all eigenvalues are real

tol_eigenGap = 5; % if the order of consecutive eigenvalues suddenly decreases by 10^(tol_eigenGap),
% then it is likely this eigenvalue, and following eigenvalues, are 0

[d,ind] = sort(abs(real(diag(D))),'descend');
V = V(:,ind);
eigenGaps = abs(diff(log10(d)));
gapIndices = find(eigenGaps >= tol_eigenGap);
if ~isempty(gapIndices)
    gapIndex = gapIndices(1);
    fprintf("rank = %d \n", gapIndex);
    V(:,gapIndex+1:end) = [];
    V_out = V;
    D_out = diag(d(1:gapIndex));
else
    fprintf("rank = %d \n", size(V,1));
    V_out = V;
    D_out = diag(d);
end

end

