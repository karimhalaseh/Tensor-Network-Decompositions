function L = reverseLexicographic(ranks)
% Given an array of positive integers ("ranks"), generates all indices from
% 1 to each rank in reverse-lexicographic order, displayed in each row

% Inputs: 1x? double: ranks, an array of positive integers
% Outputs: ??x? double: L, all possible indices from 1 to each rank, in
%                          reverse-lexicographic order, displayed in each
%                          row

n = length(ranks);
if n == 1
    L = 1:ranks;
    L = L';
else
    L = zeros(prod(ranks),n);
    v = 1:ranks(end);
    rest = ranks(1:end-1);
    L(:,end) = repelem(v,prod(rest));
    M = reverseLexicographic(rest);
    L(:,1:end-1) = repmat(M,ranks(end),1);
end

end

