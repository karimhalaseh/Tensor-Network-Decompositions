function [ranks,vecs,coeffs] = decomposeSymmOdecoTrain(T)
% Decomposes a generic symmetric, odeco tensor train satisfying the 
% Decreasing Rank Condition, and finds the orthonormal vectors and 
% corresponding coefficients of each "carriage"

% Inputs: tensor: T, a generic symmetric, odeco tensor train of length L
%                    whose components belong in R^n, satisfying the
%                    Decreasing Rank Condition
% Outputs: 1xL cell{nx? double}: vecs, a cell array containing the
%                                      orthonormal vectors used to generate
%                                      the "carriages" in the train. The
%                                      number of vectors is equal to the
%                                      rank of the "carriage"
%          1xL cell{1x? double}: coeffs, a cell array containing the 
%                                      coefficients multiplying each symmetric
%                                      tensor term. The number of
%                                      coefficients for each "carriage" is
%                                      equal to the rank of the "carriage"
%          1xL double: ranks, a positive-integer array specifying the 
%                             ranks of each "carriage" of the train,
%                             from left to right

assert(isa(T,'tensor'),'T must be of type tensor. Cast T into type tensor using TensorToolbox.');
assert(range(size(T))==0,'T must have the same number of components in each way. It must be a tensor in R^(n^L).');

fprintf("DECOMPOSING SYMMETRIC ODECO TENSOR TRAIN WITH DECREASING RANKS \n");
fprintf("\n");

n = size(T,1);
N = ndims(T); % Note: L = N-2

%% Decompose from left to right

fprintf("Decomposing from left to right... \n");
fprintf("\n");

vecsLR = cell([1,N-2]);

% Find the orthonormal vectors on left end of the train

S = T;
v = rand(n,1);
indices = 1:N;
indices(indices == 1 | indices == 2) = [];
Q = cell(1,N-2);
Q(:) = {v};
S = ttv(S,Q,indices); % contract all components except the left end with a random vector
[V,D] = eig(double(S));
[V,~] = manageEigenvectors(V,D);
vecsLR{1} = V;
currentRank = size(V,2);

% Continue decomposing from left to right

for i = 2:N-2
    S = T;
    v = rand(n,1);
    indices = 1:N;
    indices(indices == i | indices == i+1) = [];
    Q = cell(1,N-2);
    Q(:) = {v};
    S = ttv(S,Q,indices);
    V = kernelCompletion(double(S),V);
    if size(V,2) > currentRank % if the rank increases, then this violates the Decreasing Ranks Condition, so
        % the direction of decomposition must be incorrect
        fprintf("Rank increased! Stopping decomposition... \n");
        vecsLR(i:end) = {[]};
        break;
    else
        vecsLR{i} = V;
        currentRank = size(V,2);
    end
end

%% Decompose from right to left

fprintf("\n");
fprintf("Decomposing from right to left... \n");
fprintf("\n");

vecsRL = cell([1,N-2]);

% Find the orthonormal vectors on right end of the train

S = T;
v = rand(n,1);
indices = 1:N;
indices(indices == N-1 | indices == N) = [];
Q = cell(1,N-2);
Q(:) = {v};
S = ttv(S,Q,indices); % contract all components except the right end with a random vector
[V,D] = eig(double(S));
[V,~] = manageEigenvectors(V,D);
vecsRL{N-2} = V;
currentRank = size(V,2);

% Continue decomposing from right to left

for i = N-1:-1:3
    S = T;
    v = rand(n,1);
    indices = 1:N;
    indices(indices == i | indices == i-1) = [];
    Q = cell(1,N-2);
    Q(:) = {v};
    S = ttv(S,Q,indices);
    V = kernelCompletion(double(S)',V);
    if size(V,2) > currentRank % if the rank increases, then this violates the Decreasing Ranks Condition, so
        % the direction of decomposition must be incorrect
        fprintf("Rank increased! Stopping decomposition... \n");
        vecsRL(1:i-2) = {[]};
        break;
    else
        vecsRL{i-2} = V;
        currentRank = size(V,2);
    end
end

%% Select the correct vectors from the two decompositions

fprintf("\n");
fprintf("Selecting correct vectors from decompositions... \n");

vecs = cell([1,N-2]);
ranks = zeros(1,N-2);

for i = 1:N-2 % given two sets of orthonormal vectors from the two decompositions, the
    % correct set of vectors is the one with greater cardinality
    rankLR = size(vecsLR{i},2);
    rankRL = size(vecsRL{i},2);
    if rankLR >= rankRL
        vecs{i} = vecsLR{i};
        ranks(i) = rankLR;
    else
        vecs{i} = vecsRL{i};
        ranks(i) = rankRL;
    end
end

%% Find the coefficients

fprintf("Constructing coefficients tensor... \n");

C = zeros(ranks);
I = reverseLexicographic(ranks); % MATLAB's one-dimensional array indexing is in
% reverse-lexicographic order, so to construct the tensor of coefficients,
% we need to generate all possible indices in reverse-lexicographic order
% and then store the products of the coefficients into a tensor in this
% order
for i = 1:prod(ranks)
    contractors = cell([1,N-2]);
    for j = 1:N-2
        V = vecs{j};
        contractors{j} = V(:,I(i,j));
    end
    if N > 3
        dots = zeros(1,N-3);
        for j = 1:N-3
            dots(j) = dot(contractors{j},contractors{j+1});
        end
    else
        dots = 1;
    end
    contractors = [{contractors{1}}, contractors];
    contractors{end+1} = contractors{end};
    C(i) = ttv(T,contractors,1:N)/prod(dots);
end

fprintf("Running alternating least squares on coefficients tensor... \n");

sol = cp_als(tensor(C),1); % for a rank-1 tensor, alternating least squares appears
% to converge in one iteration

coeffs = cell([1,N-2]);

fprintf("\n");
fprintf("Organizing solution from CP_ALS... \n");

ranksAcc = 1;
ranksSol = size(sol);
for i = 1:length(ranksSol)
    for j = ranksAcc:N-2
        if ranksSol(i) == ranks(j)
            if i == 1
                coeffs{j} = sol.U{i}*sol.lambda;
                ranksAcc = j + 1;
                break;
            else
                coeffs{j} = sol.U{i};
                ranksAcc = j + 1;
                break;
            end
        else
            coeffs{j} = 1;
        end
    end
end

if ranksAcc <= N-2
    for j = ranksAcc:N-2
        coeffs{j} = 1;
    end
end

fprintf("Done. \n");

end

