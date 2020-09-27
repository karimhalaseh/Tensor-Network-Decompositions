function [T,ranks,vecs,coeffs] = generateTestSymmOdecoTrain(n,L)
% Creates a generic symmetric, odeco tensor train of length L whose components belong
% in R^n and which satisfies the Decreasing Rank Condition

% Inputs: n >= 1, the dimension of the component vectors in R^n
%         L >= 1, the length of the tensor train
% Outputs: tensor: T, a generic symmetric, odeco tensor train of length L
%                     satisfying the Decreasing Rank Condition
%          1xL double: ranks, a positive-integer array specifying the 
%                             randomly-generated ranks of each "carriage" of the train,
%                             from left to right, which also satisfy the 
%                             Decreasing Rank Condition
%          1xL cell{nx? double}: vecs, a cell array containing the generic,
%                                      orthonormal vectors used to generate the "carriages" in the train
%          1xL cell{1x? double}: coeffs, a cell array containing the
%                                        randomly-generated coefficients multiplying each symmetric
%                                        tensor term

assert(n>=1,'The dimension of the component vectors in R^n must be >= 1.');
assert(L>=1,'The length of the tensor train must be >= 1.');

%% Generate ranks

rand_init = randi(n); % choose a random rank to decrease from
ranks_left = sort(randi(rand_init,[1,L]),'descend'); % generate random, decreasing ranks from the left
rand_init = randi(n); % choose a random rank to decrease from
ranks_right = sort(randi(rand_init,[1,L]),'ascend'); % generate random, decreasing ranks from the right
ranks = max(ranks_left,ranks_right); % a randomly-generated array of ranks satisfying the Decreasing Rank Condition

%% Construct tensor

tol_orth = 10^(-14); % tolerance for error in orthogonality
vecs = cell([1,L]);
coeffs = cell([1,L]);
a = -10; % lower bound for coefficients
b = 10; % upper bound for coefficients

% Create the first "carriage" of the train

A = RandOrthMat(n,tol_orth); % randomly generate n orthonormal vectors
B = A(:,1:ranks(1)); % take only ranks(1) of those vectors
vecs{1} = B;
lambda = a + (b-a).*rand(ranks(1),1); % randomly generate coefficients
coeffs{1} = lambda;

A = 0;
for i = 1:ranks(1)
    A = A + lambda(i)*reshape(kron(kron(B(:,i),B(:,i)),B(:,i)),n,n,n);
end

T = tensor(A);

% Create the other "carriages" and contract

for j = 2:L
    A = RandOrthMat(n,tol_orth);
    B = A(:,1:ranks(j));
    vecs{j} = B;
    lambda = a + (b-a).*rand(ranks(j),1);
    coeffs{j} = lambda;
    
    A = 0;
    for i = 1:ranks(j)
        A = A + lambda(i)*reshape(kron(kron(B(:,i),B(:,i)),B(:,i)),n,n,n);
    end
    
    A = tensor(A);
    T = ttt(T,A,ndims(T),1); % contract
end

end

