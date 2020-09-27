function T = constructTensor(vecs,coeffs)
% Given a collection of L sets of orthonormal vectors in R^n 
% and corresponding coefficients for each vector, constructs a symmetric, 
% odeco tensor train of length L

% Inputs: 1xL cell{nx? double}: vecs, a cell array containing the orthonormal 
%                                     vectors used to generate the "carriages" 
%                                     in the train
%         1xL cell{1x? double}: coeffs, a cell array containing the
%                                       coefficients multiplying each symmetric
%                                       tensor term
% Outputs: tensor: T, symmetric, odeco tensor train of length L

L = length(vecs);

B = vecs{1};
lambda = coeffs{1};
n = size(B,1);

A = 0;
for i = 1:size(B,2)
    A = A + lambda(i)*reshape(kron(kron(B(:,i),B(:,i)),B(:,i)),n,n,n);
end

T = tensor(A);

% Create the other "carriages" and contract

for j = 2:L
    B = vecs{j};
    lambda = coeffs{j};
    
    A = 0;
    for i = 1:size(B,2)
        A = A + lambda(i)*reshape(kron(kron(B(:,i),B(:,i)),B(:,i)),n,n,n);
    end
    
    A = tensor(A);
    T = ttt(T,A,ndims(T),1); % contract
end

end

