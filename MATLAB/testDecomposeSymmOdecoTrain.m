clear all;
close all;
clc;

format long;

%% Runs a test of DecomposeSymmOdecoTrain with varying number of components, lengths, ranks, and noises

n = 4; % each vector lives in R^n
L = 3; % the length of the train
rankSame = 2; % the rank of each carriage

trials = 100;
noises = [0,10^(-6),10^(-2)];
errors = zeros(length(noises),trials);
runtimes = zeros(length(noises),trials);

for j = 1:length(noises)
    for i = 1:trials
        
        tic
        [T,ranks,vecs,coeffs] = generateTestSymmOdecoTrainTesting(n,L,rankSame);
        N = tensor(normrnd(0,1,n*ones(1,L+2)));
        T = T + noises(j)*(norm(T)/norm(N))*N;
        [ranks_sol,vecs_sol,coeffs_sol] = decomposeSymmOdecoTrain(T);
        T_sol = constructTensor(vecs_sol,coeffs_sol);
        errors(j,i) = norm(T - T_sol)/norm(T);
        runtimes(j,i) = toc;
        
    end
end

avg_errors = mean(log10(errors'));
avg_runtimes = mean(runtimes');
