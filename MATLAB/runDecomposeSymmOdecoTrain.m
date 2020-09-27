clear all;
close all;
clc;

format long;

%% Runs DecomposeSymmOdecoTrain on a randomly-generated symmetric, odeco tensor train with n components
%% and with length L, and computes the error between the solution and the test tensor

n = 4; % each vector lives in R^n
L = 5; % the length of the train

[T,ranks,vecs,coeffs] = generateTestSymmOdecoTrain(n,L);
[ranks_sol,vecs_sol,coeffs_sol] = decomposeSymmOdecoTrain(T);
T_sol = constructTensor(vecs_sol,coeffs_sol);
error = norm(T - T_sol);
