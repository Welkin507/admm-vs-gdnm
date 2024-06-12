clear; clc;
addpath('./exploration/');

rng('default');
m=4096;
n=4096;
A=randn(m,n);
b=randn(m,1);
mu=1e-3;

tic;
x = lasso_GDNM2(A,b,mu);
t = toc;
fprintf('Time: %f\n',t);

tic;
x = lasso_ADMM_ultra(A, b, mu);
t = toc;
fprintf('Time: %f\n',t);

