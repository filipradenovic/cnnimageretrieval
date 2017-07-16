function [Lw, PCAw] = whitenlearn (x, qidxs, pidxs)
% WHITENLEARN learns PCA and discriminative whitening on given data.
%  [Lw, PCAw] = whitenlearn (X, QIDXS, PIDXS)
%
%   Learn discriminative whitening Lw using annotations
%   Learn generative whitening PCAw without annotations
%   
%   Input: 
%     X     : descriptors 
%     QIDXS : query idxs
%     PIDXS : positive idxs
%   
%   Output:
%     Lw.P   : Whitening projection, learned on query-pos pairs
%     Lw.m   : Mean of query descriptors, for centering
%     PCAw.P : Whitening projection, learned unsupervised
%     PCAw.m : Mean of all descriptors, for centering
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 

D = size(x,1);
N = size(x,2);

%% Learning Lw w annotations
m = mean(x(:,qidxs),2);
df = x(:,qidxs) - x(:,pidxs);
S = df * df' ./ size(df, 2);
P = inv(chol(S))';
df = P * bsxfun(@minus, x, m);
D = df * df';
[V, eigval] = eig(D);
[eigval, ord] = sort(diag(eigval),'descend');
V = V(:,ord);

Lw.P = V' * P;
Lw.m = m;

%% Learning PCA w/o annotations
m = mean(x,2);
xc = bsxfun(@minus, x, m);
xcov = xc * xc';
xcov = (xcov + xcov') / (2 * N);
[eigvec, eigval] = eig (xcov);
[eigval, ord] = sort(diag(eigval),'descend');
eigvec = eigvec(:, ord);

PCAw.P = inv(sqrt(diag(eigval))) * eigvec';
PCAw.m = m;