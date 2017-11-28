function Lw = whitenlearn (x, qidxs, pidxs)
% WHITENLEARN learns discriminative whitening on given data.
%  Lw = whitenlearn (X, QIDXS, PIDXS)
%
%   Learn discriminative whitening Lw using annotations
%   
%   Input: 
%     X     : descriptors 
%     QIDXS : query idxs
%     PIDXS : positive idxs
%   
%   Output:
%     Lw.P   : Whitening projection, learned on query-pos pairs
%     Lw.m   : Mean of query descriptors, for centering
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