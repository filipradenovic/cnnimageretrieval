function [PCAw] = pcawhitenlearn (x)
% PCAWHITENLEARN learns PCA whitening on given data.
%  PCAw = pcawhitenlearn (X)
%
%   Learn generative whitening PCAw without annotations
%   
%   Input: 
%     X     : descriptors 
%   
%   Output:
%     PCAw.P : Whitening projection, learned unsupervised
%     PCAw.m : Mean of all descriptors, for centering
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 

N = size(x,2);

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