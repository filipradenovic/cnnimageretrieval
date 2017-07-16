function x = whitenapply(x, m, P, dimensions)
% WHITENAPPLY applies learned whitening on the given data,
% with a possible dimensionality reduction.
% 
%   X = whitenapply(X, M, P)
%     Centering by mean M, projection by P and L2 normalization.
%
%   X = whitenapply(X, M, P, DIM)
%     Centering by mean M, projection by P and L2 normalization.
%     DIM defines dimensionality reduction. Default is size(P, 2).
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 

if nargin < 4
  dimensions = size(P,2);
end

x = P(1:dimensions,:) * bsxfun(@minus,x,m);
x = vecpostproc(x);