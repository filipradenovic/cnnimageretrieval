function [y1] = cnn_batchmap(x, label, dzdy)
% CNN_BATCHMAP computes Mean Average Precision (MAP) for each query in a batch:
%   NQ query tuples, each packed in the form of (q,p,n1,..nN)
%
%   Y = cnn_batchmap(X, L) computes MAP for vectors X with labels L.
%   X has dimension 1 x 1 x D x N packing N vectors with D-dimensions. L has dimension 1 x N.
%   L has value -1 for each query image, each query is followed by 1 for positive and 0 for negative images.
%   
%   DZDX = cnn_batchmap(X, L, DZDY) computes the derivative of the block
%   projected onto DZDY. DZDX and DZDY have the same dimensions as X and Y respectively.
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 
  
  x = squeeze(x); % reshape from 1 x x D x N to D x N
  dim = size(x, 1); % D
  nq = sum(label == -1); % number of query tuples
  S  = size(x, 2) / nq; % number of images per query tuple (including query): 1 + 1 + n
  c = label;

  x1 = reshape(repmat(x(:, 1:S:end), [S-1, 1]), dim, (S-1)*nq);
  x2 = x;
  x2(:, 1:S:end) = [];    

  c = single(c);
  c(1:S:end) = [];

  % euclidean distance for all pairs
  D = sqrt(sum((x1 -x2) .^ 2));

  % compute ap for each query batch (S-1 = #pos + #neg)
  y1 = zeros('like',x);
  for q = 1:S-1:size(x1,2)
    pairsq = q:q-1+S-1;
    cq = c(pairsq);
    [~, idxq] = sort(D(pairsq));
    posq = find(cq(idxq) == 1);
    y1 = y1 + compute_ap(posq, numel(posq));
  end
  y1 = 1 - y1/nq;

  if nargin <= 2 || isempty(dzdy)
    return; 
  end

  y1 = zeros(size(x), 'like', x);
  