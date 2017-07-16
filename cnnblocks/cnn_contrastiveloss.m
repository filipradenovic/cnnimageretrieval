function [y] = cnn_contrastiveloss(x, label, m, dzdy)
% CNN_CONTRASTIVELOSS computes triplet loss for each query in a batch:
%   NQ query tuples, each packed in the form of (q,p,n1,..nN)
%
%   Y = cnn_contrastiveloss(X, L, M) computes contrastive loss with margin M, for vectors X with labels L.
%     For a pair (i, j) contrastive loss is:
%     Y(i,j) = L(i,j) * L2DIST(X(:,:,:,i), X(:,:,:,j))^2 / 2 + (1 - L(i,j)) * MAX(0, M - L2DIST(X(:,:,:,i), X(:,:,:,j)))^2 / 2
%
%   X has dimension 1 x 1 x D x N packing N vectors with D-dimensions. L has dimension 1 x N.
%   L has value -1 for each query image, each query is followed by 1 for positive and 0 for negative images.
%
%   DZDX = cnn_contrastiveloss(X, L, M, DZDY) computes the derivative of the block
%   projected onto DZDY. DZDX and DZDY have the same dimensions as X and Y respectively.
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 

  x = squeeze(x); % reshape from 1 x x D x N to D x N
  dim = size(x, 1); % D
  nq = sum(label == -1); % number of query tuples
  S  = size(x, 2) / nq; % number of images per query tuple (including query): 1 + 1 + n

  x1 = reshape(repmat(x(:, 1:S:end), [S-1, 1]), dim, (S-1)*nq);
  x2 = x;
  x2(:, 1:S:end) = [];    
  N = size(x1, 2);
  d = size(x1, 1);
  label(1:S:end) = [];

  dif = x1 - x2;
  D = sqrt(sum(dif .^ 2)); 
  D(isnan(D)) = 1; 
  D(D==0) = 1; % division with D later

  if nargin <= 3 || isempty(dzdy) 
    % forward pass, just compute the loss
    y = 0.5*label.*D.^2 + 0.5.*(1-label).*max(0, m-D).^2;
    y = sum(y);
    return; 
  end

  % distances below the threshold
  fn = (D <= m); 
  % derivative of loss for matching pairs
  dzdx = repmat(label, d, 1).*dif; 
  % derivative of loss for non-matching pairs
  dzdx = dzdx + repmat(fn, d, 1) .* repmat(1-label, d, 1) .* dif .* (1 - repmat(m, d, N)./repmat(D, d, 1));

  y_ = dzdy .* dzdx; 

  yq_ = reshape(sum(reshape(y_', S-1, dim * size(y_,2)/(S-1)))', nq, dim)';
  y = reshape([yq_; reshape(-y_, [(S-1)*size(x,1), nq])], [size(x,1), nq*S]);
  y = reshape(y, [1, 1, size(y)]);
