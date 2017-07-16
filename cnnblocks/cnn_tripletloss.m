function [y] = cnn_tripletloss(x, label, m, dzdy)
% CNN_TRIPLETLOSS computes triplet loss for each query in a batch:
%   NQ query tuples, each packed in the form of (q,p,n1,..nN)
%
%   Y = cnn_tripletloss(X, L, M) computes contrastive loss with margin M, for vectors X with labels L.
%     For a triplet (q, p, n) triplet loss is:
%     Y(q,p,n) = MAX(0, L2DIST(X(:,:,:,q), X(:,:,:,p))^2 + M - L2DIST(X(:,:,:,q), X(:,:,:,n))^2)
%   
%   X has dimension 1 x 1 x D x N packing N vectors with D-dimensions. L has dimension 1 x N.
%   L has value -1 for each query image, each query is followed by 1 for positive and 0 for negative images.
%   
%   DZDX = cnn_tripletloss(X, L, M, DZDY) computes the derivative of the block
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
  
  xq = x1(:, 1:(S-1):end);
  xp = x2(:, 1:(S-1):end);
  xn = x2;
  xn(:,1:(S-1):end) = [];

  % Forward pass
  y = 0;
  if nargin <= 3 || isempty(dzdy)
    for q = 1:nq
      for j = 1:(S-2)
        dpos = sum( (xq(:, q) - xp(:, q)).^2 );
        dneg = sum( (xq(:, q) - xn(:, (q-1)*(S-2) + j)).^2 );
        if dpos + m - dneg < 0
          continue;
        end
        y = y + dpos + m - dneg;
      end
    end    
    return;
  end

  % Backward pass
  dzdx = zeros(size(x), 'like', x);
  for q = 1:nq
    for j = 1:(S-2)
      dpos = sum( (xq(:, q) - xp(:, q)).^2 );
      dneg = sum( (xq(:, q) - xn(:, (q-1)*(S-2) + j)).^2 );
      if dpos + m - dneg < 0
        continue;
      end
      % for queries
      dzdx(:, (q-1) * S + 1) = dzdx(:, (q-1) * S + 1) + 2 .* (-xp(:, q) + xn(:, (q-1)*(S-2) + j));
      % for positives
      dzdx(:, (q-1) * S + 2) = dzdx(:, (q-1) * S + 2) - 2 .* (xq(:, q) - xp(:, q));
      % for negatives
      dzdx(:, (q-1) * S + 2 + j) = dzdx(:, (q-1) * S + 2 + j) + 2 .* (xq(:, q) - xn(:, (q-1)*(S-2) + j));
    end
  end    

  y = dzdy .* dzdx; 
  y = reshape(y, [1, 1, size(y)]);

  