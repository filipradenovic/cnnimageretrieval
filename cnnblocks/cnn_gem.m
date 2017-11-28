function [y1, y2] = cnn_gem(x, p, varargin)
% CNN_GEM is a CNN Generalized Mean Pooling of convolution activations.
%   Y = cnn_gem(X, P) computes the P norm over spatial locations i and j:
%
%     Y(i,j,d) = N^(-1/P) (SUM_ij (X(i,j,d)^P)^(1/P)
%
%   X should be I x J x D x NIM, and P should be 1 x 1 x D or 1 x 1 x 1
%
%   [DZDX, DZDP] = cnn_gem(X, P, DZDY) computes the derivative
%   of the block inputs projected onto DZDY. DZDX, DZDP and DZDY have the
%   same dimensions as X, P and Y, respectively.
%
%   cnn_gem(___, 'OPT', VAL, ...) accepts the following options:
%
%   `Epsilon`:: 1e-6
%      When computing derivatives, quantities that are divided in are
%      lower bounded by this value.
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 

% -------------------------------------------------------------------------
%                                                             Parse options
% -------------------------------------------------------------------------

opts.epsilon = 1e-6 ;
backMode = numel(varargin) > 0 && ~ischar(varargin{1}) ;
if backMode
  dzdy = varargin{1} ;
  opts = vl_argparse(opts, varargin(2:end), 'nonrecursive') ;
else
  dzdy = [] ;
  opts = vl_argparse(opts, varargin, 'nonrecursive') ;
end

% -------------------------------------------------------------------------
%                                                             Parse options
% -------------------------------------------------------------------------

% trimming p
t = p;
t(p < 1) = 1;

% image size normalization factor
N = size(x,1) * size(x,2);

% we will use this quite often
xt = bsxfun(@power, x, t);
xtsum = max(N .* vl_nnpool(xt, [size(x,1), size(x,2)], 'method', 'avg', 'cuDNN'), opts.epsilon);

if isempty(dzdy)
  % forward pass
  y1 = N.^(-1./t) .* bsxfun(@power, xtsum, 1./t);
else
  % backward pass (DERIVATIVES)
  % input der
  yt = N.^(-1./t) .* bsxfun(@power, xtsum, 1./t);
  y1 = (1/N) .* bsxfun(@times, bsxfun(@power, yt, 1-t), bsxfun(@power, x, t-1));
  y1 = bsxfun(@times, dzdy, y1);
  % params der
  xtlogxsum = N .* vl_nnpool(xt .* log(max(x, opts.epsilon)), [size(x,1), size(x,2)], 'method', 'avg', 'cuDNN');
  y2 = bsxfun(@times, yt, 1./t.^2) .* ( log(N) - log(xtsum) + ...
       bsxfun(@times, t, xtlogxsum ./ xtsum) );
  y2 = bsxfun(@times, y2, dzdy);
  y2 = sum(y2, 4);
  % if only one p sum params der
  if numel(p) == 1
    y2 = sum(y2, 3);
  end
end
