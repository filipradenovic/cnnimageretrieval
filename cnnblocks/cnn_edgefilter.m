function [y1, y2] = cnn_edgefilter(X, params, varargin)
% CNN_EDGEFILTER is an edge filtering function.
%   Y = cnn_edgefilter(X, PARAMS) computes filtered edge map:
%
%     Y = (W .* X.^P)./(exp(-S .* (X - T)) + 1);
%
%     W = PARAMS(1); P = PARAMS(2); S = PARAMS(3); T = PARAMS(4);
%   
%   X should be I x J x D x NIM, and PARAMS should be 1 x 4
%
%   [DZDX, DZDP] = cnn_edgefilter(X, PARAMS, DZDY) computes the derivative
%   of the block inputs projected onto DZDY. DZDX, DZDP and DZDY have the
%   same dimensions as X, PARAMS and Y, respectively.
%
%   cnn_edgefilter(___, 'OPT', VAL, ...) accepts the following options:
%
%   `Epsilon`:: 1e-6
%      When computing derivatives, quantities that are divided in are
%      lower bounded by this value.
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2018. 

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

w = params(1);
p = params(2);
s = params(3);
t = params(4); if (t<0.01), t=0.01; end

if isempty(dzdy)

  % Forward pass

  y1 = (w .* X.^p) ./ (exp(-s .* (X - t)) + 1);

else

  % Backward pass (DERIVATIVES)

  y1 = (X.^(p - 1) .* p .* w) ./ (exp(-s .* (X - t)) + 1) + (X.^p .* s .* w .* exp(-s .* (X - t))) ./ (exp(-s .* (X - t)) + 1).^2;
  y1 = y1 .* dzdy;
  y1(X==0) = 0;

  tmp = (X .^p ./ (exp(-s .* (X - t)) + 1)) .* dzdy; 
  tmp(X==0) = 0;
  y2(1) = sum(tmp(:));

  tmp = ((X.^p .* w .* log(X)) ./ (exp(-s .* (X - t)) + 1)) .* dzdy;
  tmp(X==0) = 0;
  y2(2) = sum(tmp(:));

  tmp = ((X.^p .* w .* exp(-s .* (X - t)) .* (X - t)) ./ (exp(-s .* (X - t)) + 1).^2) .* dzdy;
  tmp(X==0) = 0;
  y2(3) = sum(tmp(:));

  tmp = (-(X.^p .* s .* w .* exp(-s .* (X - t))) ./ (exp(-s .* (X - t)) + 1).^2) .* dzdy;
  tmp(X==0) = 0;
  y2(4) = sum(tmp(:));

end

%% Computational errors
% if (sum(isinf(y1(:)))), disp(params), error('Some elements of output are inf!!!'); end;
% if (sum(isnan(y1(:)))), disp(params), error('Some elements of output are NaN!!!'); end;
% if ~isempty(dzdy)
%   if (sum(isinf(y2(:)))), disp(params), error('Some elements of output are inf!!!'); end;
%   if (sum(isnan(y2(:)))), disp(params), error('Some elements of output are NaN!!!'); end;
%   fprintf('\nedgelayer || w %.4f || p %.4f || s %.4f || t %.4f\n', w, p, s, t);
% end