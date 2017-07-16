function x = vecpostproc(x, a)
% VECPOSTPROC is post-processing of a D-dimensional vector.
%   
%   V = vecpostproc(V) outputs L2 normalized vector:
%     V = V ./ L2NORM(V);
%
%   V = vecpostproc(V, A) outputs L2 and power-law normalized vector:
%     V = SIGN(X) .* ABS(X) .^ A;
%     V = V ./ L2NORM(V);
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 

    if ~exist('a'), a = 1; end
    x = replacenan (l2_normalize (powerlaw (x, a)));

function x = l2_normalize(x)
    l = sqrt(sum(x.^2));
    x = bsxfun(@rdivide,x,l);
    x = replacenan(x);

function x = powerlaw (x, a)
	if a == 1, return; end
	x = sign (x) .* abs(x)  .^ a;

function y = replacenan (x, v)
	if ~exist ('v')
	  v = 0;
	end
	y = x;
	y(isnan(x)) = v;