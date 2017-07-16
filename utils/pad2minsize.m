function x = pad2minsize(x, minsize, v)
% PAD2MINSIZE pads INPUT with given VALUE so that both spatial sizes are at least MINSIZE.
%
%   X = pad2minsize(X, MINSIZE, V)
%     Default minimum image size MINSIZE is 70. 
%     Default padding value V is 0.

	if ~exist('v'), v = 0; end
	if ~exist('minsize'), minsize = 70; end
	x = padarray(x, [max(minsize-size(x,1), 0), max(minsize-size(x,2), 0)], v, 'post');