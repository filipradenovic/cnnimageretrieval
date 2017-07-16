function x = pad2square(x, v)
% PAD2SQUARE pads INPUT with given VALUE so that both spatial sizes are the same (equal to larger one).
%
%	X = pad2square(INPUT)
%     Pads INPUT with 0 to SIZE x SIZE. 
%	  SIZE = MAX(SIZE(INPUT,1), SIZE(INPUT,2)).
%
%   X = pad2square(INPUT, VALUE)
%     Pads INPUT with VALUE to SIZE x SIZE.
%	  SIZE = MAX(SIZE(INPUT,1), SIZE(INPUT,2)).

	if ~exist('v'), v = 0; end

	x = padarray(x, [max(size(x))-size(x,1), max(size(x))-size(x,2)], v, 'post');