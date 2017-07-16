function v = cnn_vec(im, net)
% CNN_VEC computes a D-dimensional CNN vector for an image.
%
%   V = cnn_vec(IM, NET)
%   
%   IM  : Input image, or input image path as a string
%   NET : CNN network to evaluate on image IM
%
%   V   : D-dimensional vector
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 

	minsize = 67;
	net.mode = 'test';
	descvarname = 'l2descriptor';

	use_gpu = strcmp(net.device, 'gpu');
	if use_gpu
		gpuarrayfun = @(x) gpuArray(x);
		gatherfun = @(x) gather(x);
	else
		gpuarrayfun = @(x) x; % do not convert to gpuArray
		gatherfun = @(x) x; % do not gather
	end

	if isstr(im)
		im = imread(im);
	end
	
	im = single(im) - mean(net.meta.normalization.averageImage(:));
	if size(im, 3) == 1
		im = repmat(im, [1 1 3]);
	end
	if min(size(im, 1), size(im, 2)) < minsize
		im = pad2minsize(im, minsize, 0);
	end

	net.eval({'input', gpuarrayfun(reshape(im, [size(im), 1]))});

	v = gatherfun(squeeze(net.getVar(descvarname).value));