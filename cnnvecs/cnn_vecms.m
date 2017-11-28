function v = cnn_vecms(im, net, scales, aggregate)
% CNN_VECMS computes a multi-scale D-dimensional CNN vector for an image.
%
%   V = cnn_vecms(IM, NET, SCALES, AGGREGATE)
%   
%   IM        : Input image, or input image path as a string
%   NET       : CNN network to evaluate on image IM
%   SCALES    : Vector of scales to resize image prior to the descriptor computation (DEFAULT: [1])
%   AGGREGATE : Aggregate over scales if 1, otherwise return one vector per image scale (DEFAULT: 1)
%
%   V         : Multi-scale output, D x 1 if AGGREGATE=1, D x numel(SCALES) if AGGREGATE=0
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 
	
	if ~exist('scales'), scales = 1; end
	if ~exist('aggregate'), aggregate = 1; end

	minsize = 67;
	net.mode = 'test';
	descvarname = 'l2descriptor';
	poollayername = 'pooldescriptor';
	
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

	v = [];
	for s = scales(:)'
		im_ = imresize(im, s);

		im_ = single(im_) - mean(net.meta.normalization.averageImage(:));
		if size(im_, 3) == 1
			im_ = repmat(im_, [1 1 3]);
		end

		if min(size(im_, 1), size(im_, 2)) < minsize
			im_ = pad2minsize(im_, minsize, 0);
		end

		net.eval({'input', gpuarrayfun(reshape(im_, [size(im_), 1]))});				

		v = [v, gatherfun(squeeze(net.getVar(descvarname).value))];
	end

	if aggregate
		if isa(net.layers(net.getLayerIndex(poollayername)).block, 'GeM')
			p = gatherfun(net.params(net.layers(net.getLayerIndex(poollayername)).paramIndexes).value);
			p = reshape(p, [size(p,3), 1]);
		else 
			p = 1;
		end
		v = bsxfun(@power, v, p);
		v = (sum(v, 2) / size(v, 2)).^(1./p);
		v = vecpostproc(v);
	end