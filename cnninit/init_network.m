function net = init_network(opts)
% INIT_NETWORK loads pre-trained model and transfer it to DagNN if needed.
% Next, it prepares network for training by removing and adding necessary layers.
%
%   NET = init_network(opts)
%   
%   Check supported opts in function LOAD_OPTS_TRAIN.M
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 

	%% Load pretrained model to initialize
	fprintf('>> Loading pre-trained model %s\n', opts.model);
	net_in = load(opts.modelPath);
	if ~opts.isDagnn % simpleNN
		net_in = dagnn.DagNN.fromSimpleNN(net_in, 'canonicalNames', true);
	else 
	 	net_in = dagnn.DagNN.loadobj(net_in);	
	end

	%% Build network architecture
	net = build_network(net_in, opts);

	%% For GOOGLENET remove side classification branches
	if strcmp(opts.model, 'GOOGLENET')
	  net = remove_blocks_with_name(net, 'cls');
	end

	%% Scaling avg image, for sketches we scale it with 0
	if opts.averageImageScale ~= 1
		fprintf('>> Scaling averageImage by %.2f\n', opts.averageImageScale);
		net.meta.normalization.averageImage = net.meta.normalization.averageImage * opts.averageImageScale;
	end

	%% Merging Batch Norm (if exists) with the preceding Conv layer
	if opts.mergeBatchNorm
	  net = merge_batch_norm(net);
	end