function net = build_network(net_in, opts)
% BUILD_NETWORK is a function that removes unneeded layers and 
% appends necessary ones to prepare network as training.
%
%   NET = build_network(NET_IN, opts)
%
%   Check supported opts in function LOAD_OPTS_TRAIN.M
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 

	opts.method = lower(opts.method);
	
	isDagnn = isa(net_in, 'dagnn.DagNN') ;
	if ~isDagnn, error('!!! The network input has to be of dagnn format.'); else
		% removing all layers from lastLayer until the end
		net = net_in;
		f = true;
		while(f)
			if strcmp(net.layers(end).name, opts.lastLayer), f = false;
			else, net.removeLayer(net.layers(end).name);	end
		end
	end
	
	net.renameVar(net.vars(1).name, 'input');
	net.renameVar(net.vars(end).name, 'xx0');
	
	net.meta.outputDim = opts.outputDim;
	net = append_blocks(net, opts.method, opts.objectiveType, opts.errorType, opts.imageChannels); % append pooling, loss, ...
	if ~isDagnn, net.meta.normalization = net_in.normalization; end