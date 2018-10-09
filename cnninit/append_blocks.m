function net = append_blocks(net, method, objectivetype, errortype, imchs)
% APPEND_BLOCKS function adds necessary blocks to the CNN.
%   
%   NET = append_blocks(NET, METHOD, OBJECTIVE, ERROR, IMCHANNELS)
%
%   METHOD is a string that can contain one of the pooling strings:
%     'mac'   - global max pooling layer
%     'spoc'  - global average pooling layer
%     'gem'   - global generalized mean pooling layer with one p for all conv filters
%     'gemmp' - global generalized mean pooling layer with multiple p for each conv filter
%	If METHOD contains string 'edgefilter', the network will be pre-pended with the EdgeFilter layer.
%
%   OBJECTIVE & ERROR are a cell array with 2 values:
%     {'contrastiveloss', M} - contrastive loss with margin M
%     {'tripletloss', M}     - triplet loss with margin M
%     {'batchmap'}           - mean average precision for all queries in a batch
%
%   IMCHANNELS is a number of image channels that will be used for training, ie. 1 or 3.
%     If IMCHANNELS == 1 channels of first conv layers are aggregated using average pooling.
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 

	import dagnn.*;

	if ~exist('imchs'), imchs = 3; end
	if imchs == 1 	% sum aggregate first conv layer, if 1 input channel
		net.params(1).value = sum(net.params(1).value,3);
		net.layers(1).block.size(3) = 1;
	end
	method = lower(method);

	% Edge filtering layer
	if strfind(method, 'edgefilter')
		inputs = {'input'};
		outputs = {'x0'};
		params = struct('name', {}, 'value', {}, 'learningRate', [], 'weightDecay', []);
		name = 'edgelayer';
		block = EdgeFilter();

		params(1).name  = 'edgelayer_w'; params(1).value = single(10);
		params(2).name  = 'edgelayer_p'; params(2).value = single(0.5);
		params(3).name  = 'edgelayer_s'; params(3).value = single(500);
		params(4).name  = 'edgelayer_t'; params(4).value = single(0.1);

		net.addLayer(name, block, inputs, outputs, {params.name});

		% edgelayer_w params.value
		i = 1; f = net.getParamIndex(params(i).name);
		net.params(f).value = params(i).value; net.params(f).learningRate = 0; net.params(f).weightDecay = 0;
		% edgelayer_p params.value
		i = 2; f = net.getParamIndex(params(i).name);
		net.params(f).value = params(i).value; net.params(f).learningRate = 1; net.params(f).weightDecay = 0;
		% edgelayer_s params.value
		i = 3; f = net.getParamIndex(params(i).name);
		net.params(f).value = params(i).value; net.params(f).learningRate = 0; net.params(f).weightDecay = 0;
		% edgelayer_t params.value
		i = 4; f = net.getParamIndex(params(i).name);
		net.params(f).value = params(i).value; net.params(f).learningRate = 0.1; net.params(f).weightDecay = 0;
		
		net.layers(1).inputs{1} = outputs{1};
	end

	l = 0;

	% Global pooling layer
	l = l + 1;
	inputs = {sprintf('xx%d',l-1)};
	outputs = {'pooldescriptor'};
	params = struct('name', {}, 'value', {}, 'learningRate', [], 'weightDecay', []);
	name = 'pooldescriptor' ;
	if strfind(method, 'mac')
		block = MAC();
	elseif strfind(method, 'spoc')
		block = SPOC();
	elseif strfind(method, 'gem')
		block = GeM();
		if strfind(method, 'gemmp'), sz = [1 1 D 1]; else, sz = [1 1 1 1]; end
		params(1).name = 'gem';
		params(1).value = ones(sz, 'single') * 3;
		params(1).learningRate = 10;
		params(1).weightDecay  = 0;
	else
		error(sprintf('Unknown method %s\n', method));
	end
	net.addLayer(name, block, inputs, outputs, {params.name});
	if numel(strfind(method, 'gem'))
		findex = net.getParamIndex(params(1).name) ;
		if ~isempty(params(1).value)
			net.params(findex).value = params(1).value ;
			net.params(findex).learningRate = params(1).learningRate;
			net.params(findex).weightDecay = params(1).weightDecay;
		end
	end
	
	% L2 normalization layer
	inputs = {'pooldescriptor'};
	outputs = {'l2descriptor'};
	params = struct('name', {}, 'value', {}, 'learningRate', [], 'weightDecay', []);
	name = 'l2descriptor';
	block = L2N();
	net.addLayer(name, block, inputs, outputs, {params.name});

    % Objective layer
	inputs{1} = 'l2descriptor';
	inputs{2} = 'label';
	outputs = {'objective'};
	params = struct('name', {}, 'value', {}, 'learningRate', [], 'weightDecay', []);
	name = 'objective';
	switch objectivetype{1}
		case 'contrastiveloss'
			block = ContrastiveLoss('margin', objectivetype{2});
		case 'tripletloss'
			block = TripletLoss('margin', objectivetype{2});
		otherwise
			ferror('Unknow loss type\n');
	end
	net.addLayer(name, block, inputs, outputs, {params.name});

    % Error layer
	if exist('errortype') & numel(errortype)
		inputs{1} = 'l2descriptor';		
		inputs{2} = 'label';
		outputs = {'error'};
		params = struct('name', {}, 'value', {}, 'learningRate', [], 'weightDecay', []);
		name = 'error';
		switch errortype{1}
			case 'contrastiveloss'
				block = ContrastiveLoss('margin', objectivetype{2});	
			case 'tripletloss'
				block = TripletLoss('margin', objectivetype{2});
			case 'batchmap' 
				block = BatchMAP();
			otherwise
				ferror('Unknow error type\n');
		end
		net.addLayer(name, block, inputs, outputs, {params.name});
	end