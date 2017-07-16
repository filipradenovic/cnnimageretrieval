function net = append_blocks(net, method, objectivetype, errortype, imchs)
% APPEND_BLOCKS function adds necessary blocks to the CNN.
%   
%   NET = append_blocks(NET, METHOD, OBJECTIVE, ERROR, IMCHANNELS)
%
%   METHOD is a string:
%     'mac'   - global max pooling layer
%     'spoc'  - global average pooling layer
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

	l = 0;

	% Pooling layer
	l = l + 1;
	inputs = {sprintf('xx%d',l-1)};
	outputs = {'pooldescriptor'};
	params = struct('name', {}, 'value', {}, 'learningRate', [], 'weightDecay', []);
	name = 'pooldescriptor' ;
	if strfind(method, 'mac')
		block = MAC();
	elseif strfind(method, 'spoc')
		block = SPOC();
	else
		error(sprintf('Unknown method %s\n', method));
	end
	net.addLayer(name, block, inputs, outputs, {params.name});
	
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