function net = remove_blocks_with_name(net, name)
% REMOVE_BLOCKS_WITH_NAME function removes layers that contain specific name.
%
%   NET = remove_blocks_with_name(NET, NAME)
%   
%   NAME is a string. All layers from NET containing this string will be removed from it.
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 
	
	% find layers containing string name
	layers = {};
	for l = 1:numel(net.layers)
		if numel(strfind(net.layers(l).name, name))
			layers{1,end+1} = net.layers(l).name;
		end
	end

	% if no layers return
	if isempty(layers), return; end

	% remove found layers
	fprintf('>> Removing layers that contain word ''%s'' in their name\n', name);
	for i = 1:numel(layers)
		layer = net.layers(net.getLayerIndex(layers{i}));
		net.removeLayer(layers{i});
		net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
	end