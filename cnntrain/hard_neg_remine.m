function nidxs = hard_neg_remine(net, ims, clusterids, qidxs, nnum, nbatch, mmapfn)
% HARD_NEGATIVE_REMINE gets a set of hard negative images for given query images.
%
%   NIDXS = hard_negative_remine(NET, IMS, CLUSTERIDS, QIDXS, NUMNEG, NBATCH, MMAP)
%   
%   NET        : current CNN network
%   IMS        : cell array of images in which to search for hard negatives
%   CLUSTERIDS : vector with same size as IMS having cluster ID for each image
%   QIDXS      : vector of indexes of query images
%   NUMNEG     : number of negative images per query image
%   NBATCH     : batch size for forward pass
%   MMAP       : string defining memory map filename for multi-GPU computation
%
%   NIDXS      : [NUMNEG NUMQIDXS] matrix of negative image indexes, one column per query
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017.

%% Net mode for hard negative mining: 'normal' or 'test'
net_mode = 'test';
%% Layer name to take descriptor from
descvarname = 'l2descriptor';
%% Mean pixel value
mpx = mean(net.meta.normalization.averageImage(:));
%% cnndesc dimensionality and number of images
D = net.meta.outputDim;
N = numel(ims);

%% Set up memory map file for cnndesc if multiple gpus used
if numlabs > 1
 	mmap = init_mmap(mmapfn, D, N, 'single');
end
istart = (labindex-1) * ceil(N / numlabs) + 1;
iend = min(labindex * ceil(N / numlabs), N);

%% Initialize cnndesc for this lab
cnndesc = zeros(D, numel(istart:iend), 'single');

%% Compute cnndesc for this lab
fprintf('>> Compute cnndesc for the hard negative mining...\n');
progressbar(0);
for ib = istart:nbatch:iend
	imc = single.empty; pos = 1;
	for i = ib:min(ib+nbatch-1, iend)
		imc(:,:,:,pos) = pad2square(single(ims{i}), mpx) - mpx;
		pos = pos + 1;
	end
	net.mode = net_mode;
	net.eval({'input', gpuArray(reshape(imc, [size(imc), 1]))});	
	cnndesc(:, [ib:min(ib+nbatch-1, iend)] - istart + 1) = ...
		gather(squeeze(net.getVar(descvarname).value));
	progressbar((i-istart+1)/(iend-istart+1));
end
%% Aggregate cnndesc from all labs
if numlabs > 1
	% write this labindex part
	mmap.Data.data(:, istart:iend) = cnndesc;
	% block until every lab reaches here
	labBarrier(); 
	% read the whole thing
	cnndesc = mmap.Data.data;
end
%% Cast it to gpuArray for faster multiplication, ie search
cnndesc = gpuArray(cnndesc);

%% Set up memory map file for nidxs if multiple gpus used
qnum = numel(qidxs);
if numlabs > 1
	mmap = init_mmap(mmapfn, nnum, qnum, 'double');
end
istart = (labindex-1) * ceil(qnum / numlabs) + 1;
iend = min(labindex * ceil(qnum / numlabs), qnum);
qidxs = qidxs(istart:iend);

%% Remine negatives for this lab
fprintf('>> Remine hard negatives...\n');
nskip = 0;
nidxs = [];
pgs_cnt = 0;
progressbar(0);
for q = qidxs
	% get cluster of query image
	qclust = clusterids(q);
	% keep only images of other clusters
	neg = find(clusterids ~= qclust);
	% sort by hardness
	[~, sidx] = sort(cnndesc(:, q)' * cnndesc(:, neg), 'descend');
	neg = neg(sidx);
	% keep the hardest ones with constraint of 1 per cluster
	[~, uidx] = unique(clusterids(neg), 'stable');
	neg = neg(uidx);
	% take nnum hardest
	nidxs = [nidxs neg(1+nskip:nnum+nskip)']; 
	% progress count and print, doesnt work with multiple gpus
	pgs_cnt = pgs_cnt + 1;
	progressbar(pgs_cnt/numel(qidxs)); 
end
if numlabs > 1
	% write this labindex part
	mmap.Data.data(:, istart:iend) = nidxs;
	% block until every lab reaches here
	labBarrier(); 
	% read the whole thing
	nidxs = mmap.Data.data;
	% delete the mmap file
	delete_mmap(mmapfn);
end

% --------------------------------------------------------------------------------------------------
function mmap = init_mmap(mmapfn, nrow, ncol, precision)
% --------------------------------------------------------------------------------------------------

%% Delete if exist
delete_mmap(mmapfn);

%% Open and init a new mmap file
if ~exist(mmapfn) && (labindex == 1)
	f = fopen(mmapfn,'wb');
	fwrite(f, zeros(nrow, ncol, precision), precision);
	fclose(f);
end
labBarrier(); 
mmap = memmapfile(mmapfn, 'Format', {precision, [nrow, ncol], 'data'}, 'Writable', true) ;

% --------------------------------------------------------------------------------------------------
function delete_mmap(mmapfn)
% --------------------------------------------------------------------------------------------------

%% Delete if exists
if exist(mmapfn, 'file') && (labindex == 1)
	fprintf('>> Deleting memory map file: %s\n', mmapfn);
	delete(mmapfn);
end
labBarrier(); 