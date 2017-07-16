function [inputs, nidxs, time_remine] = get_batch(opts, db, net, batch, subbatch, mode, epoch)
% GET_BATCH gets a batch (or a subBatch) of images for one iteration of training.
% This function also performs negative re-mining when required.
%
%   [INPUTS, NIDXS, TIME_REMINE] = get_batch(opts, DB, NET, BATCH, SUBBATCH, MODE, EPOCH)
%
%   Check supported opts in function LOAD_OPTS_TRAIN.M
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 

time_remine = 0;

nbatch = floor(numel(db.(mode).qidxs)/opts.batchSize);
remine_after = ceil(nbatch/opts.numRemine);

% check if it is time to perform re-mining
if (strcmp(mode, 'train') && rem(batch-1, remine_after)==0 && subbatch==1) || (strcmp(mode, 'val') && batch == 1 && subbatch==1)
	fprintf('>> Remining negatives\n'); start_remine = tic;
	% clear intermediate values and der of training
	net = resetKeepStats(net);
	% range of batches for which we want to remine negatives
	if strcmp(mode, 'train')
		batch_start = batch;
		batch_end   = batch + remine_after - 1;
	else
		batch_start = 1;
		batch_end   = nbatch;
	end
	% get all queries from this batch and next "remine_after" batches
	rmn_idx = (batch_start-1)*opts.batchSize+1 : min(batch_end*opts.batchSize, numel(db.(mode).qidxs));
	qidxs_rmn = db.(mode).qidxs(rmn_idx);
	nnum = size(db.(mode).nidxs,1);
	remineBatchSize = ceil(ceil(opts.batchSize/numlabs)/opts.numSubBatches);
	db.(mode).nidxs(:, rmn_idx) = hard_neg_remine(net, db.(mode).data, db.(mode).cluster, qidxs_rmn, nnum, remineBatchSize*(2+nnum), opts.memoryMapRemine);
	time_remine = toc(start_remine);
end

% prepare idxs for this batch
btch_idx = (batch-1)*opts.batchSize+1 : min((batch)*opts.batchSize, numel(db.(mode).qidxs));
% take only idxs for this lab if more than one gpu
btch_idx = btch_idx(labindex:numlabs:end);
% take only idxs for this subbatch
btch_idx = btch_idx(subbatch:opts.numSubBatches:end);
% take qidx, pidx and nidx for this batch and lab
qidxs_btch = db.(mode).qidxs(:, btch_idx);
pidxs_btch = db.(mode).pidxs(:, btch_idx);
nidxs_btch = db.(mode).nidxs(:, btch_idx);

% prepare pairs input idxs, in tuple format (1-query 1-pos N-neg)
input_idxs = [qidxs_btch; pidxs_btch; nidxs_btch];
input_idxs = input_idxs(:)';

% prepare input labels: q: -1; p: 1; n: 0
input_lbls = repmat([-1, ones(1, size(pidxs_btch, 1), 'single'), zeros(1, size(nidxs_btch, 1), 'single')], 1, numel(qidxs_btch));

% prepare input images
mpx = mean(net.meta.normalization.averageImage(:));
input_ims  = single.empty;
for i = 1:numel(input_idxs)
	input_ims(:, :, :, i) = pad2square(single(db.(mode).data{input_idxs(i)}), mpx) - mpx;
end

% cast to gpu if necessary
if numel(opts.gpus) > 0
  input_ims  = gpuArray(input_ims);
  input_lbls = gpuArray(input_lbls);
end

% return inputs data structure that network expects
inputs = {'input', input_ims, 'label', input_lbls};
% return the current "fresh" negatives
nidxs = db.(mode).nidxs;

% -------------------------------------------------------------------------
function net = resetKeepStats(net)
% -------------------------------------------------------------------------
[net.vars.value] = deal([]);
[net.vars.der] = deal([]);
[net.params.der] = deal([]);