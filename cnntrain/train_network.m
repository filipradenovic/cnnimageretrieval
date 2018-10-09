function [net, state, stats, db] = train_network(net, getBatch, opts)
% TRAIN_NETWORK  Adapted version of CNN_TRAIN_DAG from MatConvNet package. 
% It performs the training of a CNN using the DagNN wrapper.

% Copyright (C) 2014-16 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.plotStatistics = true;
opts.prefetch = false ;
opts.postEpochFn = [] ;  % postEpochFn(net,params,state) called after each epoch; can return a new learning rate, 0 to stop, [] for no change
opts.extractStatsFn = @extractStats ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end

% -------------------------------------------------------------------------
%                                                         		Prepare data
% -------------------------------------------------------------------------
%
% Load db.mat for the training data and initialize negatives
if exist(opts.dbPath, 'file')
  fprintf('>> Loading DB file...\n'); t=tic;
  db = load(opts.dbPath) ;
  db.train.nidxs = zeros(opts.numNegative, numel(db.train.qidxs));
  db.val.nidxs = zeros(opts.numNegative, numel(db.val.qidxs));
  fprintf('>>>> done in %s\n', htime(toc(t)));
  if ~isfield(db.val, 'data')
    fprintf('>> Validation images not provided, loading...\n');
    fprintf('>>>> imageDir: %s\n', opts.augment.imageDir);
    % we dont use augmentation for validation data
    optsValData.imageDir = opts.augment.imageDir;
    optsValData.crop = opts.augment.crop;
    optsValData.gpus = opts.augment.gpus; 
    optsValData.imageSize = opts.augment.imageSize;
    optsValData.jitterFlip = false; 
    optsValData.jitterLocation = false; 
    optsValData.jitterScale = 1; 
    optsValData.jitterBrightness = 0;
    optsValData.jitterContrast = 0;
    optsValData.jitterSaturation = 0;
    images = cellfun(@(x) cid2filename(x, optsValData.imageDir), db.val.cids, 'UniformOutput', false);
    db.val.data = get_images(images, optsValData);
  end
else
  disp('!!! DB file doesnt exist. KEYBOARD');
  keyboard
end


% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

if isempty(opts.derOutputs)
  error('!!! DEROUTPUTS must be specified when training.\n') ;
end

modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

start = opts.continue * findLastCheckpoint(opts.expDir) ;
if start >= 1
  fprintf('>> Resuming by loading epoch %d\n', start) ;
  [net, state, stats] = loadState(modelPath(start)) ;
else
  state = [] ;
  saveState(modelPath(0), net, state) ;
end

for epoch=start+1:opts.numEpochs
  
  % Set the random seed based on the epoch and opts.randomSeed.
  % This is important for reproducibility, including when training
  % is restarted from a checkpoint.
  rng(epoch + opts.randomSeed) ;
  prepareGPUs(opts, epoch == start+1) ;

  % Train for one epoch.
  params = opts ; 
  params.epoch = epoch ;  
  params.getBatch = getBatch ;
  params.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;

  % Augment data if necessary: epochSize, imread with jitter, flip pairs, etc
  params.db = augment_database(db, opts.augment);

  if numel(opts.gpus) <= 1
    [net, state] = processEpoch(net, state, params, 'train') ;
    [net, state] = processEpoch(net, state, params, 'val') ;
    saveState(modelPath(epoch), net, state) ;
    lastStats = state.stats ;
  else
    spmd
      [net, state] = processEpoch(net, state, params, 'train') ;
      [net, state] = processEpoch(net, state, params, 'val') ;
      if labindex == 1
        saveState(modelPath(epoch), net, state) ;
      end
      lastStats = state.stats ;
    end
    lastStats = accumulateStats(lastStats) ;
  end

  stats.train(epoch) = lastStats.train ;
  stats.val(epoch) = lastStats.val ;
  clear lastStats ;
  saveStats(modelPath(epoch), stats) ;

  if opts.plotStatistics
    switchFigure(1) ; clf ;
    plots = setdiff(cat(2,fieldnames(stats.train)',fieldnames(stats.val)'), {'num', 'time'}) ;
    for p = plots
      p = char(p) ; values = zeros(0, epoch) ; leg = {} ;
      for f = {'train', 'val'}
        f = char(f) ;
        if isfield(stats.(f), p)
          tmp = [stats.(f).(p)] ; values(end+1,:) = tmp(1,:)' ; leg{end+1} = f ;
        end
      end
      subplot(1,numel(plots),find(strcmp(p,plots))) ; plot(1:epoch, values','o-') ;
      xlabel('epoch') ; title(p) ; legend(leg{:}) ; grid on ;
    end
    drawnow ; print(1, modelFigPath, '-dpdf') ;
  end
  
  if ~isempty(opts.postEpochFn)
    if nargout(opts.postEpochFn) == 0
      opts.postEpochFn(net, params, state) ;
    else
      lr = opts.postEpochFn(net, params, state) ;
      if ~isempty(lr), opts.learningRate = lr; end
      if opts.learningRate == 0, break; end
    end
  end

  % With multiple GPUs, return one copy
  if isa(net, 'Composite'), net = net{1} ; end
     
end

% -------------------------------------------------------------------------
function [net, state] = processEpoch(net, state, params, mode)
% -------------------------------------------------------------------------
% Note that net is not strictly needed as an output argument as net
% is a handle class. However, this fixes some aliasing issue in the
% spmd caller.

% create opts needed for getBatch (so that we dont send all)
getbatchopts.batchSize = params.batchSize;
getbatchopts.numSubBatches = params.numSubBatches;
getbatchopts.numRemine = params.numRemine;
getbatchopts.memoryMapRemine = params.memoryMapRemine;
getbatchopts.gpus = params.gpus;

% initialize with momentum 0
if isempty(state) || isempty(state.solverState)
  state.solverState = cell(1, numel(net.params)) ;
  state.solverState(:) = {0} ;
end

% move CNN  to GPU as needed
numGpus = numel(params.gpus) ;
if numGpus >= 1
  net.move('gpu') ;
  for i = 1:numel(state.solverState)
    s = state.solverState{i} ;
    if isnumeric(s)
      state.solverState{i} = gpuArray(s) ;
    elseif isstruct(s)
      state.solverState{i} = structfun(@gpuArray, s, 'UniformOutput', false) ;
    end
  end
end
if numGpus > 1
  parserv = ParameterServer(params.parameterServer) ;
  net.setParameterServer(parserv) ;
else
  parserv = [] ;
end

% profile
if params.profile
  if numGpus <= 1
    profile clear ;
    profile on ;
  else
    mpiprofile reset ;
    mpiprofile on ;
  end
end

num = 0 ;
epoch = params.epoch ;
adjustTime = 0 ;

stats.num = 0 ; % return something even if subset = []
stats.time = 0 ;


start = tic ;
time_remine = 0;

nbatch = floor(numel(params.db.(mode).qidxs)/params.batchSize); % floor so only complete batches
for batch=1:nbatch
  
  % #queries done in this lab and this batch
  bqnum = numel(labindex:numlabs:params.batchSize);

  for subbatch=1:params.numSubBatches
    % #queries done in this lab and this subbatch
    sbqnum = numel(subbatch:params.numSubBatches:bqnum);
    num = num + sbqnum;

    % get inputs and latest negatives
    [inputs, params.db.(mode).nidxs, time_r] = params.getBatch(getbatchopts, params.db, net, batch, subbatch, mode, epoch) ;
    time_remine = time_remine + time_r;

    if params.prefetch
      disp('Prefetch not implemented!!! Keyboard...\n');
      keyboard
    end

    if strcmp(mode, 'train')
      net.mode = 'normal' ;
      net.accumulateParamDers = (subbatch ~= 1) ;
      net.eval(inputs, params.derOutputs, 'holdOn', subbatch < params.numSubBatches) ;
    else
      net.mode = 'test' ;
      net.eval(inputs) ;
    end
  end

  % Accumulate gradient.
  if strcmp(mode, 'train')
    if ~isempty(parserv), parserv.sync() ; end
    posnegNum = size(params.db.(mode).pidxs,1) + size(params.db.(mode).nidxs,1);
    pairNum = params.batchSize * posnegNum; % number of pairs
    imNum   = params.batchSize * (1 + posnegNum); % number of images (for batchNorm)
    state = accumulateGradients(net, state, params, pairNum, imNum, parserv) ;
  end

  % Get statistics.
  time = toc(start) + adjustTime ;
  batchTime = time - stats.time ;
  stats.num = num ;
  stats.time = time ;
  stats = params.extractStatsFn(stats,net) ;
  batchSpeed = batch / (time - time_remine);
  % if t == 3*params.batchSize + 1
  %   % compensate for the first three iterations, which are outliers
  %   adjustTime = 4*batchTime - time ;
  %   stats.time = time + adjustTime ;
  % end

  fprintf('%s: ep %02d: bch %4d/%4d:', mode, epoch, batch, nbatch) ;
  
  fprintf(' %.1f bch/s:', batchSpeed) ;
  for f = setdiff(fieldnames(stats)', {'num', 'time'})
    f = char(f) ;
    fprintf(' %s %.3f:', f(1:3), stats.(f)) ;
  end
  fprintf('\n') ;
end

% Save back to state.
state.stats.(mode) = stats ;
if params.profile
  if numGpus <= 1
    state.prof.(mode) = profile('info') ;
    profile off ;
  else
    state.prof.(mode) = mpiprofile('info');
    mpiprofile off ;
  end
end
if ~params.saveSolverState
  state.solverState = [] ;
else
  for i = 1:numel(state.solverState)
    s = state.solverState{i} ;
    if isnumeric(s)
      state.solverState{i} = gather(s) ;
    elseif isstruct(s)
      state.solverState{i} = structfun(@gather, s, 'UniformOutput', false) ;
    end
  end
end

net.reset() ;
net.move('cpu') ;

% -------------------------------------------------------------------------------
function state = accumulateGradients(net, state, params, pairNum, imNum, parserv)
% pairNum : number of pairs per batch #query x (#pos + #neg)
% imNum   : number of images that pass per batch #query x (1 + #pos + #neg)
% -------------------------------------------------------------------------------

numGpus = numel(params.gpus) ;
otherGpus = setdiff(1:numGpus, labindex) ;

for p=1:numel(net.params)

  if ~isempty(parserv)
    parDer = parserv.pullWithIndex(p) ;
  else
    parDer = net.params(p).der ;
  end

  if isempty(parDer) % for GOOGLENET part that is not used
    continue;
  end

  switch net.params(p).trainMethod

    case 'average' % mainly for batch normalization
      thisLR = net.params(p).learningRate ;
      net.params(p).value = vl_taccum(...
          1 - thisLR, net.params(p).value, ...
          (thisLR/imNum/net.params(p).fanout),  parDer) ;

    case 'gradient'
      thisDecay = params.weightDecay * net.params(p).weightDecay ;
      thisLR = params.learningRate * net.params(p).learningRate ;

      if thisLR>0 || thisDecay>0
        % Normalize gradient and incorporate weight decay.
        parDer = vl_taccum(1/pairNum, parDer, ...
                           thisDecay, net.params(p).value) ;

        if isempty(params.solver)
          % Default solver is the optimised SGD.
          % Update momentum.
          state.solverState{p} = vl_taccum(...
            params.momentum, state.solverState{p}, ...
            -1, parDer) ;

          % Nesterov update (aka one step ahead).
          if params.nesterovUpdate
            delta = params.momentum * state.solverState{p} - parDer ;
          else
            delta = state.solverState{p} ;
          end

          % Update parameters.
          net.params(p).value = vl_taccum(...
            1,  net.params(p).value, thisLR, delta) ;

        else
          % call solver function to update weights
          [net.params(p).value, state.solverState{p}] = ...
            params.solver(net.params(p).value, state.solverState{p}, ...
            parDer, params.solverOpts, thisLR) ;
        end
      end
    otherwise
      error('Unknown training method ''%s'' for parameter ''%s''.', ...
        net.params(p).trainMethod, ...
        net.params(p).name) ;
  end
end

% -------------------------------------------------------------------------
function stats = accumulateStats(stats_)
% -------------------------------------------------------------------------

for s = {'train', 'val'}
  s = char(s) ;
  total = 0 ;

  % initialize stats stucture with same fields and same order as
  % stats_{1}
  stats__ = stats_{1} ;
  names = fieldnames(stats__.(s))' ;
  values = zeros(1, numel(names)) ;
  fields = cat(1, names, num2cell(values)) ;
  stats.(s) = struct(fields{:}) ;

  for g = 1:numel(stats_)
    stats__ = stats_{g} ;
    num__ = stats__.(s).num ;
    total = total + num__ ;

    for f = setdiff(fieldnames(stats__.(s))', 'num')
      f = char(f) ;
      stats.(s).(f) = stats.(s).(f) + stats__.(s).(f) * num__ ;

      if g == numel(stats_)
        stats.(s).(f) = stats.(s).(f) / total ;
      end
    end
  end
  stats.(s).num = total ;
end

% -------------------------------------------------------------------------
function stats = extractStats(stats, net)
% -------------------------------------------------------------------------
sel = find(cellfun(@(x) strcmp(x,'objective') || strcmp(x,'error'), {net.layers.name}));
for i = 1:numel(sel)
  % if net.layers(sel(i)).block.ignoreAverage, continue; end;
  stats.(net.layers(sel(i)).outputs{1}) = net.layers(sel(i)).block.average ;
end

% -------------------------------------------------------------------------
function saveState(fileName, net_, state)
% -------------------------------------------------------------------------
net = net_.saveobj() ;
save(fileName, 'net', 'state') ;

% -------------------------------------------------------------------------
function saveStats(fileName, stats)
% -------------------------------------------------------------------------
if exist(fileName)
  save(fileName, 'stats', '-append') ;
else
  save(fileName, 'stats') ;
end

% -------------------------------------------------------------------------
function [net, state, stats] = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'net', 'state', 'stats') ;
net = dagnn.DagNN.loadobj(net) ;
if isempty(whos('stats'))
  error('Epoch ''%s'' was only partially saved. Delete this file and try again.', ...
        fileName) ;
end

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;

% -------------------------------------------------------------------------
function switchFigure(n)
% -------------------------------------------------------------------------
if get(0,'CurrentFigure') ~= n
  try
    set(0,'CurrentFigure',n) ;
  catch
    figure(n) ;
  end
end

% -------------------------------------------------------------------------
function clearMex()
% -------------------------------------------------------------------------
clear vl_tmove vl_imreadjpeg ;

% -------------------------------------------------------------------------
function prepareGPUs(opts, cold)
% -------------------------------------------------------------------------
numGpus = numel(opts.gpus) ;
if numGpus > 1
  % check parallel pool integrity as it could have timed out
  pool = gcp('nocreate') ;
  if ~isempty(pool) && pool.NumWorkers ~= numGpus
    delete(pool) ;
  end
  pool = gcp('nocreate') ;
  if isempty(pool)
    parpool('local', numGpus) ;
    cold = true ;
  end

end
if numGpus >= 1 && cold
  clearMex() ;
  if numGpus == 1
    gpuinfo = gpuDevice(opts.gpus);
    fprintf('>> Running on GPU %s with Index %d\n', gpuinfo.Name, gpuinfo.Index);  
  else
    spmd
      clearMex() ;
      gpuinfo = gpuDevice(opts.gpus(labindex));
      fprintf('>> Running on GPU %s with Index %d\n', gpuinfo.Name, gpuinfo.Index);  
    end
  end
end
