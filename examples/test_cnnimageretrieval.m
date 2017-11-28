% TEST_CNNIMAGERETRIEVAL  Code to evaluate (not train) the methods presented in the papers:
% F. Radenovic, G. Tolias, O. Chum, Fine-tuning CNN Image Retrieval with No Human Annotation, arXiv 2017
% F. Radenovic, G. Tolias, O. Chum, CNN Image Retrieval Learns from BoW: Unsupervised Fine-Tuning with Hard Examples, ECCV 2016
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 

clear;

%---------------------------------------------------------------------
% Set data folder and testing parameters
%---------------------------------------------------------------------

% Set data folder, change if you have downloaded the data somewhere else
data_root = fullfile(get_root_cnnimageretrieval(), 'data');
% Check, and, if necessary, download train data (for whiten)
download_train(data_root);
% Check, and, if necessary, download test data (Oxf5k and Par6k), and fine-tuned networks
download_test(data_root); 

% Set test options
test_datasets = {'oxford5k', 'paris6k'};  % list of datasets to evaluate on
test_imdim = 1024;  % choose test image dimensionality
use_ms = 1; % use multi-scale representation, otherwise use single-scale
use_rvec = 0;  % use regional representation (R-MAC, R-GeM), otherwise use global (MAC, GeM)
use_gpu = [1];  % use GPUs (array of GPUIDs), if empty use CPU

% Choose ECCV16 fine-tuned CNN network
% network_file = fullfile(data_root, 'networks', 'retrieval-SfM-30k', 'retrievalSfM30k-siamac-alex.mat');
% network_file = fullfile(data_root, 'networks', 'retrieval-SfM-30k', 'retrievalSfM30k-siamac-vgg.mat');

% Choose arXiv17 fine-tuned CNN network
% network_file = fullfile(data_root, 'networks', 'retrieval-SfM-30k', 'retrievalSfM30k-gem-alex.mat');
% network_file = fullfile(data_root, 'networks', 'retrieval-SfM-120k', 'retrievalSfM120k-gem-vgg.mat');
network_file = fullfile(data_root, 'networks', 'retrieval-SfM-120k', 'retrievalSfM120k-gem-resnet101.mat');

% After running the training script train_cnnimageretrieval.m you can evaluate fine-tuned network
% network_file = fullfile(data_root, 'networks', 'exp', 'resnet101_gem_test', 'net-epoch-30');

%---------------------------------------------------------------------
% Set dependent variables
%---------------------------------------------------------------------

% Choose training data for whitening and set up data folder
% train_whiten_file = fullfile(data_root, 'train', 'dbs', 'retrieval-SfM-30k-whiten.mat'); % less images, faster
train_whiten_file = fullfile(data_root, 'train', 'dbs', 'retrieval-SfM-120k-whiten.mat'); % more images, better results but slower

% Set folder where original training images are stored, whitening learned on them
ims_whiten_dir = fullfile(data_root, 'train', 'ims');

% Prepare function for desc extraction
if ~use_rvec 
	if ~use_ms
		descfun = @(x, y) cnn_vecms (x, y, 1);
	else
		descfun = @(x, y) cnn_vecms (x, y, [1, 1/sqrt(2), 1/2]);
	end  
else 
	if ~use_ms
		descfun = @(x, y) cnn_vecrms (x, y, 3, 1);
	else
		descfun = @(x, y) cnn_vecrms (x, y, 3, [1, 1/sqrt(2), 1/2]);
	end  
end

%---------------------------------------------------------------------
% Testing
%---------------------------------------------------------------------
[~, network_name, ~] = fileparts(network_file);
fprintf('>> %s: Evaluating CNN image retrieval...\n', network_name);

% Load pre-trained CNN network
load(network_file);
net = dagnn.DagNN.loadobj(net);

% prepare GPUs if necessary
numGpus = numel(use_gpu);
if numGpus, fprintf('>> Prepring GPU(s)...\n'); end
if numGpus > 1
	% check parallel pool integrity as it could have timed out
	pool = gcp('nocreate');
	if ~isempty(pool) && pool.NumWorkers ~= numGpus
		delete(pool);
	end
	pool = gcp('nocreate');
	if isempty(pool)
		parpool('local', numGpus);
	end
end
if numGpus >= 1
	if numGpus == 1
		gpuinfo = gpuDevice(use_gpu);
		net.move('gpu');
		fprintf('>>>> Running on GPU %s with Index %d\n', gpuinfo.Name, gpuinfo.Index);  
	else
		spmd
			gpuinfo = gpuDevice(use_gpu(labindex));
			fprintf('>>>> Running on GPU %s with Index %d\n', gpuinfo.Name, gpuinfo.Index);  
		end
	end
end

% Load training data filenames and pairs for whitening
train_whiten = load(train_whiten_file);
if isfield(train_whiten, 'train') && isfield(train_whiten, 'val')
	cids  = [train_whiten.train.cids train_whiten.val.cids]; 
	qidxs = [train_whiten.train.qidxs train_whiten.val.qidxs+numel(train_whiten.train.cids)]; % query indexes 
	pidxs = [train_whiten.train.pidxs train_whiten.val.pidxs+numel(train_whiten.train.cids)]; % positive indexes
else
	cids  = train_whiten.cids; 
	qidxs = train_whiten.qidxs; % query indexes 
	pidxs = train_whiten.pidxs; % positive indexes
end

% learn whitening
fprintf('>> whitening: Extracting CNN descriptors for training images...\n');
vecs_whiten = cell(1, numel(cids));
if numGpus <= 1
	progressbar(0);
	for i=1:numel(cids)
		vecs_whiten{i} = descfun(imresizemaxd(imread(cid2filename(cids{i}, ims_whiten_dir)), test_imdim, 0), net);
		progressbar(i/numel(cids));
	end
else
	time = tic;
	parfor i=1:numel(cids)
		if strcmp(net.device, 'cpu'), net.move('gpu'); end
		vecs_whiten{i} = descfun(imresizemaxd(imread(cid2filename(cids{i}, ims_whiten_dir)), test_imdim, 0), net);
	end
	fprintf('>>>> done in %s\n', htime(toc(time)));
end
vecs_whiten = cell2mat(vecs_whiten);
fprintf('>> whitening: Learning...\n');
Lw = whitenlearn(vecs_whiten, qidxs, pidxs);

% extract and evaluate
for d = 1:numel(test_datasets)
	fprintf('>> %s: Processing test dataset...\n', test_datasets{d});		
	cfg = configdataset (test_datasets{d}, fullfile(data_root, 'test/')); % config file for the dataset

	fprintf('>> %s: Extracting CNN descriptors for db images...\n', test_datasets{d}); 
	vecs = cell(1, cfg.n);
	if numGpus <= 1
		progressbar(0);
		for i = 1:cfg.n
			vecs{i} = descfun(imresizemaxd(imread(cfg.im_fname(cfg, i)), test_imdim, 0), net);
			progressbar(i/cfg.n);
		end
	else
		time = tic;
		parfor i = 1:cfg.n
			if strcmp(net.device, 'cpu'), net.move('gpu'); end
			vecs{i} = descfun(imresizemaxd(imread(cfg.im_fname(cfg, i)), test_imdim, 0), net);
		end
		fprintf('>>>> done in %s\n', htime(toc(time)));
	end
	vecs = cell2mat(vecs);

	fprintf('>> %s: Extracting CNN descriptors for query images...\n', test_datasets{d}); 
	qvecs = cell(1, cfg.nq);
	if numGpus <= 1
		progressbar(0);
		for i = 1:cfg.nq
			qvecs{i} = descfun(crop_qim(imread(cfg.qim_fname(cfg, i)), cfg.gnd(i).bbx, test_imdim), net);
			progressbar(i/cfg.nq);
		end
	else
		time = tic;
		parfor i = 1:cfg.nq
			if strcmp(net.device, 'cpu'), net.move('gpu'); end
			qvecs{i} = descfun(crop_qim(imread(cfg.qim_fname(cfg, i)), cfg.gnd(i).bbx, test_imdim), net);
		end
		fprintf('>>>> done in %s\n', htime(toc(time)));
	end
	qvecs = cell2mat(qvecs);

	vecsLw = whitenapply(vecs, Lw.m, Lw.P); % apply whitening on database descriptors
	qvecsLw = whitenapply(qvecs, Lw.m, Lw.P); % apply whitening on query descriptors

	fprintf('>> %s: Retrieval...\n', test_datasets{d});
	% raw descriptors
	sim = vecs'*qvecs;
	[sim, ranks] = sort(sim, 'descend');
	map = compute_map (ranks, cfg.gnd);	
	fprintf('>> %s: mAP = %.4f, without whiten\n', test_datasets{d}, map);
	% with learned whitening
	sim = vecsLw'*qvecsLw;
	[sim, ranks] = sort(sim, 'descend');
	map = compute_map (ranks, cfg.gnd);	
	fprintf('>> %s: mAP = %.4f, with whiten\n', test_datasets{d}, map);
end
