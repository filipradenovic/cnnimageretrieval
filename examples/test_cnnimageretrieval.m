% TEST_CNNIMAGERETRIEVAL  Code to evaluate (not train) the methods presented in the paper:
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
use_rvec = 0;  % use R-MAC, otherwise use MAC
use_gpu = 1;  % use GPU (GPUID = use_gpu), otherwise use CPU

% Choose ECCV16 fine-tuned CNN network (siamac-alex or siamac-vgg)
network_file = fullfile(data_root, 'networks', 'retrieval-SfM-30k', 'retrievalSfM30k-siamac-alex.mat');
% network_file = fullfile(data_root, 'networks', 'retrieval-SfM-30k', 'retrievalSfM30k-siamac-vgg.mat');

% % After running the training script train_cnnimageretrieval.m you can evaluate fine-tuned network
% network_file = fullfile(data_root, 'networks', 'exp', 'vgg_mac_test', 'net-epoch-30');

%---------------------------------------------------------------------
% Set dependent variables
%---------------------------------------------------------------------

% Choose training data for whitening and set up data folder
train_whiten_file = fullfile(data_root, 'train', 'dbs', 'retrieval-SfM-30k-whiten.mat'); % less images, faster
% train_whiten_file = fullfile(data_root, 'train', 'dbs', 'retrieval-SfM-120k-whiten.mat'); % more images, a bit better results but slower

% Set folder where original training images are stored, whitening learned on them
ims_whiten_dir = fullfile(data_root, 'train', 'ims');

% Prepare function for desc extraction
if ~use_rvec, descfun = @(x, y) cnn_vec (x, y);  else, descfun = @(x, y) cnn_vecr(x, y); end


%---------------------------------------------------------------------
% Testing
%---------------------------------------------------------------------

[~, network_name, ~] = fileparts(network_file);
fprintf('>> %s: Evaluating CNN image retrieval...\n', network_name);

% Load pre-trained CNN network
load(network_file);
net = dagnn.DagNN.loadobj(net);
if use_gpu,	gpuDevice(use_gpu); net.move('gpu'); end

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
progressbar(0);
for i=1:numel(cids)
	vecs_whiten{i} = descfun(imresizemaxd(imread(cid2filename(cids{i}, ims_whiten_dir)), test_imdim, 0), net);
	progressbar(i/numel(cids));
end
vecs_whiten = cell2mat(vecs_whiten);
fprintf('>> whitening: Learning...\n');
Lw = whitenlearn(vecs_whiten, qidxs, pidxs);

% extract and evaluate
for d = 1:numel(test_datasets)
	fprintf('>> %s: Processing test dataset...\n', test_datasets{d});		
	cfg = configdataset (test_datasets{d}, fullfile(data_root, 'test/')); % config file for the dataset

	fprintf('>> %s: Extracting CNN descriptors for db images...\n', test_datasets{d}); 
	progressbar(0); vecs = [];
	for i = 1:cfg.n
		vecs{i} = descfun(imresizemaxd(imread(cfg.im_fname(cfg, i)), test_imdim, 0), net);
		progressbar(i/cfg.n);
	end
	vecs = cell2mat(vecs);

	fprintf('>> %s: Extracting CNN descriptors for query images...\n', test_datasets{d}); 
	progressbar(0); qvecs = [];
	for i = 1:cfg.nq
		qvecs{i} = descfun(crop_qim(imread(cfg.qim_fname(cfg, i)), cfg.gnd(i).bbx, test_imdim), net);
		progressbar(i/cfg.nq);
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
