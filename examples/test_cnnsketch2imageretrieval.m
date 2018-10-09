% TEST_CNNSKETCH2IMAGERETRIEVAL  Code to evaluate (not train) the methods presented in the paper:
% F. Radenovic, G. Tolias, O. Chum, Deep Shape Matching, ECCV 2018
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2018. 

clear;

%---------------------------------------------------------------------
% Set data folder and testing parameters
%---------------------------------------------------------------------

% Set data folder, change if you have downloaded the data somewhere else
data_root = fullfile(get_root_cnnimageretrieval(), 'data');
% Check, and, if necessary, download test data (Flickr15 sketch dataset), and fine-tuned networks
download_test_sketch(data_root); 

% Set test options
test_dataset = 'flickr15k_sketch';  % dataset to evaluate on
test_imdim = 227;  % choose test image dimensionality
use_mirror = 1; % use mirror representation, otherwise use only original image
use_ms = 1; % use multi-scale representation, otherwise use single-scale
use_gpu = [1]; % use GPUs (array of GPUIDs), if empty use CPU

% Choose ECCV18 fine-tuned CNN network
network_file = fullfile(data_root, 'networks', 'retrieval-SfM-30k', 'retrievalSfM30k-edgemac-vgg.mat');

% After running the training script train_cnnsketch2imageretrieval.m you can evaluate fine-tuned network
% network_file = fullfile(data_root, 'networks', 'exp', 'vgg_edgefilter_mac_test', 'net-epoch-20');

%---------------------------------------------------------------------
% Set dependent variables
%---------------------------------------------------------------------

% Prepare function for desc extraction
if ~use_ms
    descfun = @(x, y, z) cnn_vecms_sketch (x, y, z, 1, use_mirror);
else
    descfun = @(x, y, z) cnn_vecms_sketch (x, y, z, [1, 1/sqrt(2), sqrt(2), 1/2, 2], use_mirror);
end

%---------------------------------------------------------------------
% Testing
%---------------------------------------------------------------------
[~, network_name, ~] = fileparts(network_file);
fprintf('>> %s: Evaluating CNN image retrieval...\n', network_name);

% Load pre-trained edge detector
fprintf('>> Loading Dollar edge detector toolbox and model...\n');
emodel = load_edgedetector(data_root);

% Load pre-trained CNN network
fprintf('>> Loading CNN model...\n');
load(network_file);
net = dagnn.DagNN.loadobj(net);

% prepare GPUs if necessary
numGpus = numel(use_gpu);
if numGpus, fprintf('>> Preparing GPU(s)...\n'); end
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

% extract and evaluate
fprintf('>> %s: Processing test dataset...\n', test_dataset);       
cfg = configdataset (test_dataset, fullfile(data_root, 'test/')); % config file for the dataset

fprintf('>> %s: Extracting CNN descriptors for db images...\n', test_dataset); 
vecs = cell(1, cfg.n);
if numGpus <= 1
    progressbar(0);
    for i = 1:cfg.n
        vecs{i} = descfun(imresizemaxd(imread(cfg.im_fname(cfg, i)), test_imdim), net, emodel);
        progressbar(i/cfg.n);
    end
else
    time = tic;
    parfor i = 1:cfg.n
        if strcmp(net.device, 'cpu'), net.move('gpu'); end
        vecs{i} = descfun(imresizemaxd(imread(cfg.im_fname(cfg, i)), test_imdim), net, emodel);
    end
    fprintf('>>>> done in %s\n', htime(toc(time)));
end
vecs = cell2mat(vecs);

fprintf('>> %s: Extracting CNN descriptors for query images...\n', test_dataset); 
qvecs = cell(1, cfg.nq);
if numGpus <= 1
    progressbar(0);
    for i = 1:cfg.nq
        qvecs{i} = descfun(imresizemaxd(imread(cfg.qim_fname(cfg, i)), test_imdim), net, 0);
        progressbar(i/cfg.nq);
    end
else
    time = tic;
    parfor i = 1:cfg.nq
        if strcmp(net.device, 'cpu'), net.move('gpu'); end
        qvecs{i} = descfun(imresizemaxd(imread(cfg.qim_fname(cfg, i)), test_imdim), net, 0);
    end
    fprintf('>>>> done in %s\n', htime(toc(time)));
end
qvecs = cell2mat(qvecs);


fprintf('>> %s: Retrieval...\n', test_dataset);
sim = vecs'*qvecs;
[sim, ranks] = sort(sim, 'descend');
map = compute_map (ranks, cfg.gnd);   
fprintf('>> %s: mAP = %.4f\n', test_dataset, map);
