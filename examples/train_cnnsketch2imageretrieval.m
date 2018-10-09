% TRAIN_CNNSKETCH2IMAGERETRIEVAL Code to train the methods presented in the paper:
% F. Radenovic, G. Tolias, O. Chum, Deep Shape Matching, ECCV 2018
%
% Note: The method has been re-coded since our ECCV 2018 paper and minor differences in performance might appear.
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2018. 

clear;

%-------------------------------------------------------------------------------
% Set data folder
%-------------------------------------------------------------------------------

% Set data folder, change if you have downloaded the data somewhere else
data_root = fullfile(get_root_cnnimageretrieval(), 'data');
% Check, and, if necessary, download train data (db with edgemaps), and pre-trained imagenet networks
download_train_sketch(data_root);


%-------------------------------------------------------------------------------
% Reproduce training from ECCV18 paper: Deep Shape Matching ...
%-------------------------------------------------------------------------------

% Set architecture and initialization parameters
opts.init.model = 'VGG'; % (ALEX | VGG | GOOGLENET | RESNET101)
opts.init.modelDir = fullfile(data_root, 'networks', 'imagenet');
opts.init.method = 'edgefilter_mac';
opts.init.objectiveType = {'contrastiveloss', 0.7};
opts.init.errorType = {'batchmap'};
opts.init.averageImageScale = 0;
opts.init.imageChannels = 1;

% Set train parameters
opts.train.dbPath = fullfile(data_root, 'train', 'dbs', 'retrieval-SfM-30k-edgemap.mat');
opts.train.batchSize = 20;
opts.train.numSubBatches = 4;
opts.train.numEpochs = 20;
opts.train.learningRate = 0.001 .* exp(-(0:99)*0.1);
opts.train.numNegative = 5;
opts.train.numRemine = 3;
opts.train.gpus = [1];

opts.train.augment.jitterFlip = true;
opts.train.augment.jitterQueryBinarize = true;

% Trial name (to name a save directory)
trialName = 'test';

% Export directory expDir named after model, method and trialName
opts.init.method = [opts.init.method, '_', trialName];
opts.train.expDir = fullfile(data_root, 'networks', 'exp', [lower(opts.init.model) '_' lower(opts.init.method)]);
if ~exist(opts.train.expDir); mkdir(opts.train.expDir); end % create folder if its not there

% Load opts by respecting added opts
opts = load_opts_train(opts);

% Initialize and train the network
fprintf('>> Experiment folder is set to %s\n', opts.train.expDir);
net = init_network(opts.init);
[net, state, stats] = train_network(net, @(o,i,n,b,s,m,e) get_batch(o,i,n,b,s,m,e), opts.train);