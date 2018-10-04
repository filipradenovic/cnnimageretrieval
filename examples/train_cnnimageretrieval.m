% TRAIN_CNNIMAGERETRIEVAL Code to train the methods presented in the papers:
% F. Radenovic, G. Tolias, O. Chum, Fine-tuning CNN Image Retrieval with No Human Annotation, TPAMI 2018
% F. Radenovic, G. Tolias, O. Chum, CNN Image Retrieval Learns from BoW: Unsupervised Fine-Tuning with Hard Examples, ECCV 2016
%
% Note: The method has been re-coded since our ECCV 2016 paper and minor differences in performance might appear.
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 

clear;

%-------------------------------------------------------------------------------
% Set data folder
%-------------------------------------------------------------------------------

% Set data folder, change if you have downloaded the data somewhere else
data_root = fullfile(get_root_cnnimageretrieval(), 'data');
% Check, and, if necessary, download train data (dbs, and original images), 
% and pre-trained imagenet networks
download_train(data_root);


%-------------------------------------------------------------------------------
% Reproduce training from TPAMI18 paper: Fine-tuning CNN Image Retrieval ...
%-------------------------------------------------------------------------------

%% RESNET101 -------------------------------------------------------------------
clear opts

% Set architecture and initialization parameters
opts.init.model = 'RESNET101'; % (ALEX | VGG | GOOGLENET | RESNET101)
opts.init.modelDir = fullfile(data_root, 'networks', 'imagenet');
opts.init.method = 'gem'; % (gem: one p for all filters | gemmp: one p per filter)
opts.init.objectiveType = {'contrastiveloss', 0.85}; % (0.7 ALEX | 0.75 VGG | 0.85 RESNET101)
opts.init.errorType = {'batchmap'};

% Set train parameters
% We provide 2 pools of training images comprising 30k and 120k images
% The latter is used in our TPAMI18 paper
opts.train.dbPath = fullfile(data_root, 'train', 'dbs', 'retrieval-SfM-120k.mat');
opts.train.batchSize = 5;
opts.train.numEpochs = 30;
opts.train.learningRate = 10^-6 * exp(-0.1 * [0:99]);
opts.train.solver = @solver.adam;
opts.train.numNegative = 5;
opts.train.numRemine = 3;
opts.train.gpus = [1];

opts.train.augment.epochSize = [6000, 22000];
opts.train.augment.imageDir = fullfile(data_root, 'train', 'ims');
opts.train.augment.imageSize = 362; 

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

%% VGG -------------------------------------------------------------------------
clear opts

% Set architecture and initialization parameters
opts.init.model = 'VGG'; % (ALEX | VGG | GOOGLENET | RESNET101)
opts.init.modelDir = fullfile(data_root, 'networks', 'imagenet');
opts.init.method = 'gem'; % (gem: one p for all filters | gemmp: one p per filter)
opts.init.objectiveType = {'contrastiveloss', 0.75}; % (0.7 ALEX | 0.75 VGG | 0.85 RESNET101)
opts.init.errorType = {'batchmap'};

% Set train parameters
% We provide 2 pools of training images comprising 30k and 120k images
% The latter is used in our TPAMI18 paper
opts.train.dbPath = fullfile(data_root, 'train', 'dbs', 'retrieval-SfM-120k.mat');
opts.train.batchSize = 5;
opts.train.numEpochs = 30;
opts.train.learningRate = 10^-6 * exp(-0.1 * [0:99]);
opts.train.solver = @solver.adam;
opts.train.numNegative = 5;
opts.train.numRemine = 3;
opts.train.gpus = [1];

opts.train.augment.epochSize = [6000, 22000];
opts.train.augment.imageDir = fullfile(data_root, 'train', 'ims');
opts.train.augment.imageSize = 362; 

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


%-------------------------------------------------------------------------------
% Reproduce training from ECCV16 paper: CNN Image Retrieval Learns from BoW: ...
%-------------------------------------------------------------------------------

%% VGG -------------------------------------------------------------------------
clear opts

% Set architecture and initialization parameters
opts.init.model = 'VGG'; % (ALEX | VGG | RESNET101)
opts.init.modelDir = fullfile(data_root, 'networks', 'imagenet');
opts.init.method = 'mac';
opts.init.objectiveType = {'contrastiveloss', 0.7};
opts.init.errorType = {'batchmap'};

% Set train parameters
% We provide 2 pools of training images comprising 30k and 120k images
% The former is used in our ECCV16 paper
opts.train.dbPath = fullfile(data_root, 'train', 'dbs', 'retrieval-SfM-30k.mat');
opts.train.batchSize = 5;
opts.train.numEpochs = 30;
opts.train.learningRate = [0.001*ones(1,10), 0.001*(1/5)*ones(1,10), 0.001*(1/25)*ones(1,10)];
opts.train.numNegative = 5;
opts.train.numRemine = 3;
opts.train.gpus = [1];

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