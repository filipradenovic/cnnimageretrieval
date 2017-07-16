function opts = load_opts_train(varargin)
% LOAD_OPTS_TRAIN  List of all training options.
%
%   OPTS = load_opts_train(OPTS)
%
%   Default OPTS are overwritten with input OPTS.
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 
	
	%--------------------------------------------------------------------------------------------------
	% DEFAULT OPTS VALUES
	%--------------------------------------------------------------------------------------------------

	%% Network architecture and initialization options
	opts.init.model = 'ALEX'; % initialization model (ALEX | VGG | RESNET50 | RESNET101 | RESNET152)
	opts.init.modelDir = fullfile(get_root_cnnimageretrieval(), 'data', 'networks', 'imagenet'); % Directory with pretrained models
	opts.init.method = 'mac'; % method (mac | spoc)
	opts.init.objectiveType = {'contrastiveloss', 0.7}; % loss function (contrastiveloss | tripletloss)
	opts.init.errorType =  {'batchmap'}; %  error function (for validation)
	opts.init.averageImageScale = 1; % [0, 1] scaling for the average image
	opts.init.imageChannels = 3; % number of image channels, 3 - rgb images, 1 - grayscale
	opts.init.mergeBatchNorm = 1; % merge BatchNorm layers with the preceding Conv layers

	%% Training options (from matconvnet)
	opts.train.expDir = fullfile(get_root_cnnimageretrieval(), 'data', 'networks', 'exp');
	opts.train.dbPath = fullfile(get_root_cnnimageretrieval(), 'data', 'train', 'dbs', 'retrieval-SfM-30k.mat');
	opts.train.continue = true; % continue from last saved
	opts.train.batchSize = 5; % #queries per batch
	opts.train.numSubBatches = 1; % spliting the batch to numSubBatches
	opts.train.gpus = [1]; % which GPUID to use (can be more than one)
	opts.train.numEpochs = 30; % maximum number of epochs to train
	opts.train.learningRate = 0.001 .* exp(-(0:99)*0.1); % learning rate per epoch
	opts.train.weightDecay = 0.0005; % weight decay, has to be one for all epochs
	opts.train.solver = [];  % empty array means use the default SGD solver
	opts.train.derOutputs = {'objective', 1};
	opts.train.momentum = 0.9; % momentum, only one value
	opts.train.saveSolverState = false;
	opts.train.nesterovUpdate = false;
	opts.train.randomSeed = 0; % for reproducibility of results
	opts.train.profile = false;
	opts.train.parameterServer.method = 'tmove'; % comm. for multiple GPUS
	opts.train.parameterServer.prefix = 'mcn'; % comm. for multiple GPUS

	%% Additional training options
	opts.train.numNegative = 5; % number of hard negatives per query
	opts.train.numRemine = 3; % remine negatives numRemine times per epoch
	opts.train.memoryMapRemine = fullfile(tempdir, 'mmr'); % for remining with multiple GPUS

	%% Training data augmentation
	%% NOTE: If data is provided in db, only epochSize and jitterFlip options are valid/used
	opts.train.augment.epochSize = [inf, inf]; % [#queries, #ims] inf: full db
	opts.train.augment.imageDir = fullfile(get_root_cnnimageretrieval(), 'data', 'train', 'ims'); % directory for training images to be loaded, if no data in db
	opts.train.augment.imageSize = 362; % image size for training
	opts.train.augment.crop = false; % true: square crop, false: original size, padded with zeros
	opts.train.augment.gpus = []; % use gpus for image loading, empty: do not use gpus
	opts.train.augment.jitterFlip = false; % horizontal flip of rand positive pairs or random negatives
	opts.train.augment.jitterLocation = false; % true: random location when cropping and scaling, false: center location
	opts.train.augment.jitterScale = 1; % [0, 1]: scaling jittering, 1: no scaling, original scale
	opts.train.augment.jitterBrightness = 0;
	opts.train.augment.jitterContrast = 0;
	opts.train.augment.jitterSaturation = 0;


	%--------------------------------------------------------------------------------------------------
	% OVERWRITE WITH INPUT TO LOAD_OPTS
	%--------------------------------------------------------------------------------------------------
	opts = vl_argparse(opts, varargin); 


	%--------------------------------------------------------------------------------------------------
	% OVERWRITE WITH SAVED FILE BUT ALSO RESPECT INPUT TO LOAD_OPTS
	%--------------------------------------------------------------------------------------------------
	optsFile = fullfile(opts.train.expDir, 'opts.mat');
	%% Load opts file
	if exist(optsFile, 'file')
		opts_ = load(optsFile);
		opts_ = opts_.opts;
		% load the saved file but also keep the input (load_opts) values 		
		opts = vl_argparse(opts_, opts); 
	end

	%--------------------------------------------------------------------------------------------------
	% CREATE EXPORT DIRECTORY (expDir) AND SAVE OPTS IN IT 
	%--------------------------------------------------------------------------------------------------
	%% Create exoDir to save fine-tuned networks if it doesnt exist
	if ~exist(opts.train.expDir)
		mkdir(opts.train.expDir);
	end
	%% Save opts file
	save(optsFile, 'opts');

	%--------------------------------------------------------------------------------------------------
	% SET DEPENDENT VALUES
	%--------------------------------------------------------------------------------------------------
	
	%% Find initialization modelPath for a given model, 
	%% and name of the last layer to be used (everything after it is removed)
	if strcmp(opts.init.model,'ALEX')
		opts.init.modelPath = fullfile(opts.init.modelDir, 'imagenet-caffe-alex.mat'); 
		opts.init.lastLayer = 'relu5';
		opts.init.outputDim = 256;
		opts.init.isDagnn   = 0;
	elseif strcmp(opts.init.model, 'VGG')
		opts.init.modelPath = fullfile(opts.init.modelDir, 'imagenet-vgg-verydeep-16.mat');
		opts.init.lastLayer = 'relu5_3';
		opts.init.outputDim = 512;
		opts.init.isDagnn   = 0;
	elseif strcmp(opts.init.model, 'GOOGLENET')
		opts.init.modelPath = fullfile(opts.init.modelDir, 'imagenet-googlenet-dag.mat');
		opts.init.lastLayer = 'icp9_out';
		opts.init.outputDim = 1024;
		opts.init.isDagnn   = 1;
	elseif strcmp(opts.init.model, 'RESNET50')
		opts.init.modelPath = fullfile(opts.init.modelDir, 'imagenet-resnet-50-dag.mat'); 
		opts.init.lastLayer = 'res5c_relu';
		opts.init.outputDim = 2048;
		opts.init.isDagnn   = 1;
	elseif strcmp(opts.init.model, 'RESNET101')
	    opts.init.modelPath = fullfile(opts.init.modelDir, 'imagenet-resnet-101-dag.mat'); 
	    opts.init.lastLayer = 'res5c_relu';
	    opts.init.outputDim = 2048;
	    opts.init.isDagnn   = 1;
	elseif strcmp(opts.init.model, 'RESNET152')
	    opts.init.modelPath = fullfile(opts.init.modelDir, 'imagenet-resnet-152-dag.mat'); 
	    opts.init.lastLayer = 'res5c_relu';
	    opts.init.outputDim = 2048;
	    opts.init.isDagnn   = 1;
	else
		disp('Unknown model!!! KEYBOARD invoked...');
		keyboard
	end

	%% Training solver initialization
	if ~isempty(opts.train.solver)
  		assert(isa(opts.train.solver, 'function_handle') && nargout(opts.train.solver) == 2,...
    					'Invalid solver; expected a function handle with two outputs.');
  		% Call without input arguments, to get default options
  		opts.train.solverOpts = opts.train.solver();
	end

	%% Prefix for memory map files
	if(strcmp(opts.train.expDir(end),'/'))
		[~, foldername] = fileparts(opts.train.expDir(1:end-1));
	else
		[~, foldername] = fileparts(opts.train.expDir);
	end
	opts.train.parameterServer.prefix = [foldername '_matconvnet_cnnimageretrieval'];
	opts.train.memoryMapRemine = fullfile(tempdir, [foldername '_mmapremine_cnnimageretrieval.bin']);

	%% GPU for image loading same as for training
	opts.train.augment.gpus = opts.train.gpus;