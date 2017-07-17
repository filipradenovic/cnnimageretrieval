function download_train(data_dir)
% DOWNLOAD_TRAIN Checks, and, if required, downloads the necessary data and networks for the training.
%
%   download_train(DATA_ROOT) checks if the data and networks necessary for running the training script exist.
%   If not it downloads it in the folder structure:
%     DATA_ROOT/train/dbs/         : folder with training database mat files
%     DATA_ROOT/train/ims/         : folder with original images used for training
%     DATA_ROOT/networks/imagenet/ : CNN models pretrained for classification using imagenet data
    
    % Create data folder if it does not exist
    if ~exist(data_dir, 'dir')
        mkdir(data_dir);
    end

    % Create train folder if it does not exist
    train_dir = fullfile(data_dir, 'train');
    if ~exist(train_dir, 'dir')
        mkdir(train_dir);
    end
    
    % Download folder train/db/
    src_dir = fullfile('http://cmp.felk.cvut.cz/cnnimageretrieval/data', 'train', 'dbs');
    dst_dir = fullfile(data_dir, 'train', 'dbs');
    dl_files = {'retrieval-SfM-30k.mat',  'retrieval-SfM-30k-whiten.mat', ...
                'retrieval-SfM-120k.mat', 'retrieval-SfM-120k-whiten.mat'};
    if ~exist(dst_dir, 'dir')
        fprintf('>> Database directory does not exist. Creating: %s\n', dst_dir);
        mkdir(dst_dir);
        fprintf('>> Downloading database files from cmp.felk.cvut.cz/cnnimageretrieval\n');
    end
    for i = 1:numel(dl_files)
        src_file = fullfile(src_dir, dl_files{i});
        dst_file = fullfile(dst_dir, dl_files{i});
        if ~exist(dst_file, 'file')
            fprintf('>> DB file %s does not exist. Downloading...\n', dl_files{i});
            system(sprintf('wget %s -O %s', src_file, dst_file)); 
        end
    end

    % Download folder train/ims/
    src_dir = fullfile('http://cmp.felk.cvut.cz/cnnimageretrieval/data', 'train', 'ims');
    dst_dir = fullfile(data_dir, 'train', 'ims');
    dl_file = 'ims.tar.gz';
    if ~exist(dst_dir, 'dir')
        src_file = fullfile(src_dir, dl_file);
        dst_file = fullfile(dst_dir, dl_file);
        fprintf('>> Image directory does not exist. Creating: %s\n', dst_dir);
        mkdir(dst_dir);
        fprintf('>> Downloading ims.tar.gz...\n');
        system(sprintf('wget %s -O %s', src_file, dst_file));
        fprintf('>> Extracting %s...\n', dst_file);
        system(sprintf('tar -zxf %s -C %s', dst_file, dst_dir));
        fprintf('>> Extracted, deleting %s...\n', dst_file);
        system(sprintf('rm %s', dst_file));
    end
    dl_files = {'retrieval-SfM-30k-imagenames-clusterids.mat', 'retrieval-SfM-120k-imagenames-clusterids.mat'};
    for i = 1:numel(dl_files)
        src_file = fullfile(src_dir, dl_files{i});
        dst_file = fullfile(dst_dir, dl_files{i});
        if ~exist(dst_file, 'file')
            fprintf('>> Dataset image split %s does not exist. Downloading...\n', dl_files{i});
            system(sprintf('wget %s -O %s', src_file, dst_file)); 
        end
    end

    % Download folder networks/imagenet/
    src_dir = fullfile('http://www.vlfeat.org/matconvnet/', 'models');
    dst_dir = fullfile(data_dir, 'networks', 'imagenet');
    dl_files = {'imagenet-caffe-alex.mat', 'imagenet-vgg-verydeep-16.mat', 'imagenet-googlenet-dag.mat', ...
                'imagenet-resnet-50-dag.mat', 'imagenet-resnet-101-dag.mat', 'imagenet-resnet-152-dag.mat'};
    if ~exist(dst_dir, 'dir')
        fprintf('>> Imagenet networks directory does not exist. Creating: %s\n', dst_dir);
        mkdir(dst_dir);
        fprintf('>> Downloading imagenet networks from http://www.vlfeat.org/matconvnet\n');
    end
    for i = 1:numel(dl_files)
        src_file = fullfile(src_dir, dl_files{i});
        dst_file = fullfile(dst_dir, dl_files{i});
        if ~exist(dst_file, 'file')
            fprintf('>> Network %s does not exist. Downloading...\n', dl_files{i});
            system(sprintf('wget %s -O %s', src_file, dst_file)); 
        end
    end