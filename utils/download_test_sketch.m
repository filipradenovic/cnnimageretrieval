function download_test_sketch(data_dir)
% DOWNLOAD_TEST_SKETCH Checks, and, if required, downloads the necessary data and networks for the testing.
%
%   download_test_sketch(DATA_ROOT) checks if the data and networks necessary for running the sketch-based image retrieval testing script exist.
%   If not it downloads it in the folder structure:
%     DATA_ROOT/test/flickr15k_sketch/       : folder with Flickr15k images and ground truth file
%     DATA_ROOT/networks/retrieval-SfM-30k/  : CNN models fine-tuned for sketch-based image retrieval using retrieval-SfM-30k edge-map data

    % Create data folder if it does not exist
    if ~exist(data_dir, 'dir')
        mkdir(data_dir);
    end

    % Create datasets folder if it does not exist
    datasets_dir = fullfile(data_dir, 'test');
    if ~exist(datasets_dir, 'dir')
        mkdir(datasets_dir);
    end

    % Download datasets folders test/DATASETNAME/
    dataset = 'flickr15k_sketch';
    src_dir = fullfile('http://www.cvssp.org/data/Flickr25K');
    dl_file = 'Flickr15K.zip';
    dst_dir = fullfile(datasets_dir, dataset, 'jpg');
    dst_dir_tmp = fullfile(datasets_dir, dataset, 'Flickr15K');
    dst_dir_parent = fullfile(datasets_dir, dataset);
    if ~exist(dst_dir, 'dir')
        fprintf('>> Dataset %s directory does not exist. Creating: %s\n', dataset, dst_dir);
        mkdir(dst_dir_parent);
        src_file = fullfile(src_dir, dl_file);
        dst_file = fullfile(dst_dir_parent, dl_file);
        fprintf('>> Downloading dataset %s archive %s...\n', dataset, dl_file);
        system(sprintf('wget %s -O %s', src_file, dst_file));
        fprintf('>> Extracting dataset %s archive %s...\n', dataset, dl_file);
        % extract in tmp folder
        system(sprintf('unzip -q %s -d %s', dst_file, dst_dir_parent));
        % move tmp folder to be jpg
        system(sprintf('mv %s %s', dst_dir_tmp, dst_dir));
        % delete zip file
        fprintf('>> Extracted, deleting dataset %s archive %s...\n', dataset, dl_file);
        system(sprintf('rm %s', dst_file));
    end
    gnd_src_dir = fullfile('http://cmp.felk.cvut.cz/cnnimageretrieval/data', 'test', dataset);
    gnd_dst_dir = fullfile(datasets_dir, dataset);
    gnd_dl_file = sprintf('gnd_%s.mat', dataset);
    gnd_src_file = fullfile(gnd_src_dir, gnd_dl_file);
    gnd_dst_file = fullfile(gnd_dst_dir, gnd_dl_file);
    if ~exist(gnd_dst_file, 'file')
        fprintf('>> Downloading dataset %s ground truth file...\n', dataset);
        system(sprintf('wget %s -O %s', gnd_src_file, gnd_dst_file));
    end

    % Download folder networks/retrieval-SfM-30k/
    src_dir = fullfile('http://cmp.felk.cvut.cz/cnnimageretrieval/data', 'networks', 'retrieval-SfM-30k');
    dst_dir = fullfile(data_dir, 'networks', 'retrieval-SfM-30k');
    dl_files = {'retrievalSfM30k-edgemac-vgg.mat'};
    if ~exist(dst_dir, 'dir')
        fprintf('>> Fine-tuned networks directory does not exist. Creating: %s\n', dst_dir);
        mkdir(dst_dir);
        fprintf('>> Downloading fine-tuned network files from http://cmp.felk.cvut.cz/cnnimageretrieval\n');
    end
    for i = 1:numel(dl_files)
        src_file = fullfile(src_dir, dl_files{i});
        dst_file = fullfile(dst_dir, dl_files{i});
        if ~exist(dst_file, 'file')
            fprintf('>> Network %s does not exist. Downloading...\n', dl_files{i});
            system(sprintf('wget %s -O %s', src_file, dst_file)); 
        end
    end