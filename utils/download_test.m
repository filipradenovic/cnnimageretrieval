function download_test(data_dir)
% DOWNLOAD_TEST Checks, and, if required, downloads the necessary data and networks for the testing.
%
%   download_test(DATA_ROOT) checks if the data and networks necessary for running the testing script exist.
%   If not it downloads it in the folder structure:
%     DATA_ROOT/test/oxford5k/               : folder with Oxford5k images and ground truth file
%     DATA_ROOT/test/paris6k/                : folder with Paris6k images and ground truth file
%     DATA_ROOT/test/roxford5k/              : folder with Oxford5k images and revisited ground truth file
%     DATA_ROOT/test/rparis6k/               : folder with Paris6k images and revisited ground truth file
%     DATA_ROOT/networks/retrieval-SfM-30k/  : CNN models fine-tuned for image retrieval using retrieval-SfM-30k data
%     DATA_ROOT/networks/retrieval-SfM-120k/ : CNN models fine-tuned for image retrieval using retrieval-SfM-120k data

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
    datasets = {'oxford5k', 'paris6k', 'roxford5k', 'rparis6k'};
    for di = 1:numel(datasets)
        dataset = datasets{di};
        switch dataset
            case 'oxford5k'
                src_dir = fullfile('https://www.robots.ox.ac.uk/~vgg/data/oxbuildings');
                dl_files = {'oxbuild_images-v1.tgz'};
            case 'paris6k'
                src_dir = fullfile('https://www.robots.ox.ac.uk/~vgg/data/parisbuildings');
                dl_files = {'paris_1-v1.tgz', 'paris_2-v1.tgz'};
            case 'roxford5k'
                src_dir = fullfile('https://www.robots.ox.ac.uk/~vgg/data/oxbuildings');
                dl_files = {'oxbuild_images-v1.tgz'};
            case 'rparis6k'
                src_dir = fullfile('https://www.robots.ox.ac.uk/~vgg/data/parisbuildings');
                dl_files = {'paris_1-v1.tgz', 'paris_2-v1.tgz'};
            otherwise
                error ('Unkown dataset %s\n', dataset);
        end

        dst_dir = fullfile(datasets_dir, dataset, 'jpg');
        if ~exist(dst_dir, 'dir')
            % for oxford and paris download images
            if strcmp(dataset, 'oxford5k') || strcmp(dataset, 'paris6k')
                fprintf('>> Dataset %s directory does not exist. Creating: %s\n', dataset, dst_dir);
                mkdir(dst_dir);
                for dli = 1:numel(dl_files)
                    dl_file = dl_files{dli};
                    src_file = fullfile(src_dir, dl_file);
                    dst_file = fullfile(dst_dir, dl_file);
                    fprintf('>> Downloading dataset %s archive %s...\n', dataset, dl_file);
                    system(sprintf('wget %s -O %s', src_file, dst_file));
                    fprintf('>> Extracting dataset %s archive %s...\n', dataset, dl_file);
                    % create tmp folder
                    dst_dir_tmp = fullfile(dst_dir, 'tmp');
                    system(sprintf('mkdir %s', dst_dir_tmp));
                    % extract in tmp folder
                    system(sprintf('tar -zxf %s -C %s', dst_file, dst_dir_tmp));
                    % remove all (possible) subfolders by moving only files in dst_dir
                    system(sprintf('find %s -type f -exec mv -i {} %s \\;', dst_dir_tmp, dst_dir));
                    % remove tmp folder
                    system(sprintf('rm -rf %s', dst_dir_tmp));
                    fprintf('>> Extracted, deleting dataset %s archive %s...\n', dataset, dl_file);
                    system(sprintf('rm %s', dst_file));
                end
            % for roxford and rparis just make sym links
            elseif strcmp(dataset, 'roxford5k') || strcmp(dataset, 'rparis6k')
                fprintf('>> Dataset %s directory does not exist. Creating: %s\n', dataset, dst_dir);
                dataset_old = dataset(2:end);
                dst_dir_old = fullfile(datasets_dir, dataset_old, 'jpg');
                mkdir(fullfile(datasets_dir, dataset));
                system(sprintf('ln -s %s %s', dst_dir_old, dst_dir));
                fprintf('>> Created symbolic link from %s jpg to %s jpg\n', dataset_old, dataset);
            end
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
    end

    % Download folder networks/retrieval-SfM-30k/
    src_dir = fullfile('http://cmp.felk.cvut.cz/cnnimageretrieval/data', 'networks', 'retrieval-SfM-30k');
    dst_dir = fullfile(data_dir, 'networks', 'retrieval-SfM-30k');
    dl_files = {'retrievalSfM30k-siamac-alex.mat', 'retrievalSfM30k-siamac-vgg.mat', 'retrievalSfM30k-gem-alex.mat'};
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

    % Download folder networks/retrieval-SfM-120k/
    src_dir = fullfile('http://cmp.felk.cvut.cz/cnnimageretrieval/data', 'networks', 'retrieval-SfM-120k');
    dst_dir = fullfile(data_dir, 'networks', 'retrieval-SfM-120k');
    dl_files = {'retrievalSfM120k-gem-vgg.mat', 'retrievalSfM120k-gem-resnet101.mat'};
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
