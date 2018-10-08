function emodel = load_edgedetector(data_dir)
% LOAD_EDGEDETECTOR Checks, and, if required, downloads the necessary code and data for edge detector.
%       Then, it setups the paths, and loads the edge detector model.
%
%   load_edgedetector(DATA_ROOT) Checks, and, if required, downloads the necessary code and data for edge detector.
%     Then, it setups the paths, loads and returns the edge detector model. The edge detector used is from the paper:
%       P. Dollar, L. Zitnick, Fast edge detection using structured forests, TPAMI 2015

    % Create data folder if it does not exist
    if ~exist(data_dir, 'dir')
        mkdir(data_dir);
    end

    % Create datasets folder if it does not exist
    edgedetector_dir = fullfile(data_dir, 'edgedetector');
    if ~exist(edgedetector_dir, 'dir')
        mkdir(edgedetector_dir);
    end

    % Download Dollar MATLAB toolbox
    toolbox_dir = fullfile(edgedetector_dir, 'toolbox-master');
    if ~exist(toolbox_dir, 'dir')
        fprintf('>> Downloading Dollar MATLAB toolbox\n');
        src_file = 'https://github.com/pdollar/toolbox/archive/master.zip';
        dst_file = fullfile(edgedetector_dir, 'master.zip');
        system(sprintf('wget %s -O %s', src_file, dst_file));
        system(sprintf('unzip -q %s -d %s', dst_file, edgedetector_dir));
        system(sprintf('rm %s', dst_file));
    end

    % Download Dollar edge detector
    edges_dir = fullfile(edgedetector_dir, 'edges-master');
    if ~exist(edges_dir, 'dir')
        fprintf('>> Downloading Dollar edge detector toolbox\n');
        src_file = 'https://github.com/pdollar/edges/archive/master.zip';
        dst_file = fullfile(edgedetector_dir, 'master.zip');
        system(sprintf('wget %s -O %s', src_file, dst_file));
        system(sprintf('unzip -q %s -d %s', dst_file, edgedetector_dir));
        system(sprintf('rm %s', dst_file));

        % % Compile mex files (linux64 binaries included, no need to compile)
        % mex_files = {'edgesDetectMex.cpp', 'edgesNmsMex.cpp', 'spDetectMex.cpp', 'edgeBoxesMex.cpp'};
        % for mfi = 1:numel(mex_files)
        %     mex_file = fullfile(edges_dir, 'private', mex_files{mfi});
        %     out_dir = fullfile(edges_dir, 'private');
        %     mex(mex_file, '-outdir', out_dir);
        % end
    end

    % Load and setup emodel
    addpath(genpath(toolbox_dir));
    addpath(genpath(edges_dir));
    emodel_file = fullfile(edges_dir, 'models', 'forest', 'modelBsds.mat');
    emodel = load(emodel_file); 
    emodel = emodel.model; 
    emodel.opts.nms = 0; 
    emodel.opts.nThreads = 4; 
    emodel.opts.multiscale = 0; 
    emodel.opts.sharpen = 2;