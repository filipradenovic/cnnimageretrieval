function setup_cnnimageretrieval()
% SETUP_CNNIMAGERETRIEVAL Setup the toolbox.
%
%   setup_cnnimageretrieval()  Adds the toolbox to MATLAB path.
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 

[root] = fileparts(mfilename('fullpath')) ;

% add paths from this package
addpath(fullfile(root, 'cnnblocks'));
addpath(fullfile(root, 'cnninit'));
addpath(fullfile(root, 'cnntrain'));
addpath(fullfile(root, 'cnnvecs'));
addpath(fullfile(root, 'examples'));
addpath(fullfile(root, 'whiten')); 
addpath(fullfile(root, 'utils')); 