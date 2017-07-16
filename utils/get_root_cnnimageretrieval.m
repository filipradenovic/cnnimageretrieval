function root = get_root_cnnimageretrieval()
% GET_ROOT_CNNIMAGERETRIEVAL Get the root path of the toolbox.
%
%   ROOT_PATH = get_root_cnnimageretrieval()

root = fileparts(fileparts(mfilename('fullpath'))) ;