function data = get_images(images, opts)
% GET_IMAGES gets set of IMAGES from list of IMAGE_PATHS.
%
%   IMAGES = get_images(IMAGE_PATHS, opts)
%
%   Check supported opts in function LOAD_OPTS_TRAIN.M
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 

opts.useGpu = numel(opts.gpus) >= 1;
opts.numThreads = 12;
opts.batchSize = 256;

data = cell(1, numel(images));

numGpus = numel(opts.gpus);
if numGpus > 0
  clear mex;
  gpuinfo = gpuDevice(opts.gpus(1));
  fprintf('>>>> Running on GPU %s with Index %d\n', gpuinfo.Name, gpuinfo.Index);  
end

progressbar(0);
for t=1:opts.batchSize:numel(images)
  time = tic;
  batch = t : min(t+opts.batchSize-1, numel(images));
  d = getImageBatch(images(batch), opts);
  data(batch) = arrayfun(@(x) uint8(gather(d{x})), 1:numel(d), 'UniformOutput', false);
  progressbar(batch(end) / numel(images));
end

data = cellfun(@(x) imchannelscheck(x), data, 'UniformOutput', false);

if numGpus > 0
  clear mex;
  gpuinfo = gpuDevice(opts.gpus(1));
end

%----------------------------------------------------
function data = getImageBatch(imagePaths, opts)
% GETIMAGEBATCH  Load and jitter a batch of images.
%----------------------------------------------------

args = [];

args{1} = {imagePaths, ...
           'NumThreads', opts.numThreads, ...
           'Interpolation', 'bilinear', ...
           'CropSize', opts.jitterScale, ...
           'CropAnisotropy', 1, ...
           'Brightness', opts.jitterBrightness, ...
           'Contrast', opts.jitterContrast, ...
           'Saturation', opts.jitterSaturation};

% if crop resize it while loading
% otherwise it will be resized after loading
% to longer size equaling opts.imageSize
if opts.crop
  args{end+1} = {'Resize', [opts.imageSize opts.imageSize]};
end

if opts.jitterLocation
  args{end+1} = {'CropLocation', 'random'};
else
  args{end+1} = {'CropLocation', 'center'};
end

if opts.useGpu
  args{end+1} = {'Gpu'};
end

args = horzcat(args{:});

data = vl_imreadjpeg(args{:});

% if no crop resize to longer size equaling opts.imageSize
if ~opts.crop
  data = cellfun(@(x) imresizemaxd(x, opts.imageSize), data, 'UniformOutput', false);
end

%----------------------------------------------------
function im = imchannelscheck(im)
%----------------------------------------------------
if size(im, 3) == 1
  im = repmat(im, [1 1 3]);
end