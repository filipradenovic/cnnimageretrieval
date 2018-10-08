function v = cnn_vecms_sketch(im, net, emodel, scales, mirror, aggregate)
% CNN_VECMS_SKETCH computes a multi-scale D-dimensional CNN sketch vector for an image.
%
%   V = cnn_vecms_sketch(IM, NET, EMODEL, SCALES, MIRROR, AGGREGATE)
%   
%   IM        : Input image, or input image path as a string
%   NET       : CNN network to evaluate on image IM
%   EMODEL    : Edge detector model, 0 means do not extract edges because input is already sketch
%   SCALES    : Vector of scales to resize image prior to the descriptor computation (DEFAULT: [1])
%   MIRROR    : Mirror (horizontal flip) of image for the descriptor computation (DEFAULT: 0)
%   AGGREGATE : Aggregate over scales and mirror if 1, otherwise return one vector per image scale and mirror (DEFAULT: 1)
%
%   V         : Multi-scale output, D x 1 if AGGREGATE=1, D x numel(SCALES) x (MIRROR+1) if AGGREGATE=0
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2018. 

    if ~exist('scales'), scales = 1; end
    if ~exist('mirror'), mirror = 0; end
    if ~exist('aggregate'), aggregate = 1; end

    padsize = 30; 

    v = [];
    for i = 1:(mirror+1)*numel(scales)
        if i == numel(scales)+1, im = fliplr(im); end;
        imr = imresize(im, scales(mod(i-1,numel(scales))+1));

        if ~isstruct(emodel)
            %% Treated as SKETCH
            if size(imr, 3) > 1; imr = rgb2gray(imr);  end
            % making sure its a binary sketch image
            imr = single(imr); imr = 255 * ((imr - min(imr(:))) ./ (max(imr(:)) - min(imr(:))));
            imr = single(imr < 0.8 * max(imr(:)));
            imr = padarray(imr, [padsize padsize], 0, 'both');
            % unifying sketches:
            % morphological thinning followed by dilation
            imr = single(bwmorph(imr, 'thin', inf));
            imr = single(imdilate(imr,strel('disk',1,0)));
            imr = single(bwmorph(imr, 'thin', inf));
            imr = single(imdilate(imr,strel('disk',2,0)));
            % final sketch2edgemap
            e = imr;
        else
            %% Treated as IMAGE
            if size(imr, 3) == 1; imr = repmat(imr, 1, 1, 3); end
            [edg, ori] = edgesDetect(imr, emodel);
            e = padarray(edg, [padsize padsize], 0, 'both');
        end

        v = [v, cnn_vec_edge(e, net)];
    end

    if aggregate
        v = sum(v, 2);
        v = vecpostproc(v);
    end


function v = cnn_vec_edge(im, net)
% CNN_VEC_EDGE computes a D-dimensional CNN edge vector for a given edge-map.
%
%   V = cnn_vec_edge(IM, NET)
%   
%   IM  : Input edge-map
%   NET : CNN network to evaluate on image IM
%
%   V   : D-dimensional vector
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2018. 

    minsize = 67;
    net.mode = 'test';
    descvarname = 'l2descriptor';

    use_gpu = strcmp(net.device, 'gpu');
    if use_gpu
        gpuarrayfun = @(x) gpuArray(x);
        gatherfun = @(x) gather(x);
    else
        gpuarrayfun = @(x) x; % do not convert to gpuArray
        gatherfun = @(x) x; % do not gather
    end

    if isstr(im)
        im = imread(im);
    end
    
    im = single(im) - mean(net.meta.normalization.averageImage(:));
    if min(size(im, 1), size(im, 2)) < minsize
        im = pad2minsize(im, minsize, 0);
    end

    net.eval({'input', gpuarrayfun(reshape(im, [size(im), 1]))});

    v = gatherfun(squeeze(net.getVar(descvarname).value));