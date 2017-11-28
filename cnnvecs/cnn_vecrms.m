function v = cnn_vecrms(im, net, L, scales, aggregate)
% CNN_VECRMS computes a multi-scale regional D-dimensional CNN vector for an image.
%
%   V = cnn_vecrms(IM, NET, L, SCALES, AGGREGATE)
%   
%   IM        : Input image, or input image path as a string
%   NET       : CNN network to evaluate on image IM
%	  L         : Number of pyramid levels for regions extraction (DEFAULT: 3)
%   SCALES    : Vector of scales to resize image prior to the descriptor computation (DEFAULT: [1])
%   AGGREGATE : Aggregate over scales if 1, otherwise return one vector per image scale (DEFAULT: 1)
%
%   V         : Multi-scale output, D x 1 if AGGREGATE=1, D x numel(SCALES) if AGGREGATE=0
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 
  
  if ~exist('L'), L = 3; end
  if ~exist('scales'), scales = 1; end
  if ~exist('aggregate'), aggregate = 1; end
  
  minsize = 67;
  net.mode = 'test';
  descvarname = 'l2descriptor';
  prepoolvarname = 'xx0';
  poollayername = 'pooldescriptor';

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

  v = [];
  for s = scales(:)'
    im_ = imresize(im, s);

    im_ = single(im_) - mean(net.meta.normalization.averageImage(:));;
    if size(im_, 3) == 1
  	  im_ = repmat(im_, [1 1 3]);
    end	
  
    if min(size(im_, 1), size(im_, 2)) < minsize
    	im_ = pad2minsize(im_, minsize, 0);
    end
  
    net.vars(net.getVarIndex(prepoolvarname)).precious = 1;
    net.eval({'input', gpuarrayfun(reshape(im_, [size(im_), 1]))});
    vt = gatherfun(squeeze(net.getVar(descvarname).value));
    X = gatherfun(squeeze(net.getVar(prepoolvarname).value));
    net.vars(net.getVarIndex(prepoolvarname)).precious = 0;
  
    if isa(net.layers(net.getLayerIndex(poollayername)).block, 'MAC')
      p = inf;
    elseif isa(net.layers(net.getLayerIndex(poollayername)).block, 'SPOC')
      p = 1;
    elseif isa(net.layers(net.getLayerIndex(poollayername)).block, 'GeM')
      p = gatherfun(net.params(net.layers(net.getLayerIndex(poollayername)).paramIndexes).value);
    else
      disp('UNKNOWN REGIONAL POOLING! INVOKING KEYBOARD!')
      keyboard;
    end
    vt = [vt, vecpostproc(rvec_from_act(X, L, p))];
    vt = sum(vt, 2);
    vt = vecpostproc(vt);
    v = [v, vt];
  end

  if aggregate
    if isa(net.layers(net.getLayerIndex(poollayername)).block, 'GeM')
      p = gatherfun(net.params(net.layers(net.getLayerIndex(poollayername)).paramIndexes).value);
      p = reshape(p, [size(p,3), 1]);
    else 
      p = 1;
    end
    v = bsxfun(@power, v, p);
    v = (sum(v, 2) / size(v, 2)).^(1./p);
    v = vecpostproc(v);
  end

% ------------------------------------------------------------------------------------------------
function vecs = rvec_from_act(X, L, p)
% ------------------------------------------------------------------------------------------------

  ovr = 0.4; % desired overlap of neighboring regions
  steps = [2 3 4 5 6 7]; % possible regions for the long dimension
  
  W = size(X, 2);
  H = size(X, 1);
  
  w = min([W H]);
  w2 = floor(w/2 -1);
  
  b = (max(H, W)-w)./(steps-1);
  [~, idx] = min(abs(((w.^2 - w.*b)./w.^2)-ovr)); % steps(idx) regions for long dimension
  
  % region overplus per dimension
  Wd = 0;
  Hd = 0;
  if H < W  
    Wd = idx;
  elseif H > W
    Hd = idx;
  end
  
  vecs = [];
  
  for l = 1:L
  
    wl = floor(2*w./(l+1));
    wl2 = floor(wl/2 - 1);
  
    b = (W-wl)./(l+Wd-1);  
    if isnan(b), b = 0; end % for the first level
    cenW = floor(wl2 + [0:l-1+Wd]*b) -wl2; % center coordinates
    b = (H-wl)./(l+Hd-1);
    if isnan(b), b = 0; end % for the first level
    cenH = floor(wl2 + [0:l-1+Hd]*b) - wl2; % center coordinates
  
    for i_ = cenH
      for j_ = cenW
        R = X(i_+[1:wl],j_+[1:wl],:);
        if ~min(size(R))
          continue;
        end
        x = vec_from_act(R, p); % get mac per region
        vecs = [vecs, x];
      end
    end
  
  end

% ------------------------------------------------------------------------------------------------
function x = vec_from_act(x, p)
% ------------------------------------------------------------------------------------------------

  if ~max(size(x, 1), size(x, 2))
    x = zeros(size(x, 3), 1, class(x));
    return;
  end

  if isinf(p)
    x = reshape(max(max(x, [], 1), [], 2), [size(x,3) 1]);
  else
    x = reshape(cnn_gem(x, p), [size(x,3) 1]);
  end

% ------------------------------------------------------------------------------------------------
function x = pad2square(x, v)
% ------------------------------------------------------------------------------------------------

  if ~exist('v'), v = 0; end
  x = padarray(x, [max(size(x))-size(x,1), max(size(x))-size(x,2)], v, 'post');

% ------------------------------------------------------------------------------------------------
function x = pad2minsize(x, minsize, v)
% ------------------------------------------------------------------------------------------------

  if ~exist('v'), v = 0; end
  if ~exist('minsize'), minsize = 70; end
  x = padarray(x, [max(minsize-size(x,1), 0), max(minsize-size(x,2), 0)], v, 'post');
