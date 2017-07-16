function v = cnn_vecr(im, net, L)
% CNN_VECR computes a regional D-dimensional CNN vector for an image.
%
%   V = cnn_vecr(IM, NET, L)
%   
%   IM  : Input image, or input image path as a string
%   NET : CNN network to evaluate on image IM
%	  L   : Number of pyramid levels for regions extraction (Default: 3)
%
%   V   : Aggregated regional D-dimensional vector
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 
  
  if ~exist('L'), L = 3; end
  
  minsize = 67;
  net.mode = 'test';
  descvarname = 'xx0';

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

  im = single(im) - mean(net.meta.normalization.averageImage(:));;
  if size(im, 3) == 1
	im = repmat(im, [1 1 3]);
  end	

  if min(size(im, 1), size(im, 2)) < minsize
  	im = pad2minsize(im, minsize, 0);
  end

  net.vars(net.getVarIndex(descvarname)).precious = 1;
  net.eval({'input', gpuarrayfun(reshape(im, [size(im), 1]))});
  X = gatherfun(squeeze(net.getVar(descvarname).value));
  net.vars(net.getVarIndex(descvarname)).precious = 0;

  v = rmac_from_act(X, L);
  v = bsxfun(@rdivide,v,sqrt(sum(v.^2)));
  v = sum(v, 2);
  v = bsxfun(@rdivide,v,sqrt(sum(v.^2)));

% ------------------------------------------------------------------------------------------------
function vecs = rmac_from_act(X, L)
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
        x = mac_from_act(R); % get mac per region
        vecs = [vecs, x];
      end
    end
  
  end

% ------------------------------------------------------------------------------------------------
function x = mac_from_act(x)
% ------------------------------------------------------------------------------------------------

  if ~max(size(x, 1), size(x, 2))
	x = zeros(size(x, 3), 1, class(x));
	return;
  end

  x = reshape(max(max(x, [], 1), [], 2), [size(x,3) 1]);

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
