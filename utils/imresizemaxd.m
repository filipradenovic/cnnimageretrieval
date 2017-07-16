function im = imresizemaxd(im, sz, increase_allowed, method)
% IMRESIZEMAXD resizes image so that longer edge is maximum to the given size.
%   
%   IM = IMRESIZEMAXD(IM, SIZE)
%     Resize IM so that longer edge is maximum SIZE.
%
%   IM = IMRESIZEMAXD(IM, SIZE, INCREASE_ALLOWED, METHOD)
%     Resize IM so that longer edge is maximum SIZE.
%     INCREASE_ALLOWED defines if smaller images will be upscaled. Default is TRUE.
%     METHOD defined the MATLAB supported imresize method. Default is 'BICUBIC'.

	if ~exist('increase_allowed'), increase_allowed = 1; end
	if ~exist('method'), method = 'bicubic'; end	

	if size(im,1) <= sz && size(im,2) <= sz && ~increase_allowed
		return;
	end
	if size(im,1) > size(im,2)
		im = imresize(im, [sz NaN], method);
	elseif size(im,1) < size(im,2)
		im = imresize(im, [NaN sz], method);
	else
		im = imresize(im, [sz sz], method);
	end