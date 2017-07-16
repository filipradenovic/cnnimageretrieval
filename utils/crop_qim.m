function im = crop_qim(im, bbx, imgsz)
% CROP_QIM crops query image with defined bounding box.
%
%   IM = crop_qim(IM, BBX)  Crop IM with BBX.
%   
%   IM = crop_qim(IM, BBX, IMG_SIZE) Resize image so that longer edge is IMG_SIZE.
%     After that crop IM with BBX.

	allow_increase = 0;

	if size(im, 1) == 1, im = imread(im); end
	if ~exist('imgsz'), 
		im = im(bbx(2):min(bbx(4),size(im,1)), bbx(1):min(bbx(3),size(im,2)), :); 
		return;
	end

	bbx = uint32(max(imgsz * (bbx + 1) / max(size(im,1), size(im,2)), 1));
	im = imresizemaxd(im, imgsz, allow_increase);
	im = im(bbx(2):min(bbx(4),size(im,1)), bbx(1):min(bbx(3),size(im,2)), :);
