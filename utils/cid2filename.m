function filename = cid2filename(cid, prefix)
% CID2FILENAME creates a training image path out of its CID name
%
%   FILENAME = cid2filename(CID, PREFIX)
%
%   CID      : name of the image
%   PREFIX   : root directory where images are saved
%
%   FILENAME : full image filename

   filename = fullfile(prefix, cid(end-1:end), cid(end-3:end-2), cid(end-5:end-4), cid);
end