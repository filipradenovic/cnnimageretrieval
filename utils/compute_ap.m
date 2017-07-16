function ap = compute_ap (ranks, nres)
% COMPUTE_AP computes average precision for given ranked indexes.
%
%   AP = compute_ap(RANKS, NUMRES)
%
%   RANKS  : ranks  of positive images
%   NUMRES : number of positive images
%
%   AP     : average precision
%
% Authors: G. Tolias, Y. Avrithis, H. Jegou. 2013. 

% number of images ranked by the system
nimgranks = length (ranks);  
ranks = ranks - 1;	
  
% accumulate trapezoids in PR-plot
ap = 0;

recall_step = 1 / nres;

for j = 1:nimgranks
  rank = ranks(j);
  
  if rank == 0
    precision_0 = 1.0;
  else
    precision_0 = (j - 1) / rank;
  end
  
  precision_1 = j / (rank + 1);
  ap = ap + (precision_0 + precision_1) * recall_step / 2;
end

end
