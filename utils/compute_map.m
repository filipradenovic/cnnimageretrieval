function [map, aps] = compute_map (ranks, gnd, verbose)
% COMPUTE_MAP computes the mAP for a given set of returned results.
%
%   mAP = compute_map (RANKS, GND);
%
%   RANKS starts from 1, size(ranks) = db_size X #queries.
%   Junk results (e.g., the query itself) should be declared in the gnd stuct array
%
% Authors: G. Tolias, Y. Avrithis, H. Jegou. 2013. 

if nargin < 3
  verbose = false;
end

map = 0;
nq = numel (gnd);   % number of queries
aps = zeros (nq, 1);

for i = 1:nq
  qgnd = gnd(i).ok; 
  if isfield (gnd(i), 'junk')
    qgndj = gnd(i).junk; 
  else 
    qgndj = []; 
  end
  
	% positions of positive and junk images
  [~, pos] = intersect (ranks (:,i), qgnd);
  [~, junk] = intersect (ranks (:,i), qgndj);

	pos = sort(pos);
	junk = sort(junk);

	k = 0;  
	ij = 1;

	if length (junk)
		% decrease positions of positives based on the number of junk images appearing before them
		ip = 1;
		while ip <= numel (pos)

			while ( ij <= length (junk) & pos (ip) > junk (ij) )
				k = k + 1;
				ij = ij + 1;
			end

			pos (ip) = pos (ip) - k;
			ip = ip + 1;
		end
	end

  ap = compute_ap (pos, length (qgnd));
	
  if verbose
    fprintf ('query no %d -> gnd = ', i);
    fprintf ('%d ', qgnd);
    fprintf ('\n              tp ranks = ');
    fprintf ('%d ', pos);
    fprintf (' -> ap=%.3f\n', ap);
  end
  map = map + ap;
	aps (i) = ap;

end
map = map / nq;

end
