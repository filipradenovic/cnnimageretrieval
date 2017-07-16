function htime_str = htime(sec)
% HTIME formats number of seconds to human readable time.
%
%   HTIME_STRING = htime(SECONDS)

	if isnan(sec)
		htime_str = '?';
	elseif sec < 1 * 60  % max 1 minute
		htime_str = [num2str(sec, '%.1f') 's'];
	elseif sec < 1 * 60 * 60 % max 1 hour 
		htime_str = [num2str(sec/60, '%.1f') 'm'];
	elseif sec < 2 * 60 * 60 * 24 % max 2 days
		htime_str = [num2str(sec/60/60, '%.1f') 'h'];
	elseif sec < 1 * 60 * 60 * 24 * 365 % max 1 year
		htime_str = [num2str(sec/60/60/24, '%.1f') 'd'];
	else
		htime_str = [num2str(sec/60/60/24/365, '%.1f') 'r'];
	end
end
