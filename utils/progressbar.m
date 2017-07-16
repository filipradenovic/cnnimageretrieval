function progressbar(name, part, opt);
% PROGRESSBAR prints a nice progress bar.
% 
%   progressbar(0) initialize the progress bar.
%   progressbar(PART) prints the current progress bar.
%     PART should be between 0 and 1.
%
%   progressbar(NAME, 0) initialize the progress bar NAME.
%   progressbar(NAME< PART) prints the NAME progress bar.
%     PART should be between 0 and 1.

   if nargin == 1
      if isnumeric(name)
         part = name;
         name = 'unnamed';
      else
         return
      end
   end
   part = max(0, min(1, double(part)));

   global progressbar_data;
   if (isempty(progressbar_data))
      progressbar_data.anonymous = [];
   end

   if (isnumeric(name))
      var = ['n' num2str(name)];
   else
      var = name;
   end

   elapsed = 0;
   if (~isfield(progressbar_data, var) || part == 0)
      eval(['progressbar_data.' var '.start=tic;']);
      eval(['progressbar_data.' var '.astart=now;']);
      eval(['progressbar_data.' var '.last = progressbar_data.' var '.start;']);
      eval(['progressbar_data.' var '.part = 0;']);
   else
      eval(['fromlast = toc(progressbar_data.' var '.last);']);
      if (fromlast < 1 && part ~= 1)
         return % we don't want to print progressbar too often
      end
      eval(['elapsed = toc(progressbar_data.' var '.start);']);

      % another checks if this is new run
      eval(['valid = progressbar_data.' var '.part - 0.01 < ' num2str(part) ';']);
      if (~valid)
         eval(['progressbar_data.' var '.start=tic;']);
         eval(['progressbar_data.' var '.astart=now;']);
      end

      eval(['progressbar_data.' var '.last=tic;']);
      eval(['progressbar_data.' var '.part=' num2str(part) ';']);
   end

   eval(['astart = progressbar_data.' var '.astart;']);
   start = datestr(astart, 'HH:MM');
   if (part == 0)
      afinish = now;
      finish = '?';
   else
      afinish = now + (elapsed/part-elapsed)/3600/24;
      finish = datestr(afinish, 'HH:MM');
   end
   
   dots = '=============================================';
   fprintf('\r>>>> [%6.2f%%] [%-10s] [t %s] [e %s] [r %s] ', ...
           part * 100, dots(1:uint8(floor(part * 10))), ...
           htime(elapsed / part),htime(elapsed), htime(elapsed / part - elapsed) ...
           );

   if (nargin == 3)
      evalin('caller', opt);
   end

   if (part == 1)
      rmfield(progressbar_data, var);
      fprintf('done.\n');
   end

end