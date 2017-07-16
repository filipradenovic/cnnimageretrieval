function db = augment_database(db, opts)
% AUGMENT_DATABASE sets the DB for one epoch of training
%
%   DB = augment_database(DB, opts)
%
%   opts.epochSize(1) : Maximum number of queries for this epoch
%   opts.epochSize(2) : Maximum number of images  for this epoch
%
%   Check other supported opts in function LOAD_OPTS_TRAIN.M
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 

% only keep unique query/pos pairs
qpidxs = unique([db.train.qidxs', db.train.pidxs'], 'rows', 'stable')'; qidxs = qpidxs(1,:); pidxs = qpidxs(2,:);
if (numel(qidxs) < opts.epochSize(1)), opts.epochSize(1) = numel(qidxs); end
if (numel(db.train.cids)  < opts.epochSize(2)), opts.epochSize(2) = numel(db.train.cids); end
fprintf('>> Number of queries: %d. Number of images: %d\n', opts.epochSize(1), opts.epochSize(2));

% take random queries
keepidxs = randperm(numel(qidxs));
qidxs = qidxs(keepidxs(1 : opts.epochSize(1)));
pidxs = pidxs(keepidxs(1 : opts.epochSize(1)));

% take random cids (including query+pos)
[qpunqidxs, qp2qpunq, qpunq2qp] = unique([qidxs, pidxs], 'stable');
keepidxs = randperm(numel(db.train.cids));
keepidxs = setdiff(keepidxs, qpunqidxs);
keepidxs = [qpunqidxs, keepidxs(1 : opts.epochSize(2) - numel(qpunqidxs))];

% use only selected data for db
db.train.cids    = db.train.cids(keepidxs);
db.train.cluster = db.train.cluster(keepidxs);
db.train.qidxs   = qpunq2qp(1 : numel(qidxs))';
db.train.pidxs   = qpunq2qp(numel(qidxs)+1 : end)';

% load images
if isfield(db.train, 'data')
	fprintf('>> Images provided with DB, no loading\n');
	db.train.data = db.train.data(keepidxs);
else
	fprintf('>> Images not provided with DB, loading...\n');
	images = cellfun(@(x) cid2filename(x, opts.imageDir), db.train.cids, 'UniformOutput', false);
	db.train.data = get_images(images, opts);
end
 
% flip jittering has to be done after loading because we flip query/pos in pairs
if opts.jitterFlip

	fprintf('>> Horizontal flip of positive pairs...\n'); t=tic;
	flipped = false(1, numel(db.train.data));
	flidxs = randperm(numel(db.train.qidxs));
	flidxs = flidxs(1 : floor(0.5*numel(db.train.qidxs)));
	for fl = flidxs
		db.train.data{db.train.qidxs(fl)} = fliplr(db.train.data{db.train.qidxs(fl)}); flipped(db.train.qidxs(fl)) = true;
		db.train.data{db.train.pidxs(fl)} = fliplr(db.train.data{db.train.pidxs(fl)}); flipped(db.train.pidxs(fl)) = true;
	end

	% check if some pairs were half flipped (due to overlap of query and positive images)
	half_flip = true;
	while (half_flip)
		half_flip = false;
		for fl = 1:numel(db.train.qidxs)
			if xor(flipped(db.train.qidxs(fl)), flipped(db.train.pidxs(fl)))
				half_flip = true;
				if ~flipped(db.train.qidxs(fl))
					db.train.data{db.train.qidxs(fl)} = fliplr(db.train.data{db.train.qidxs(fl)}); flipped(db.train.qidxs(fl)) = true;
				else
					db.train.data{db.train.pidxs(fl)} = fliplr(db.train.data{db.train.pidxs(fl)}); flipped(db.train.pidxs(fl)) = true;
				end
			end
		end
	end
	fprintf('>>>> done in %s\n', htime(toc(t)));

	fprintf('>> Horizontal flip of potential negatives (image pool)...\n'); t=tic;
	flidxs = 1:numel(db.train.data);
	flidxs = setdiff(flidxs, db.train.qidxs(:)');
	flidxs = setdiff(flidxs, db.train.pidxs(:)');
	flidxs = flidxs(randperm(numel(flidxs)));
	flidxs = flidxs(1 : floor(0.5*numel(flidxs)));
	for fl = flidxs
	 	db.train.data{fl} = fliplr(db.train.data{fl});
	end
	fprintf('>>>> done in %s\n', htime(toc(t)));

end