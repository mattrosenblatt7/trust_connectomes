function indices = cv_indices(n, kfolds)
% make indices for cross validation
randinds=randperm(n);
ksample=floor(n/kfolds);
all_ind=[];
fold_size=floor(n/kfolds);
for idx=1:kfolds
    all_ind=[all_ind; idx*ones(fold_size, 1)];
end
leftover=mod(n, kfolds);
indices=[all_ind; randperm(kfolds, leftover)'];
indices=indices(randperm(length(indices)));
