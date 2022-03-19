function run_trust_taskswap(all_mats, train_task_idx, y, seed_start, seed_end, algo_all)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   REQUIRED INPUTS
%           all_mats        all connectome data, size (# nodes x # nodes x
%                           # participants x # tasks)
%           train_task_idx  index for task you will be training with (i.e.,
%                           if you want to train with resting-state data
%                           and that is the second set of connectomes in
%                           your all_mats matrix, put 2)
%           y               binary response variable
%           seed_start      starting random seed
%           seed_end        end random seed. for only one iteration (seed),
%                           set seed_end=seed_start
%           algo_all           which learner to use. this code supports inputs
%                           {'svm'}, {'logistic'}, or both 
%                           {'svm','logistic'}. Please include this input
%                           in brackets as shown
%
%
%   OUTPUTS
%       Save a .mat file for each seed in save_path (set this path below).
%       The file contains the following outputs:
%           y               binary class variable you input
%           ypred           model predictions of size (# particpants x #
%                           tasks). For each participant, predictions are
%                           made using all available tasks, with a model
%                           that was trained using the task specified by
%                           train_task_idx
%
% example call:
%   run_trust_taskswap(all_mats, 2, sex, 1, 10, {'svm'})
%   In the above example, you might have a matrix all_mats where the order
%   is (# nodes x # nodes x # participants x [breath-hold, rest]). If you
%   want to train on resting-state data and evaluate performance using both
%   breath-hold and resting-state scans, call the function as above.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% setting some parameters
addpath(genpath('../../utils'))
save_path = './tmp';

nsub=size(all_mats, 3);
ntasks = size(all_mats, 4);
X_traintask = mat2edge(all_mats(:, :, :, train_task_idx));  % convert train matrix to edges
y_class1 = min(y); y_class2 = max(y);  % get classes (e.g., 1 for F and 2 for male)

% model parameters
perfeat=1;  % fraction of features to use (1 uses all features)
kfolds=10;
%% running task swap prediction
% loop over all algorithms
for algo_idx = 1:length(algo_all)
    learner = algo_all{algo_idx};
    
    % loop over all seeds
    for seed = seed_start:seed_end
        rng(seed)
        cv_idx = cv_indices(nsub, kfolds);
        disp(['****************************', num2str(seed),'*****************************'])
        
        ypred = zeros(nsub, ntasks);
        
        
        for fold_idx = 1:kfolds
            disp(['**** Fold ', num2str(fold_idx), ' ****'])
            % get train and test
            train_idx = find(cv_idx~=fold_idx);
            test_idx = find(cv_idx==fold_idx);
            Xtrain = X_traintask(:, train_idx)';
            ytrain = y(train_idx);
            
            % feature selection
            [~, p_all] = ttest2(Xtrain(ytrain==y_class1, :), Xtrain(ytrain==y_class2, :));
            feat_loc = find(p_all<=prctile(p_all, 100*perfeat));
            
            % search for regularization parameters w/ nested CV
            % (Note: you can typically remove the optimization and see similar results in much shorter time)
            mdl = fitclinear(Xtrain(:, feat_loc), ytrain, 'Learner', learner, ...
                'Regularization', 'Ridge', 'OptimizeHyperparameters', {'Lambda'},...
                'HyperparameterOptimizationOptions', struct('Kfold', 5, 'ShowPlots', 0, 'Verbose', 0));
            
            for task_idx = 1:ntasks
                Xtest = mat2edge(all_mats(:, :, test_idx, task_idx))';
                ypred(test_idx, task_idx) = predict(mdl, Xtest(:, feat_loc));
            end
            
        end
        
        save([save_path, '/swap_', algo_all{algo_idx}, '_perfeat', num2str(perfeat), '_seed', num2str(seed), '.mat'],...
            'y', 'ypred')
    end
    

end