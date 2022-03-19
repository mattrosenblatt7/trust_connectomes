function run_adv_noise(dataset, seed_start, seed_end, algo_all)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   REQUIRED INPUTS
%        dataset            dataset name that you want to load. you will
%                           need to alter these paths (see in code below)
%        seed_start         start random seed
%        seed_end           end random seed. for only one iteration (seed),
%                           set seed_end=seed_start
%        algo_all           which learner to use. this code supports inputs
%                           {'svm'}, {'logistic'}, or both 
%                           {'svm','logistic'}. Please include this input
%                           in brackets as shown
%
%
%   OUTPUTS
%       Save a .mat file for each seed in save_path (set this path below).
%       The file contains the following outputs:
%           y               binary class variable you input
%           N_all           base adversarial noise for each fold, this is
%                           normalized so the edge-wise mean of the noise
%                           pattern is 0.001
%           adv_scale_all   factors by which N_all is scaled (multiplied)
%           ypred           model predictions of size (# particpants x #
%                           adversarial scales). This gives the prediction
%                           at each scale of adversarial noise
%           corr_all        edge-wise correlations between original and
%                           adversarial connectomes at each scale of
%                           adversarial noise
%           cv_idx          cross-validation group membership
%           id_pred         ID predictions for original connectomes
%           id_pred_adv     ID predictions for adversarial connectomes
%           id_pred_eg      ID predictions by edge group (subnetwork) for
%                           original connectomes
%           id_pred_eg_adv  ID predictions by edge group (subnetwork) for
%                           adversarial connectomes
%
%
%
%
% example call:
%  run_adv_noise('hcp', 1, 10, {'svm'})
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% Load data and set save path
addpath(genpath('../../utils'))
save_path = 'my save folder';


% You will need to change this to fit your own data / paths. You should
% load your matrices into a variable called "mat" and your binary response
% variable (e.g., self-reported sex in this case) into a variable called
% "y". The matrices should be size (# nodes x # nodes x # participants)
if strcmp(dataset, 'abcd')
    load('/data22/mri_group/dustinlab_data/dustinlab/Desktop/Matt/abcd/abcd_rest_sex.mat')
    mats = mat;
    clear mat
    y = sex;  
elseif strcmp(dataset, 'hcp')
    load('/data22/mri_group/dustinlab_data/dustinlab/Desktop/Matt/hcp_dat/hcp_rest_sex.mat')
    mats = mat_rest1;
    clear mat_rest1
    y = sex+1;  % 1/2 instead of 0/1 
 
    % for hcp also load other task data (for fingerprinting)
    load('/data22/mri_group/dustinlab_data/dustinlab/Desktop/Matt/hcp_dat/all_mats_updated.mat')
    mats_id = all_mats;
    clear all_mats
elseif strcmp(dataset, 'pnc')
    load('/home/dustinlab/Desktop/Matt/pnc/pnc_rest_update.mat')
    mats = rest_mats;
    clear rest_mats
    y = sex;
end

%% set some parameters
% these are the scales we used for this study, but feel free to change
% these
adv_scale_all = 0:30;

% set some parameters
nsub=length(sex);
perfeat=1;  % fraction of features to use (1 uses all features)
kfolds=10;
%% Run adversarial noise

% load in Shen edge groups
load('../utils/shen_subnetwork_edges.mat', 'edge_groups');
eg_unique = unique(edge_groups);

% convert matrix to edges
X = mat2edge(mats);

% get classes (e.g., 1 for F and 2 for male)
y_class1 = min(y); y_class2 = max(y);  

% loop over all algorithms
for algo_idx = 1:length(algo_all)
    learner = algo_all{algo_idx};
    
    % loop over all seeds
    for seed = seed_start:seed_end
        rng(seed)
        cv_idx = cv_indices(nsub, 10);
        disp(['****************************', num2str(seed),'*****************************'])
        
        ypred = zeros(nsub, length(adv_scale_all));
        corr_all = zeros(nsub, length(adv_scale_all));  % correlation between original / adversarial edges
        
        % id only used for hcp
        id_pred = zeros(nsub, 9);  % 9 hcp tasks
        id_pred_eg = zeros(nsub, length(eg_unique));
        id_pred_adv = zeros(nsub, 9);  % 9 hcp tasks
        id_pred_eg_adv = zeros(nsub, length(eg_unique));
        
        % initialize variable to store noise patterns
        N_all = zeros(size(X, 1), kfolds); 
        
        % k-fold CV
        for fold_idx = 1:kfolds
            disp(['**** Fold ', num2str(fold_idx), ' ****']) 
            
            % get train and test indices for this fold
            train_idx = find(cv_idx~=fold_idx);
            test_idx = find(cv_idx==fold_idx);
            Xtrain = X(:, train_idx)';
            Xtest = X(:, test_idx)';
            ytrain = y(train_idx);
            ytest = y(test_idx);
            
            % feature selection (t-test)
            [~, p_all] = ttest2(Xtrain(ytrain==y_class1, :), Xtrain(ytrain==y_class2, :));
            feat_loc = find(p_all<=prctile(p_all, 100*perfeat));
            
            % search for regularization parameters (Note: you can typically remove
            % the optimization and see similar results in much shorter time)
            mdl = fitclinear(Xtrain(:, feat_loc), ytrain, 'Learner', learner, ...
                'Regularization', 'Ridge', 'OptimizeHyperparameters', {'Lambda'},...
                'HyperparameterOptimizationOptions', struct('Kfold', 5, 'ShowPlots', 0, 'Verbose', 0));
            
            % for linear models, noise pattern (N) proportional to
            % coefficients
            N = mdl.Beta';
            N = N / (mean(abs(N))) * .001; 
            
            % evaluate effect of adversarial noise at multiple scales
            for adv_scale_idx = 1:length(adv_scale_all)
                adv_scale = adv_scale_all(adv_scale_idx); 
                
                % add (scaled) adversarial noise
                Xtest_adv = Xtest(:, feat_loc);
                Xtest_adv(ytest==y_class1, :) = Xtest_adv(ytest==y_class1, :) + adv_scale*N;
                Xtest_adv(ytest==y_class2, :) = Xtest_adv(ytest==y_class2, :) - adv_scale*N;
                
                % predict
                ypred(test_idx, adv_scale_idx) = predict(mdl, Xtest_adv);
                
                % edge corrPelations between original and adversarial data
                corr_all(test_idx, adv_scale_idx) = diag(corr(Xtest_adv', Xtest(:, feat_loc)'));
                
                % stop if 0% acc
                if mean(ypred(test_idx, adv_scale_idx)==ytest)==0
                   break; 
                end
                
            end
                        
            % store adversarial noise
            N_all(feat_loc, fold_idx) = N;
            
            % fingerprint within each fold (for noise pattern that caused 0% acc)
            if strcmp(dataset, 'hcp') && perfeat==1
                
                % fingerprint with all tasks
                for task_idx = 1:size(mats_id, 4)
                    Xtest_session2 = mat2edge(mats_id(:, :, test_idx, task_idx));
                    Xtest_session2 = Xtest_session2(feat_loc, :)';
                    
                    % normal fingerprint               
                    cmat=corr(Xtest', Xtest_session2');
                    [~, max_corr_loc] = max(cmat, [], 2);
                    id_pred(test_idx, task_idx) = test_idx(max_corr_loc);
                    
                    % adv fingerprint
                    cmat_adv=corr(Xtest_adv', Xtest_session2');
                    [~, max_corr_loc_adv] = max(cmat_adv, [], 2);
                    id_pred_adv(test_idx, task_idx) = test_idx(max_corr_loc_adv);
                end
                
                % fingerprint by subnetwork
                eg_by_feat_loc = edge_groups(feat_loc);
                Xtest_session2 = mat2edge(mats_id(:, :, test_idx, 3));  % 3 is rest2
                Xtest_session2 = Xtest_session2(feat_loc, :)';
                for eg_idx = 1:length(eg_unique)
                    
                    % baseline fingerprint
                    cmat =corr(Xtest(:, eg_by_feat_loc==eg_idx)', Xtest_session2(:, eg_by_feat_loc==eg_idx)');
                    [~, max_corr_loc] = max(cmat, [], 2);
                    id_pred_eg(test_idx, eg_idx) = test_idx(max_corr_loc);
                    
                    % adv fingerprint
                    cmat_adv=corr(Xtest_adv(:, eg_by_feat_loc==eg_idx)', Xtest_session2(:, eg_by_feat_loc==eg_idx)');
                    [~, max_corr_loc_adv] = max(cmat_adv, [], 2);
                    id_pred_eg_adv(test_idx, eg_idx) = test_idx(max_corr_loc_adv);
                end
            end             
        end
          
        % save results for each seed
        save([save_path, '/adv_noise_', dataset, '_', algo_all{algo_idx}, '_perfeat', num2str(perfeat), '_seed', num2str(seed), '.mat'],...
            'cv_idx', 'N_all', 'ypred', 'corr_all', 'id_pred', 'id_pred_eg', 'id_pred_adv', 'id_pred_eg_adv', 'y', 'adv_scale_all')
    end
end
