function [p_all, corr_strength, corr_assort, corr_cc, id_pred, id_pred_enhanced, id_pred_eg, id_pred_eg_enhanced] =...
    analysis_pipeline( original_mats, enhanced_mats, mats_id, cv_idx)
% calls analysis functions

% brain connectivity toolbox https://sites.google.com/site/bctnet/
addpath(genpath('/data22/mri_group/dustinlab_data/dustinlab/Desktop/Matt/trust/adversarial_noise/corr_adv/analysis/BCT'))
addpath(genpath('../../utils'))

nsub = size(original_mats, 3);

% set diagonals to zero
original_mats = original_mats - diag(diag(original_mats(:, :, 1)));
enhanced_mats = enhanced_mats - diag(diag(enhanced_mats(:, :, 1)));

% convert to edges
original_edges = mat2edge(original_mats);
enhanced_edges = mat2edge(enhanced_mats);

% ks test
p_all = zeros(nsub, 1);
for sub_idx = 1:nsub
    % ks test
    [~, p_all(sub_idx)] = kstest2( original_edges(:, sub_idx), enhanced_edges(:, sub_idx) );
end


% correlations between edges
edge_corr = mean(diag(corr(original_edges, enhanced_edges)));


% **** Graph metrics ****
corr_strength = zeros(nsub, 1);
corr_assort = zeros(nsub, 1);
corr_cc = zeros(nsub, 1);
for sub_idx = 1:nsub
    original_strength = strengths_und_sign(original_mats(:, :, sub_idx));
    enhanced_strength = strengths_und_sign(enhanced_mats(:, :, sub_idx));
    corr_strength(sub_idx) = corr(original_strength(:), enhanced_strength(:));
    
    original_assort = local_assortativity_wu_sign(original_mats(:, :, sub_idx));
    enhanced_assort = local_assortativity_wu_sign(enhanced_mats(:, :, sub_idx));
    corr_assort(sub_idx) = corr(original_assort(:), enhanced_assort(:));
    
    original_cc = clustering_coef_wu_sign(original_mats(:, :, sub_idx));
    enhanced_cc = clustering_coef_wu_sign(enhanced_mats(:, :, sub_idx));
    corr_cc(sub_idx) = corr(original_cc(:), enhanced_cc(:));
end


% **** ID rate ****
if exist('mats_id', 'var') && exist('cv_idx', 'var')
    % load in Shen atlas subnetworks
    load('/data22/mri_group/dustinlab_data/dustinlab/Desktop/Matt/trust/github/adversarial_noise/shen_subnetwork_edges.mat', 'edge_groups');
    eg_unique = unique(edge_groups);
    
    ntasks = size(mats_id, 4);
    kfolds = length(unique(cv_idx));
    
    % initialize arrays
    id_pred = zeros(nsub, ntasks);
    id_pred_enhanced = zeros(nsub, ntasks);
    id_pred_eg = zeros(nsub, length(eg_unique));
    id_pred_eg_enhanced = zeros(nsub, length(eg_unique));
    
    % split based on folds for ID (for reasonable sample size)
    for k = 1:kfolds
        
        test_idx = find(cv_idx==k);
        Xtest = original_edges(:, test_idx);
        Xtest_enhanced = enhanced_edges(:, test_idx);
        
        % fingerprint with all tasks
        for task_idx = 1:ntasks
            Xtest_session2 = mat2edge(mats_id(:, :, test_idx, task_idx));
            
            % normal fingerprint
            cmat=corr(Xtest, Xtest_session2);
            [~, max_corr_loc] = max(cmat, [], 2);
            id_pred(test_idx, task_idx) = test_idx(max_corr_loc);
            
            % enhanced fingerprint
            cmat_enhanced=corr(Xtest_enhanced, Xtest_session2);
            [~, max_corr_loc_enhanced] = max(cmat_enhanced, [], 2);
            id_pred_enhanced(test_idx, task_idx) = test_idx(max_corr_loc_enhanced);
        end
        
        % fingerprint by subnetwork
        Xtest_session2 = mat2edge(mats_id(:, :, test_idx, 3));  % 3 is rest2
        for eg_idx = 1:length(eg_unique)
            
            % baseline fingerprint
            cmat = corr(Xtest(edge_groups==eg_idx, :), Xtest_session2(edge_groups==eg_idx, :));
            [~, max_corr_loc] = max(cmat, [], 2);
            id_pred_eg(test_idx, eg_idx) = test_idx(max_corr_loc);
            
            % enhanced fingerprint
            cmat_enhanced=corr(Xtest_enhanced(edge_groups==eg_idx, :), Xtest_session2(edge_groups==eg_idx, :));
            [~, max_corr_loc_enhanced] = max(cmat_enhanced, [], 2);
            id_pred_eg_enhanced(test_idx, eg_idx) = test_idx(max_corr_loc_enhanced);
        end
        
        
    end
    
else
    id_pred = 0; id_pred_enhanced = 0; id_pred_eg = 0; id_pred_eg_enhanced = 0;
    
end
