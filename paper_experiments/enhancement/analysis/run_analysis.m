function run_analysis(dataset, seed_start, seed_end)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run the analysis on enhanced data, including edge correlations, graph
% metrics, and ID rates
%
%   REQUIRED INPUTS
%        dataset            dataset name that you want to load. you will
%                           need to alter these paths (see in code below)
%        seed_start         start random seed
%        seed_end           end random seed. for only one iteration (seed),
%                           set seed_end=seed_start
%
%   OUTPUTS
%       Save a .mat file for each seed in save_path (set this path below).
%       The file contains the following outputs:
%           p_all               correlations between predicted and meaured
%                               behavior (Pearson's r)
%           corr_assort         correlations between original/enhanced
%                               assortativity
%           corr_cc             correlations between origina/enhanced
%                               clustering coefficient
%           corr_strength       correlations between original/enhanced
%                               strengths
%           id_pred             id predictions for original data
%           id_pred_enhanced    id predictions for enhanced data
%           id_pred_eg          id predictions (by subnetwork) for original
%           id_pred_eg_enhanced id predictions (by subnetwork) for enhanced
%
% example call:
%  run_analysis('hcp', 1, 10)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load data and set paths
addpath(genpath('../../../utils'))
save_path = './tmp';
if strcmp(dataset, 'hcp')
    % matrices with sex, iq data, and second session rest matrices for
    % figerprinting
    load('/home/dustinlab/Desktop/Matt/hcp_dat/hcp_rest_sex.mat')
    load('/home/dustinlab/Desktop/Matt/hcp_dat/hcp_iq.mat')
    load('/home/dustinlab/Desktop/Matt/hcp_dat/hcp_rest2_sex.mat')
    sex=sex+1;  % 1/2 instead of 0/1
    y = hcp_iq;
    X = mat_rest1;
    
    load('/data22/mri_group/dustinlab_data/dustinlab/Desktop/Matt/hcp_dat/all_mats_updated.mat');
    mats_id = all_mats;
    clear all_mats;
    
elseif strcmp(dataset, 'pnc')
    % for pnc
    load('/data22/mri_group/dustinlab_data/dustinlab/Desktop/Matt/pnc/pnc_rest_update.mat')
    y = all_behav;
    X = rest_mats;
elseif strcmp(dataset, 'abcd')
    load('/data22/mri_group/dustinlab_data/dustinlab/Desktop/Matt/abcd/abcd_rest_sex.mat')  
    y = iq;
    X = mat;
    clear mat
elseif strcmp(dataset, 'slim')
    load('/data22/mri_group/dustinlab_data/dustinlab/Desktop/Matt/SLIM/SLIM_mats.mat')
    y = behav;
    X = all_mats;
    nsub = size(X, 3);
end


%% make enhanced data, then compare to original

nsub = size(X, 3);

 % scale (mean abs value) of injected pattern
scale_factor = [0, 0.004, 0.007, 0.01, 0.014, 0.02, 0.03] ; 

for seed=seed_start:seed_end
    disp(['******************Seed ', num2str(seed), '****************************'])
    
    % for analysis only want final (largest) scale factor
    for scale_idx = length(scale_factor):length(scale_factor)
        rng(seed)
        % noise is proportional to each individuals iq score
        sub_scale_noise = zscore(y)*scale_factor(scale_idx);
        sub_scale_noise = repmat(sub_scale_noise', [35778, 1]);
        % sub_scale_noise = sub_scale_noise + .004*randn(size(sub_scale_noise));  % may want to add random noise here
        
        % random sign flips/edge selections
        % NOTE: you can make these edges not random to enforce a specific
        % neuroscientific interpretation (see pre-print)
        random_edges_to_change = rand(size(sub_scale_noise, 1), 1) - .5;
        % pick random 20% edges to alter
        random_edges_to_change(random_edges_to_change>=.4) = 1;
        random_edges_to_change(random_edges_to_change<=-.4) = -1;
        random_edges_to_change(abs(random_edges_to_change)<.4) = 0;
        
        % add or subtract noise for the 20% of edges
        sub_scale_noise(random_edges_to_change==1, :) = sub_scale_noise(random_edges_to_change==1, :);
        sub_scale_noise(random_edges_to_change==-1, :) = -1*sub_scale_noise(random_edges_to_change==-1, :);
        sub_scale_noise(random_edges_to_change==0, :) = 0;
        
        % make enhanced matrices
        noise_mat = zeros(size(X));
        for noise_idx = 1:nsub
            noise_mat(:, :, noise_idx) = edge2mat(sub_scale_noise(:, noise_idx));
        end
        mat_enhanced = X + noise_mat;
        
        % run analysis pipeline on enhanced data
        if strcmp(dataset, 'hcp')
            rng(seed);
            cv_idx = cv_indices(nsub, 10);  % kfold for fingerprinting
            [p_all, corr_strength, corr_assort, corr_cc, id_pred, id_pred_enhanced, id_pred_eg, id_pred_eg_enhanced] = analysis_pipeline(X, mat_enhanced, mats_id, cv_idx);
            
        else
            [p_all, corr_strength, corr_assort, corr_cc, id_pred, id_pred_enhanced, id_pred_eg, id_pred_eg_enhanced] = analysis_pipeline(X, mat_enhanced);
        end
        
        
        save([save_path, '/', dataset, '_analysis_seed', num2str(seed), '.mat'],...
            'p_all', 'corr_strength', 'corr_assort', 'corr_cc', 'id_pred', 'id_pred_enhanced', 'id_pred_eg', 'id_pred_eg_enhanced')
    end
end

