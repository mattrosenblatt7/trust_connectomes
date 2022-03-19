function lambda_sensitivity(dataset, seed_start, seed_end)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Evaluation of sensitivity of enhancement to regularization parameter
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
%           all_corr            correlations between predicted and meaured
%                               behavior (Pearson's r) for each lambda
%           all_q               prediction R^2 for each lambda
%           all_r_rank          Spearman's rank for each lambda
%           lambda_gridsearch   all lambdas searched over
%
% example call:
%  lambda_sensitivity('hcp', 1, 10)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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
    
elseif strcmp(dataset, 'pnc')
    % for pnc
    load('/data22/mri_group/dustinlab_data/dustinlab/Desktop/Matt/pnc/pnc_rest_update.mat')
    y = all_behav;
    X = rest_mats;
    nsub = size(X, 3);
elseif strcmp(dataset, 'abcd')
    load('/data22/mri_group/dustinlab_data/dustinlab/Desktop/Matt/abcd/abcd_rest_sex.mat')
    
    y = iq;
    X = mat;
    
    clear mat
elseif strcmp(dataset, 'slim')
    load('/data22/mri_group/dustinlab_data/dustinlab/Desktop/Matt/SLIM/SLIM_mats.mat')
    y = behav;
    X = all_mats;
end


nsub = size(X, 3);
scale_factor = [0.01];


for seed=seed_start:seed_end
    disp(['******************Seed ', num2str(seed), '****************************'])
    
    % alter data, running over different scales
    % scale 0 is original pred
    for scale_idx = 1:length(scale_factor)
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
        
        noise_mat = zeros(size(X));
        for noise_idx = 1:nsub
            noise_mat(:, :, noise_idx) = edge2mat(sub_scale_noise(:, noise_idx));
        end
        
        mat_enhanced = X + noise_mat;
        
        [all_q, all_corr, all_r_rank, ypred, coef_total, coef0_total, lambda_gridsearch] =...
            ridge_sensitivity(mat_enhanced, y, .1, seed);
        
        
    end
    
    save([save_path, '/',dataset, '_enhancement_' ,num2str(seed), '.mat'],...
        'all_corr','all_q', 'all_r_rank', 'lambda_gridsearch')
end

