function run_enhancement(dataset, seed_start, seed_end)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Enhancement attacks on rCPM models of IQ
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
%           all_corr        correlations between predicted and meaured
%                           behavior (Pearson's r)
%           all_q           prediction R^2
%           all_r_rank   	Spearman's rank
%           edge_corr       correlations (Pearson's) between original and
%                           enhanced connectome edges
%           noise_abs_mean  mean absolute change of edges that were changed
%           noise_abs_max   max absolute change of edges that were changed
%           perc_change_all calculation of how large enhancement pattern is
%                           relative to entire connectome
%           all_sex_acc     self-reported sex classification accuracy for
%                           connectomes that were enhnaced for another
%                           prediction
%
% example call:
%  run_enhancement('hcp', 1, 10)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load data and set save path
addpath(genpath('../../utils'))
addpath(genpath('../../cpm_kfold_classification'))
save_path = './tmp';

% to load data for your own dataset, you should have a .mat file that loads
% variables into X (# nodes x # nodes x # participants) matrix, y
% (behavioral variable), and sex
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

%% initialize some parameters and run enhancement
nsub = size(X, 3);
scale_factor = [0, 0.004, 0.007, 0.01, 0.014, 0.02, 0.03];

% initialize variables
all_corr = zeros(length(scale_factor), 1);
all_q = zeros(length(scale_factor), 1);
all_r_rank = zeros(length(scale_factor), 1);
edge_corr = zeros(length(scale_factor), nsub);
all_sex_acc = zeros(length(scale_factor), 1);
noise_abs_max = zeros(length(scale_factor), 1);
noise_abs_mean = zeros(length(scale_factor), 1);
perc_change_all = zeros(length(scale_factor), 1);
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
        
       [all_q(scale_idx), all_corr(scale_idx), all_r_rank(scale_idx), ypred, coef_total, coef0_total] =...
            ridgeCPM(mat_enhanced, y, 'per_feat', 0.1, 'v_alpha', 1e-6, 'kfolds', 10, 'seed', seed);

        % find edge correlation between original and enhanced
        % connectomes
        edges_original=mat2edge(X);
        edges_adv=mat2edge(mat_enhanced);
        edge_corr(scale_idx, :) = (diag(corr(edges_original, edges_adv)));
        
        % size of noise and a percentage change score
        if scale_factor(scale_idx)>0
            noise_abs_mean(scale_idx) = mean(abs(sub_scale_noise(sub_scale_noise~=0)));
            noise_abs_max(scale_idx) = max(abs(sub_scale_noise(sub_scale_noise~=0)));
            perc_change_all(scale_idx) = mean(sum(abs(mat2edge(noise_mat)))./sum(abs(mat2edge(X))));
        else
            noise_abs_mean(scale_idx) = 0;
            noise_abs_max(scale_idx) = 0;
            perc_change_all(scale_idx) = 0;
        end

        %****sex classification on connectomes enhanced for IQ prediction****
        % sex classification baseline
        [~, all_sex_acc(scale_idx)] = cpm_classifier_main(mat_enhanced, sex, 'per_feat', .1,...
            'kfolds', 10, 'learner', 'svm', 'optimize_hyperparams', 1, 'seed', seed);
              
    end
    
    save([save_path, '/',dataset, '_enhancement_' ,num2str(seed), '.mat'],...
        'all_corr','all_q', 'all_r_rank', 'edge_corr',...
        'noise_abs_mean', 'noise_abs_max', 'all_sex_acc', 'perc_change_all')
end

