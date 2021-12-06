function enhancement_attack(seed_start, seed_end)
% code to inject "noise" to enhance predictions
% will need to modify data paths for use
% only inputs are start and end seeds (if just wanting a single seed,
% set seed_start and seed_end to the same number)

addpath(genpath('../utils'))
addpath(genpath('./cpm'))
addpath(genpath('./sex_classification'))

% matrices with sex, iq data, and second session rest matrices for
% figerprinting
load('/home/dustinlab/Desktop/Matt/hcp_dat/hcp_rest_sex.mat')
load('/home/dustinlab/Desktop/Matt/hcp_dat/hcp_iq.mat')
load('/home/dustinlab/Desktop/Matt/hcp_dat/hcp_rest2_sex.mat')
sex=sex+1;  % 1/2 instead of 0/1
nsub = size(mat_rest1, 3);

rng(6);  % pick seed just to find very balanced split
randorder=randperm(nsub);
ntrain=round(.8*nsub);
train_idx = randorder(1:ntrain);
test_idx = randorder((ntrain+1):end);

[x, y]=cpm_check_errors(mat_rest1, hcp_iq, 1);

xtrain=x(:,train_idx);
ytrain=y(train_idx);
xtest=x(:,test_idx);
ytest=y(test_idx);

% inversely proportional to scale of injected noise (will need to find
% these values for your application)
division_scale = [278, 147, 100, 71, 51] ;

% initialize variables
orig_corr = zeros(1, 100);
new_corr = zeros(length(division_scale), 100);
edge_corr = zeros(length(division_scale), nsub, 100);
acc_sex_altered = zeros(length(division_scale), 100);
id_acc_baseline = zeros(length(division_scale), 100);
id_acc_altered = zeros(length(division_scale), 100);
noise_abs_max = zeros(length(division_scale), 100);
noise_abs_mean = zeros(length(division_scale), 100);
perc_change_all = zeros(length(division_scale), 100);

% sex classifier - training the base model
mat_train = mat_rest1(:, :, train_idx);
sex_train = sex(train_idx);
[mdl, feat_loc, dec_1]= cpm_sex_train(mat_train,...
    sex_train,...
    1, 'per_feat', 1,...
    'use_feat', 1, 'learner', 'svm');
% baseline sex acc
[acc_sex_baseline] = ...
    sex_test(mdl, mat_rest1(:, :, test_idx), sex(test_idx), feat_loc);


for seed=seed_start:seed_end
    disp(['******************Seed ', num2str(seed), '****************************'])
     
    % original (unaltered) ridge regression
    [q_s, orig_corr(seed), r_rank, y] =...
        ridgeCPM_vector(mat_rest1, [],hcp_iq,...
        .1,  seed);

    % alter data, running over different scales
    for division_scale_idx = 1:length(division_scale)
        
        % noise is proportional to each individuals iq score
        sub_scale_noise = zscore(hcp_iq)/division_scale(division_scale_idx);
        sub_scale_noise = repmat(sub_scale_noise', [35778, 1]);
        sub_scale_noise = sub_scale_noise + .004*randn(size(sub_scale_noise));  % may want to add random noise here
        
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
        
        noise_mat = zeros(size(mat_rest1));
        for noise_idx = 1:nsub
            noise_mat(:, :, noise_idx) = edge2mat(sub_scale_noise(:, noise_idx));
        end
        
        mat_enhanced = mat_rest1 + noise_mat;
        
        % altered data ridge regression
        [q_s, new_corr(division_scale_idx, seed), r_rank, y] =...
            ridgeCPM_vector(mat_enhanced, [],hcp_iq,...
            .1, seed);
        
        % find edge correlation between original and enhanced
        % connectomes
        edges_original=mat2edge(mat_rest1);
        edges_adv=mat2edge(mat_enhanced);
        edge_corr(division_scale_idx, :, seed) = (diag(corr(edges_original, edges_adv)));
        
        % size of noise and a percentage change score
        noise_abs_mean(division_scale_idx, seed) = mean(abs(sub_scale_noise(sub_scale_noise~=0)));
        noise_abs_max(division_scale_idx, seed) = max(abs(sub_scale_noise(sub_scale_noise~=0)));
        perc_change_all(division_scale_idx, seed) = mean(sum(abs(mat2edge(noise_mat)))./sum(abs(mat2edge(mat_rest1))));
        
        % fingerprinting between rest1 (real or enhanced) and rest2
        % see Finn et al., 2015
        r1_baseline_edge = mat2edge(mat_rest1(:, :, test_idx));
        r1_enhanced_edge = mat2edge(mat_enhanced(:, :, test_idx));
        r2_edge = mat2edge(mat_rest2(:, :, test_idx));
        [~, id_pred] = max(corr(r1_baseline_edge, r2_edge));
        [~, id_pred_enhanced] = max(corr(r1_enhanced_edge, r2_edge));
        id_acc_baseline(division_scale_idx, seed) = mean(id_pred == (1:length(test_idx)));
        id_acc_altered(division_scale_idx, seed) = mean(id_pred_enhanced == (1:length(test_idx)));
        
        
        %****sex classification on connectomes enhanced for IQ prediction****
        mat_train = mat_rest1(:, :, train_idx);
        sex_train = sex(train_idx);
        [mdl, feat_loc, dec_1]= cpm_sex_train(mat_enhanced(:, :, train_idx),...
            sex_train,...
            seed, 'per_feat', 1,...
            'use_feat', 1, 'learner', 'svm');
        [acc_sex_altered(division_scale_idx, seed)] = ...
            sex_test(mdl, mat_enhanced(:, :, test_idx), sex(test_idx), feat_loc);
        
        
        
    end
    
    save(['my save path' ,num2str(seed), '.mat'],...
        'orig_corr', 'new_corr', 'edge_corr', 'id_acc_baseline', 'id_acc_altered',...
        'noise_abs_mean', 'noise_abs_max', 'acc_sex_baseline', 'acc_sex_altered', 'perc_change_all')
end

