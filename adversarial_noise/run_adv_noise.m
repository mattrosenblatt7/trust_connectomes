function run_adv_noise(dataset, seed_start, seed_end)
% function for adding adversarial noise to connectomes
% example call: run_adv_noise('hcp', 1, 100)

% Make sure to update paths for loading and saving data

% Your data can be any functional connectivity matrices (size nnodes x
% nnodes x nsub) and some y varaible (size nsub x 1)

% The results will be cells of size # algorithms x # feature thresholds
% Each cell contains a structure with the following outputs:
% n_1_2: final adversarial noise for switching class 1 to 2
% max_abs_1_2: maximum absolute value noise recorded at each step
% of gradient descent
% mean_abs_1_2: mean absolute value of noise recorded at each step of
% gradient descent
% acc_1_2: accuracy of class 1 predictions recorded at each step of
% gradient descent
% corr_1_2: correlation between real and adversarial features (when using
% 100% of feeatures, this is the correlation between connectome edges)
% All the above outputsare also duplicated for switching class 2 to 1

%% loading in data
% please change your paths accordingly, and feel free to use other datasets
addpath(genpath('../utils'))
save_folder = 'my path to output folder';
algo_all = {'svm', 'lr'};  % specify algorithms to use
perfeat_all = [1];  % specify feature selection, this is 100% of features

if strcmp(dataset, 'abcd')
    load('my data path here')
    
    % train/test split (same split every time, will resample train data)
    rng(7);  % pick seed just to find very balanced split
    nsub = size(mat, 3);
    randorder=randperm(nsub);
    ntrain=round(.8*nsub);
    train_idx = randorder(1:ntrain);
    test_idx = randorder((ntrain+1):end);
    
    xrest_train = mat(:, :, train_idx);
    phenotype_train = sex(train_idx);
    xrest_test = mat(:, :, test_idx);
    phenotype_test = sex(test_idx);
    
    
    cross_dat = load('my first cross dataset prediction path');
    xcross = cross_dat.mat_rest1;
    ycross = cross_dat.sex + 1;
    
    cross_dat2 = load('my second cross dataset prediction path');
    xcross2 = cross_dat2.rest_mats;
    ycross2 = cross_dat2.sex;
    
elseif strcmp(dataset, 'hcp')
    load('my data path here')
    sex = sex+1;  % 1/2 instead of 0/1
    
    rng(6);  % pick seed just to find very balanced split
    nsub = size(mat_rest1, 3);
    randorder=randperm(nsub);
    ntrain=round(.8*nsub);
    train_idx = randorder(1:ntrain);
    test_idx = randorder((ntrain+1):end);
    
    % train/test split (same split every time, will resample train data)
    xrest_train = mat_rest1(:, :, train_idx);
    phenotype_train = sex(train_idx);
    xrest_test = mat_rest1(:, :, test_idx);
    phenotype_test = sex(test_idx);
    
    % load other datasets for cross-dataset predictions
    cross_dat = load('my first cross dataset prediction path');
    xcross = cross_dat.mat;
    ycross = cross_dat.sex;
    
    cross_dat2 = load('my second cross dataset prediction path');
    xcross2 = cross_dat2.rest_mats;
    ycross2 = cross_dat2.sex;
    
elseif strcmp(dataset, 'pnc')
    load('my data path here')
    
    nsub = size(rest_mats, 3);
    
    rng(2);  % pick seed just to find very balanced split
    randorder=randperm(nsub);
    ntrain=round(.8*nsub);
    train_idx = randorder(1:ntrain);
    test_idx = randorder((ntrain+1):end);
    
    % train/test split (same split every time, will resample train data)
    xrest_train = rest_mats(:, :, train_idx);
    phenotype_train = sex(train_idx);
    xrest_test = rest_mats(:, :, test_idx);
    phenotype_test = sex(test_idx);
    
    % load other datasets for cross-dataset predictions
    cross_dat = load('my first cross dataset prediction path');
    xcross = cross_dat.mat;
    ycross = cross_dat.sex;
    
    cross_dat2 = load('my second cross dataset prediction path');
    xcross2 = cross_dat2.mat_rest1;
    ycross2 = cross_dat2.sex + 1;
    
end

%% adversarial noise over specified seeds
for seed = seed_start:seed_end
    disp(['****************************', num2str(seed),'*****************************'])
    for algo_idx = 1:length(algo_all)
        
        
        for pf_idx = 1:length(perfeat_all)
            
            % bootstrap resample train data
            rng(seed);
            temp = 1:ntrain;
            boot_idx = datasample(temp, ntrain);
            x_train_resampled = xrest_train(:, :, boot_idx);
            phenotype_train_resampled = phenotype_train(boot_idx);
            
            % training the base model
            [mdl, feat_loc, dec_1]= cpm_classifier_train(x_train_resampled, phenotype_train_resampled,...
                seed, 'per_feat', perfeat_all(pf_idx),...
                'use_feat', 1, 'learner', algo_all{algo_idx});
            % finding adversarial noise
            [adv_noise, ~] = ...
                adv_noise(mdl, xrest_test, phenotype_test,...
                feat_loc, 'train', [], dec_1, algo_all{algo_idx});
            
            
            % cross prediction with adversarial noise: note this uses 2
            % other datasets for cross-prediction, but this may vary based
            % on your application (or you can exclude entirely)
            [~, cross_acc12] = ...
                adv_noise(mdl, xcross(:,:,ycross==1), ycross(ycross==1),...
                feat_loc, 'test', adv_noise.n_1_2, []);  
            
            [~, cross_acc21] = ...
                adv_noise(mdl, xcross(:,:,ycross==2), ycross(ycross==2),...
                feat_loc, 'test', adv_noise.n_2_1, []); 
            
            % second cross dataset
            [~, cross_acc12_2] = ...
                adv_noise(mdl, xcross2(:,:,ycross2==1), ycross2(ycross2==1),...
                feat_loc, 'test', adv_noise.n_1_2, []); 
            
            [~, cross_acc21_2] = ...
                adv_noise(mdl, xcross2(:,:,ycross2==2), ycross2(ycross2==2),...
                feat_loc, 'test', adv_noise.n_2_1, []);  
            
            adv_noise.cross_acc12 = cross_acc12;
            adv_noise.cross_acc21 = cross_acc21;
            adv_noise.cross_acc12_2 = cross_acc12_2;
            adv_noise.cross_acc21_2 = cross_acc21_2;
        end
        adv_noise.feat_loc = feat_loc;
        
        results{algo_idx, pf_idx} = adv_noise;
    end
end

save([save_folder, dataset,'_seed_', num2str(seed), '.mat'],...
    'results')
end