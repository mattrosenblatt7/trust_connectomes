function run_trust_taskswap(dataset, seed_start, seed_end)
% example: run_trust_taskswap('hcp', 1, 10)
addpath(genpath('../utils'))
% change percentage features and model type
perfeat=1;
algo_all = {'svm', 'lr', 'ksvm'};

%% loading data

% this path contains all_mats variable with nnode x nnode x nsub x ntask
% matrix and a y variable with classes 1 and 2
load('my path')

nsub=size(all_mats, 3);
rng(6);  % pick seed just to find very balanced split
randorder=randperm(nsub);
ntrain=round(.8*nsub); 
train_idx = randorder(1:ntrain);
test_idx = randorder((ntrain+1):end);

% train/test split (same split every time, will resample train data)
xrest_train = all_mats(:, :, train_idx, 2);
sex_train = y(train_idx);
sex_test = y(test_idx);


%%
for seed = seed_start:seed_end
    disp(['****************************', num2str(seed),'*****************************'])

    tasks_all = 1:size(all_mats, 4);  % in hcp, 9 scans
    acc_all = zeros(length(algo_all), length(tasks_all), 3);
   

    for algo_idx = 1:length(algo_all)
        % bootstrap train data
        rng(seed);
        temp = 1:ntrain;
        boot_idx = datasample(temp, ntrain);
        x_train_resampled = xrest_train(:, :, boot_idx);
        sex_train_resampled = sex_train(boot_idx);
        
        % train model with resting state scan
        [mdl, feat_loc, ~]= cpm_classifier_train(x_train_resampled, sex_train_resampled,...
            seed, 'per_feat', perfeat,...
            'use_feat', 1, 'learner', algo_all{algo_idx});
        
        % test model with each task scan
        for task_idx = 1:length(tasks_all)
            
            which_task = tasks_all(task_idx);
            x_test = all_mats(:, :, test_idx, which_task);
            [acc, acc_class1, acc_class2] =...
                test_task(mdl, x_test,sex_test,...
                feat_loc);
            acc_all(algo_idx, task_idx, 1) = acc;
            acc_all(algo_idx, task_idx, 2) = acc_class1;
            acc_all(algo_idx, task_idx, 3) = acc_class2;
        end
    end
    
    save(['./task_swap_8_26/', dataset,'_seed_', num2str(seed), '_ksvm.mat'],...
        'acc_all')
end