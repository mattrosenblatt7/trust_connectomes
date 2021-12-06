function analysis_pipeline(dataset, type, algo_idx)


% brain connectivity toolbox https://sites.google.com/site/bctnet/
addpath(genpath('./BCT'))

thresh = .25;  % for matrix binarization
discriminator_kfolds = 10;
plot_results = 0;
algo_all = {'svm', 'lr'};
% load in data: important to use the same train/test split that you used
% when creating the adversarial noise.
% IMPORTANT: assign variable "x" as test matrices and variable
% "phenotype_test" as test phenotypes. Set "results_path" as path to load
% your adv noise results
if strcmp(dataset, 'abcd')
    load('my load path')  % load matrices
    
    % train/test split (same split every time, will resample train data)
    rng(7);  % pick seed just to find very balanced split
    nsub = size(mat, 3);
    randorder=randperm(nsub);
    ntrain=round(.8*nsub);
    train_idx = randorder(1:ntrain);
    test_idx = randorder((ntrain+1):end);
    
    xrest_train = mat(:, :, train_idx);
    phenotype_train = y(train_idx);
    xrest_test = mat(:, :, test_idx);
    
    phenotype_test = y(test_idx);
    x=xrest_test;
    num_iters = 100;
    
elseif strcmp(dataset, 'hcp')
    load('my load path')  % load matrices
    y = y+1;  % 1/2 instead of 0/1
    
    nsub = size(mat_rest1, 3);
    
    rng(6);  % pick seed just to find very balanced split
    randorder=randperm(nsub);
    ntrain=round(.8*nsub);
    train_idx = randorder(1:ntrain);
    test_idx = randorder((ntrain+1):end);
    
    % train/test split (same split every time, will resample train data)
    xrest_train = mat_rest1(:, :, train_idx);
    phenotype_train = y(train_idx);
    xrest_test = mat_rest1(:, :, test_idx);
    
    phenotype_test = y(test_idx);
    x=xrest_test;
    num_iters = 100;
elseif strcmp(dataset, 'pnc')
    load('my load path')  % load matrices
    
    nsub = size(rest_mats, 3);
    
    rng(2);  % pick seed just to find very balanced split
    randorder=randperm(nsub);
    ntrain=round(.8*nsub);
    train_idx = randorder(1:ntrain);
    test_idx = randorder((ntrain+1):end);
    
    % train/test split (same split every time, will resample train data)
    xrest_train = rest_mats(:, :, train_idx);
    phenotype_train = y(train_idx);
    xrest_test = rest_mats(:, :, test_idx);
    phenotype_test = y(test_idx);
    x=xrest_test;
    num_iters = 100;
    
end

% convert from matrix to vector of edges
numsub = size(x, 3);
numtask = size(x, 4);
numedge = size(x, 1)*(size(x, 2) - 1) / 2;
numnode = size(x, 1);
edges = zeros(numedge*numtask, numsub);
for idx = 1:numtask
    start = 1+(idx-1)*numedge;
    stop = start + numedge-1;
    [edges(start:stop, :),~]=cpm_check_errors(x(:, :, :, idx),phenotype_test, 1);
end
x=edges;
disp(size(x))
nsub = size(x, 2);

% initialize variables
acc1=zeros(num_iters, 1); acc2=zeros(num_iters, 1); accboth=zeros(num_iters, 1);
p_mean=zeros(num_iters, 1); p_min=zeros(num_iters, 1); edge_corr=zeros(num_iters, 1);
deg_d_pos=zeros(num_iters, 1); dens_d_pos=zeros(num_iters, 1);
cc_d_pos=zeros(num_iters, 1); eff_d_pos=zeros(num_iters, 1); assort_d_pos=zeros(num_iters, 1);
xadv_all = zeros(size(x));  % initialize adversarial edges
% loop over all seeds
for seed_idx = 1:num_iters
    % display progress
    if mod(seed_idx, 10)==0
        disp(seed_idx)
    end
    
    % load in adversarial results (update your path here)
    if ~strcmp(dataset, 'pnc')
        if strcmp(type, 'linear')
            load(['/home/dustinlab/Desktop/Matt/trust/adversarial_noise/corr_adv/results_adv_full_feat/',...
                dataset, '_seed_', num2str(seed_idx), '.mat'])
        elseif strcmp(type, 'kernel')
            load(['/home/dustinlab/Desktop/Matt/trust/adversarial_noise/corr_adv/kernel_svm/results_adv_kernel/',...
                dataset, '_seed_',num2str(seed_idx), '.mat'])
        end
    elseif strcmp(dataset, 'pnc')
        if strcmp(type, 'linear')
            load(['/home/dustinlab/Desktop/Matt/trust/adversarial_noise/corr_adv/results_adv_full_feat_10_5/',...
                dataset, '_seed_', num2str(seed_idx), '.mat'])
        elseif strcmp(type, 'kernel')
            load(['/home/dustinlab/Desktop/Matt/trust/adversarial_noise/corr_adv/kernel_svm/results_ksvm_10_5/',...
                dataset, '_seed_',num2str(seed_idx), '.mat'])
        end
    end
    
      
    % make adversarial connectomes
    xadv_all = zeros(size(x));
    
    % class 1
    xadv_all(results{algo_idx}.feat_loc, phenotype_test==1) =...
        x(results{algo_idx}.feat_loc, phenotype_test==1) +...
        results{algo_idx}.n_1_2';
    
    % class 2
    xadv_all(results{algo_idx}.feat_loc, phenotype_test==2) =...
        x(results{algo_idx}.feat_loc, phenotype_test==2) +...
        results{algo_idx}.n_2_1';
    
    % ks test
    for idx_sub = 1:nsub
        % ks test
        [h, p_all(idx_sub)] = kstest2(x(:, idx_sub), xadv_all(:, idx_sub)); 
    end
    % mean and min p values
    p_mean(seed_idx) = mean(p_all);
    p_min(seed_idx) = min(p_all);
    
    % correlations between edges
    edge_corr(seed_idx) = mean(diag(corr(x, xadv_all)));
    
    % **** Discriminator ****
    % discriminator for class 1
    class1_loc = find(phenotype_test==1);
    x_1 = x(:, class1_loc);
    x_1_adv = xadv_all(:, class1_loc);
    x_combined_1(:, :, 1) = x_1; x_combined_1(:, :, 2) = x_1_adv;
    class1_fake_real = [zeros(length(class1_loc), 1), ones(length(class1_loc), 1)];  % 0 for real, 1 for fake
    acc1(seed_idx) =...
        discriminator([x_combined_1], class1_fake_real, discriminator_kfolds);
    
    % discriminator for class 2
    class2_loc = find(phenotype_test==2);
    x_2 = x(:, class2_loc);
    x_2_adv = xadv_all(:, class2_loc);
    x_combined_2(:, :, 1) = x_2; x_combined_2(:, :, 2) = x_2_adv;
    class2_fake_real = [zeros(length(class2_loc), 1), ones(length(class2_loc), 1)];  % 0 for real, 1 for fake
    acc2(seed_idx)=...
        discriminator(x_combined_2, class2_fake_real, discriminator_kfolds);
    
    % discriminator for all classes
    x_combined(:, :, 1) = x;
    x_combined(:, :, 2) = xadv_all;
    class_fake_real = [zeros(size(x, 2), 1), ones(size(x, 2), 1)];
    accboth(seed_idx) = discriminator(x_combined, class_fake_real, 10);
    
    % **** Graph metrics ****
    % convert matrices
    xmat = zeros(numnode, numnode, nsub);
    for idx = 1:size(x, 2)
        xmat(:, :, idx) = edge2mat(x(:, idx));
    end
    xmat_adv = zeros(numnode, numnode, nsub);
    for idx = 1:size(x, 2)
        xmat_adv(:, :, idx) = edge2mat(xadv_all(:, idx));
    end
    
    % **** Positive binarized matrices ****
    % binarize matrices
    xmat_bin_pos = xmat; xmat_bin_pos((xmat_bin_pos)<thresh)=0; xmat_bin_pos((xmat_bin_pos)>=thresh)=1;
    xmat_adv_bin_pos = xmat_adv; xmat_adv_bin_pos((xmat_adv_bin_pos)<thresh)=0; xmat_adv_bin_pos((xmat_adv_bin_pos)>=thresh)=1;
    
    % degree
    deg = zeros(numnode, nsub); deg_adv = zeros(numnode, nsub);
    for idx = 1:nsub
        deg(:, idx) = degrees_und(xmat_bin_pos(:, :, idx));
        deg_adv(:, idx) = degrees_und(xmat_adv_bin_pos(:, :, idx));
    end
    sdpooled_allnodes = sqrt(0.5*(var(deg, [], 2) + var(deg_adv, [], 2)));
    deg_d_pos(seed_idx) = mean(abs(mean(deg-deg_adv, 2)) ./ sdpooled_allnodes);
    
    % density
    % loop over subjects
    dens = zeros(nsub, 1); dens_adv = zeros(nsub, 1);
    for idx = 1:nsub
        dens(idx) = squeeze(density_und(xmat_bin_pos(:, :, idx)));
        dens_adv(idx) = squeeze(density_und(xmat_adv_bin_pos(:, :, idx)));
    end
    sdpooled = sqrt(0.5*(var(dens)+var(dens_adv)));
    dens_d_pos(seed_idx) = abs(mean(dens-dens_adv))/sdpooled;
    
    % clustering coefficienct
    % loop over subjects
    cc = zeros(numnode, nsub);
    cc_adv = zeros(numnode, nsub);
    for idx = 1:nsub
        cc(:, idx) = squeeze(clustering_coef_bu(xmat_bin_pos(:, :, idx)));
        cc_adv(:, idx) = squeeze(clustering_coef_bu(xmat_adv_bin_pos(:, :, idx)));
    end
    sdpooled_allnodes = sqrt(0.5*(var(cc, [], 2) + var(cc_adv, [], 2)));
    cc_d_pos(seed_idx) = mean(abs(mean(cc-cc_adv, 2)) ./ sdpooled_allnodes);
    
    
    % efficiency
    % loop over subjects
    eff = zeros(nsub, 1); eff_adv = zeros(nsub, 1);
    for idx = 1:nsub
        eff(idx) = squeeze(efficiency_bin(xmat_bin_pos(:, :, idx)));
        eff_adv(idx) = squeeze(efficiency_bin(xmat_adv_bin_pos(:, :, idx)));
    end
    sdpooled = sqrt(0.5*(var(eff) + var(eff_adv)));
    eff_d_pos(seed_idx) = abs(mean(eff-eff_adv)) / sdpooled;
    
    % assortativity
    assort = zeros(nsub, 1); assort_adv = zeros(nsub, 1);
    for idx = 1:nsub
        assort(idx) = squeeze(assortativity_bin(xmat_bin_pos(:, :, idx), 0));
        assort_adv(idx) = squeeze(assortativity_bin(xmat_adv_bin_pos(:, :, idx), 0));
    end
    sdpooled = sqrt(0.5*(var(assort) + var(assort_adv)));
    assort_d_pos(seed_idx) = abs(mean(assort-assort_adv)) / sdpooled;
    

end

% save
clear results
results.acc1=acc1; results.acc2=acc2; results.accboth=accboth;
results.edge_corr=edge_corr; results.p_mean=p_mean;
results.p_min=p_min;
results.deg_d_pos=deg_d_pos;
results.dens_d_pos=dens_d_pos; results.cc_d_pos=cc_d_pos;
results.eff_d_pos=eff_d_pos; results.assort_d_pos=assort_d_pos;
save(['./adv_metrics/results_', dataset, '_', type, '_', algo_all{algo_idx}, '.mat'], 'results')










