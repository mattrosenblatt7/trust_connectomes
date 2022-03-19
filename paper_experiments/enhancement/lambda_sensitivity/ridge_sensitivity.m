function [q_s, r_pearson, r_rank, y, coef_total, coef0_total, fit_lambda] = ridge_sensitivity(all_mats, all_behav, thresh, seed)
%ridgeCPM to assess sensitivity to regularization parameter lambda
%
%   [q_s, r_pearson, r_rank, y, coef_total, coef0_total, lambda_total] = ridgeCPM(all_mats, all_vectors, all_behav, thresh, v_alpha, lambda, k, seed)
%
%   Input:      all_mats,           connectome of all the subjects and tasks
%                                   [regions x regions x subjects x tasks]
%
%               all_vectors,         additional features for each subject
%                                   [n_features x subjects]
%
%               all_behav,          behavior of all the subjects
%                                   [subjects x 1]
%
%               thresh,             feature selection threshold (%
%                                   features)
%
%
%               v_alpha(optional),  value of the alpha parameter in elastic
%                                   net, default is 1e-6 which makes the
%                                   regression method to be ridge
%                                   regression, v_alpha=1 makes it lasso.
%
%               lambda(optional),   value of the lambda, if not provided,
%                                   cross-validation will be used
%
%               k(optional),        number of folds in k-fold cross
%                                   validation, default is 10
%
%               seed(optional),     random seed, default is 665
%
%   Output:     q_s,                cross-validated R^2 between predicted
%                                   value with ground truth (all_behav)
%
%               r_pearson,          WRONG! direct pearson correlation
%                                   between predicted value with ground
%                                   truth, only kept here for comparison
%                                   will be removed afterwards
%
%               r_rank,             cross-validated spearman correlation
%                                   between predicted value with ground
%                                   truth (all_behav)
%
%               y,                  predicted value for all the subjects
%
%               coef_total,         regression coefficients of all the edges
%                                   in all the k folds
%
%               coef0_total,        regression intercept in all the k folds
%
%               fit_lambda,         lambdas for which model was fit
%
%   Siyuan Gao, Yale University, 2018-2019
%   Adapted by Abby Greene, 2019
%   Further modified by Matt Rosenblatt, 2022


%% initialization
if nargin < 3
    error('not enough arguments, please check the help')
end

if ~exist('k', 'var')
    k = 10;
end

if ~exist('seed', 'var')
    rng('shuffle')
    tmp = randperm(10000000);
    seed=tmp(1);
end

if ~exist('v_alpha', 'var') || length(v_alpha) ~= 1
    v_alpha = 1e-6;
end

num_sub_total = size(all_mats, 3);
num_node = size(all_mats, 1);
num_task = size(all_mats, 4);


% assumes symmetric matrices
num_edge = num_node * (num_node - 1) / 2;


coef_total = zeros(num_edge*num_task, k); %store all the coefficients
coef0_total = zeros(1, k); % store all the intercept
lambda_total = zeros(1, k); % store all the lambda

%% convert connectivity to edge matrix (could made easier by squareform) - only if there's an all_mats input; if only vector, skip this
all_edges = mat2edge(all_mats);

%% main
lambda_gridsearch = exp(linspace(log(1e3), log(1e-3), 50)); num_search = length(lambda_gridsearch);
max_coef = zeros(k, num_search);
y = zeros(num_sub_total, num_search);
rng(seed);
indices = cv_indices(num_sub_total, k);

for i_fold = 1 : k
    fprintf('%dth fold\n', i_fold);
    
    test_idx = (indices==i_fold);
    train_idx = (indices~=i_fold);
    
    
    train_mats = all_edges(:, train_idx);
    test_mats = all_edges(:, test_idx);
    
    
    train_behav = all_behav;
    train_behav(test_idx) = [];
    
    % first step univariate edge selection
    [~, edge_p] = corr(train_mats', train_behav);
    edges_1 = find(edge_p<prctile(edge_p, 100*thresh));
    disp(['edge size = ' num2str(size(edges_1))]);
    
    % build model on TRAIN subs (with gridsearch)
    [fit_coef, fit_info] = lasso(train_mats(edges_1, :)', train_behav, 'Alpha',v_alpha, 'Lambda', lambda_gridsearch, 'CV', 10);
    idxLambda1SE = fit_info.Index1SE;
    coef = fit_coef(:,idxLambda1SE);
    coef0 = fit_info.Intercept(idxLambda1SE);
    lambda_total(i_fold) = fit_info.Lambda(idxLambda1SE);
    
    
    % run model on TEST sub with the best lambda parameter
    
    y(test_idx, :) = test_mats(edges_1, :)'*fit_coef + fit_info.Intercept;
    max_coef(i_fold, :) = max(abs(fit_coef), [], 1);
    
    coef_total(edges_1, i_fold) = coef;
    coef0_total(:, i_fold) = coef0;
    
    
    
end

% compare predicted and observed behaviors
[r_pearson, ~] = corr(y, all_behav);
[r_rank, ~] = corr(y, all_behav, 'type', 'spearman');
mse = sum((y - all_behav).^2) / num_sub_total;
q_s = 1 - mse / var(all_behav, 1);


fit_lambda = fit_info.Lambda;


end

