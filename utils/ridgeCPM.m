function [q_s, r_pearson, r_rank, y, coef_total, coef0_total] = ridgeCPM(all_mats, all_behav, varargin)
%ridgeCPM Connectome-based predictive modeling using univariate
%feature selection and ridge regression
%
%   [q_s, r_pearson, r_rank, y, coef_total, coef0_total, lambda_total] = ridgeCPM(all_mats, all_vectors, all_behav, thresh, v_alpha, lambda, k, seed)
%
%   Input:      all_mats,           connectome of all the subjects and tasks
%                                   [regions x regions x subjects x tasks]
%
%              
%               all_behav,          behavior of all the subjects
%                                   [subjects x 1]
%
%               thresh (optional),  feature selection threshold (%
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
%               lambda_total,       penalty parameter chosen at each
%                                   iteration
%
%   Siyuan Gao, Yale University, 2018-2019 
%   https://www.sciencedirect.com/science/article/pii/S1053811919306196?via%3Dihub#bib2
%   Adapted by Abby Greene, 2019
%   Adapted further by Matt Rosenblatt, 2022
%% parse inputs
p=inputParser;
addRequired(p,'all_mats',@isnumeric);
addRequired(p,'all_behav',@isnumeric);
addParameter(p,'per_feat',0.1,@isnumeric);
addParameter(p,'v_alpha',1e-6,@isnumeric);
addParameter(p,'lambda',NaN,@isnumeric);
addParameter(p,'kfolds',10,@isnumeric);
addParameter(p,'seed',1,@isnumeric);

parse(p, all_mats, all_behav, varargin{:});

per_feat = p.Results.per_feat;
v_alpha = p.Results.v_alpha;
lambda = p.Results.lambda;
kfolds = p.Results.kfolds;
seed = p.Results.seed;

%% initialization

num_sub_total = size(all_mats, 3);
num_node = size(all_mats, 1);
num_task = size(all_mats, 4);

is_sym = issymmetric(all_mats(:, :, 1, 1));
% assumes symmetric matrices
num_edge = num_node * (num_node - 1) / 2;

coef_total = zeros(num_edge*num_task, kfolds); %store all the coefficients
coef0_total = zeros(1, kfolds); % store all the intercept
lambda_total = zeros(1, kfolds); % store all the lambda

%% convert connectivity to edge matrix (could made easier by squareform) - only if there's an all_mats input; if only vector, skip this
all_edges = mat2edge(all_mats);

%% main
y = zeros(num_sub_total, 1);
rng(seed);
indices = cv_indices(num_sub_total, kfolds);
%     indices = kfold_family(famid,k,seed); % Added to run kfold_family idx generation (ie assign subjects to folds by family membership) on the fly
change_all=1:10;  % points to change prediction
corr_all = zeros(num_sub_total, length(change_all));
xadv_all = zeros(size(all_edges, 1), size(all_edges, 2), length(change_all));
for i_fold = 1 : kfolds
    fprintf('%dth fold\n', i_fold);
    
    test_idx = (indices==i_fold);
    train_idx = (indices~=i_fold);
    
    
    train_mats = all_edges(:, train_idx);
    test_mats = all_edges(:, test_idx);
    
    
    train_behav = all_behav;
    train_behav(test_idx) = [];
    
    % first step univariate edge selection
    [~, edge_p] = corr(train_mats', train_behav);
    edges_1 = find(edge_p<prctile(edge_p, 100*per_feat));
    disp(['edge size = ' num2str(size(edges_1))]);
    
    % build model on TRAIN subs
    
    if isnan(lambda)
        [fit_coef, fit_info] = lasso(train_mats(edges_1, :)', train_behav, 'Alpha',v_alpha, 'CV', 10);
        idxLambda1SE = fit_info.Index1SE;
        coef = fit_coef(:,idxLambda1SE);
        coef0 = fit_info.Intercept(idxLambda1SE);
        lambda_total(i_fold) = fit_info.Lambda(idxLambda1SE);
    elseif length(lambda)==1
        [coef, fit_info] = lasso(train_mats(edges_1, :)', train_behav, 'Alpha',v_alpha, 'Lambda', lambda);
        coef0 = fit_info.Intercept;
    elseif length(lambda)>1
        [fit_coef, fit_info] = lasso(train_mats(edges_1, :)', train_behav, 'Alpha',v_alpha, 'CV', 10, 'Lambda', lambda);
        idxLambda1SE = fit_info.Index1SE;
        coef = fit_coef(:,idxLambda1SE);
        coef0 = fit_info.Intercept(idxLambda1SE);
        lambda_total(i_fold) = fit_info.Lambda(idxLambda1SE);
    end
    
    % run model on TEST sub with the best lambda parameter  
    y(test_idx) = test_mats(edges_1, :)'*coef+coef0;  
    coef_total(edges_1, i_fold) = coef;
    coef0_total(:, i_fold) = coef0;
    
    
    
end

% compare predicted and observed behaviors
[r_pearson, ~] = corr(y, all_behav);
[r_rank, ~] = corr(y, all_behav, 'type', 'spearman');
mse = sum((y - all_behav).^2) / num_sub_total;
q_s = 1 - mse / var(all_behav, 1);


end

