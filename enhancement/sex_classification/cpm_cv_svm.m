function [mdl, feat_loc, dec_1]=...
    cpm_cv_svm(x,y,per_feat, use_feat, seed, learner)
% inputs:
% x (connectome) and y (phenotype) data
% per_feat: top % of features to use (if applicable)
% use_feat 0/1, whether to use feature selection

% Set seed and initialize predicted y variable
nsubs=size(x,2);
rng(seed)
y_predict = zeros(nsubs, 1);


% feature selection, if specified by use_feat
if use_feat
    [h, p_all] = ttest2(x(:, y==1)', x(:, y==2)');
    
    [p_sorted, sort_idx] = sort(p_all);
    num_feat = round(per_feat*length(sort_idx));
    feat_loc = sort_idx(1:num_feat);  % top percent of features
    x = x(feat_loc, :);
end

% fit model
if strcmp(learner, 'svm')
    mdl = fitclinear(x',y, 'Learner', 'svm', 'Lambda', 0);  % can manually set regularization Lambda
    % mdl = fitcsvm(x_train',y_train, 'KernelFunction','linear');
elseif strcmp(learner, 'lr')
    mdl = fitclinear(x',y, 'Learner', 'logistic', 'Lambda', 0);
    
end


df = mdl.Beta'*x + mdl.Bias;  % decision function
dec_1 = sign(mean(df(y==1)));



