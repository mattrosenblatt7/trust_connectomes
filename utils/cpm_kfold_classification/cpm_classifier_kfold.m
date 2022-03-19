function [ypred] = cpm_classifier_kfold(X, y, k, per_feat, learner, optimize_hyperparams, seed)
% inputs:
% X (connectome) and y (phenotype) data
%
% seed: random seed
%
% per_feat: top % of features to use (e.g. 0.1 for 10%)
%
% learner: svm or lr (logistic regression) supported. typically better
% performance is seen with svm for connectomes
%
% NOTE: this is for binary classification only

% classes, often 0/1
y_class1 = min(y); y_class2 = max(y);

% Set seed and initialize predicted y variable
nsub=size(X,2);
rng(seed)
cv_idx = cv_indices(nsub, k);  % cross validation indices
ypred = zeros(nsub, 1);
for fold_idx = 1:k
    
    disp(['**** Fold ', num2str(fold_idx), ' ****']) 
    
    % get train and test
    train_idx = find(cv_idx~=fold_idx);
    test_idx = find(cv_idx==fold_idx);
    Xtrain = X(:, train_idx)';
    Xtest = X(:, test_idx)';
    ytrain = y(train_idx);
    
    % feature selection
    [~, p_all] = ttest2(Xtrain(ytrain==y_class1, :), Xtrain(ytrain==y_class2, :));
    feat_loc = find(p_all<=prctile(p_all, 100*per_feat));
    
    % search for regularization parameters (Note: you can typically remove
    % the optimization and see similar results in much shorter time)
    if optimize_hyperparams
        mdl = fitclinear(Xtrain(:, feat_loc), ytrain, 'Learner', learner, ...
            'Regularization', 'Ridge', 'OptimizeHyperparameters', {'Lambda'},...
            'HyperparameterOptimizationOptions', struct('Kfold', 5, 'ShowPlots', 0, 'Verbose', 0));
    else
        mdl = fitclinear(Xtrain(:, feat_loc), ytrain, 'Learner', learner);  
    end
    
    ypred(test_idx) = predict(mdl, Xtest(:, feat_loc));
end





