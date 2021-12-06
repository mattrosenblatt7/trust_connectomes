function [mdl, feat_loc, dec_1]=...
    cpm_cv_svm(x,y,per_feat, use_feat, seed, learner)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   REQUIRED INPUTS
%        x           Predictor variable (e.g., connectivity matrix)
%                    Allowed dimensions are 3D (node x node x nsubs x
%                    ntasks)
%        y           variable to be predicted, 1D vector, class 1 or 2
%        per_feat    top % of features to use (if applicable)
%        use_feat    0/1, whether to use feature selection
%        seed        seed to initialize random number generator
%        learner     'svm' or 'lr'
%
%   OUTPUTS
%        mdl         trained classifier
%        feat_loc    feature selection indices
%        dec_1       used in gradient descent for adversarial noise, sign
%                    decision function for class 1
%   Adapted from rCPM script from Siyuan Gao https://github.com/YaleMRRC/CPM

% example input:
% [mdl, feat_loc, dec_1]= cpm_cv_svm(x,y,1, 1, 44, 'svm');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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
elseif strcmp(learner, 'lr')
    mdl = fitclinear(x',y, 'Learner', 'logistic', 'Lambda', 0);
elseif strcmp(learner, 'ksvm')
    % note: selected these parameters with 10-fold cv on train data
    mdl = fitcsvm(x',y, 'KernelFunction','polynomial', 'BoxConstraint', 999.0043,...
        'KernelScale', 78.6810);
end


df = mdl.Beta'*x + mdl.Bias;  % decision function

% sign of decision function for first class (assumes that at least 50% of
% train data can be classifier correctly)
dec_1 = sign(mean(df(y==1)));



