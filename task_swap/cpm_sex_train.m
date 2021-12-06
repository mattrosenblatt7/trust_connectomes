function [mdl, feat_loc] =...
    cpm_sex_train(x,sex, seed, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   REQUIRED INPUTS
%        x           Predictor variable (e.g., connectivity matrix)
%                    Allowed dimensions are 3D (node x node x nsubs x
%                    ntasks) OR 4D (node x node x nsubs x ntasks x nsubs)
%        y           Sex to be predicted, 1D vector
%        seed        seed to initialize random number generator
%
%   OPTIONAL INPUTS
%        'per_feat'  top percentage of features to be used (if use_feat==1)
%        'kfolds'    for k-fold cv, if ignoring site data
%        'use_feat'  0/1 for no/yes using feature selection
%        'use_site'  0/1 for no/yes using site data
%        'site'      site data (nsub vector), [] if not needed
%        'ens'       0/1 for no/yes to use model ensemble averaging
%   OUTPUTS
%        mdl_all     All models from all folds in cells
%        mdl         Average mdl coefficients over all folds
%        y_predict   Sex predictions
%        pred_acc    Accuracy of prediction
%        sens        Sensitivity of prediction
%        spec        Specificity of prediction
%        cmat        Confusion matrix
%        fold_info   Fold by fold accuracy, sens, spec
%   Adapted from rCPM script from Siyuan Gao

% example input:
%  [mdl_all, mdl, y_predict, pred_acc, sens, spec, cmat, fold_info...
%                         = cpm_svm(x, y, seed, 'per_feat', 0.1, 'kfolds', 21,...
%                         'use_feat', 0, 'use_site', 1,'site', sub_site,...
%                         'ens', 0);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Parse input
p=inputParser;
defaultper_feat=0.01;

addRequired(p,'x',@isnumeric);
addRequired(p,'sex',@isnumeric); % must be n x nsubs
addParameter(p,'per_feat',defaultper_feat,@isnumeric);
addParameter(p,'use_feat', 0, @isnumeric);
addParameter(p, 'learner', 'svm');


parse(p,x,sex,varargin{:});

per_feat = p.Results.per_feat;
use_feat = p.Results.use_feat;
learner = p.Results.learner;

clearvars p

%% Check for errors
numsub = size(x, 3);
numtask = size(x, 4);
numedge = size(x, 1)*(size(x, 2) - 1) / 2;
edges = zeros(numedge*numtask, numsub);
for idx = 1:numtask
    start = 1+(idx-1)*numedge;
    stop = start + numedge-1;
    [edges(start:stop, :),sex]=cpm_check_errors(x(:, :, :, idx),sex, 1);
end
x=edges;
disp(size(x))
%% Train Connectome-Based Predictive Model
[mdl, feat_loc]=...
    cpm_cv_svm(x,sex,per_feat, use_feat, seed, learner);

a=1;
end
