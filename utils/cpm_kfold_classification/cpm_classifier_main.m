function [ypred, acc] =...
    cpm_classifier_main(X,y, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   REQUIRED INPUTS
%        X           Predictor variable (e.g., connectivity matrix)
%                    Allowed dimensions are 3D (node x node x nsubs x
%                    ntasks) OR 4D (node x node x nsubs x ntasks x nsubs)
%        y           Sex to be predicted, 1D vector

%
%   OPTIONAL INPUTS
%        'per_feat'             top percentage of features to be used (e.g., 0.1 for 10%)
%        'kfolds'               number of cross-validation folds
%        'learner'              'svm' or 'lr'
%        'optimize_hyperparams' whether to optimize model regularization parameters, 0/1 
%        'seed'                 seed to initialize random number generator
%
%   OUTPUTS
%        ypred       Predicted y values
%        acc         Model accuracy
%   Adapted from rCPM script from Siyuan Gao https://github.com/YaleMRRC

% example input:
%  [ypred, acc] = cpm_classifier_main(all_mats, classes, 'per_feat', 0.1, 'kfolds', 10, 'learner', 'svm', 'optimize_hyperparams', 1, 'seed', 77)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Parse input
p=inputParser;
defaultper_feat=0.1;
addRequired(p,'X',@isnumeric);
addRequired(p,'y',@isnumeric);
addParameter(p,'per_feat',defaultper_feat,@isnumeric);
addParameter(p,'kfolds',10,@isnumeric);
addParameter(p, 'learner', 'svm');
addParameter(p,'optimize_hyperparams',0,@isnumeric);
addParameter(p,'seed',1,@isnumeric);

parse(p,X,y,varargin{:});

per_feat = p.Results.per_feat;
k = p.Results.kfolds;
learner = p.Results.learner;
optimize_hyperparams = p.Results.optimize_hyperparams;
seed = p.Results.seed;

clearvars p

%% Check for errors
numsub = size(X, 3);
numtask = size(X, 4);
numedge = size(X, 1)*(size(X, 2) - 1) / 2;
edges = zeros(numedge*numtask, numsub);
for idx = 1:numtask
    start = 1+(idx-1)*numedge;
    stop = start + numedge-1;
    [edges(start:stop, :),y]=cpm_check_errors(X(:, :, :, idx),y, 1);
end
X=edges;
disp(size(X))
%% Train Connectome-Based Predictive Model
ypred = cpm_classifier_kfold(X, y, k, per_feat, learner, optimize_hyperparams, seed);
acc = mean(ypred==y);
