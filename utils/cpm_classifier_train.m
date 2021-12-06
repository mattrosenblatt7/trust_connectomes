function [mdl, feat_loc, dec_1] =...
    cpm_classifier_train(x,y, seed, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   REQUIRED INPUTS
%        x           Predictor variable (e.g., connectivity matrix)
%                    Allowed dimensions are 3D (node x node x nsubs x
%                    ntasks)
%        y           variable to be predicted, 1D vector, class 1 and 2
%        seed        seed to initialize random number generator
%
%   OPTIONAL INPUTS
%        'per_feat'  top percentage of features to be used (if use_feat==1)
%        'learner'    for k-fold cv, if ignoring site data
%        'use_feat'  0/1 for no/yes using feature selection
%   OUTPUTS
%        mdl         trained classifier
%        feat_loc    feature selection indices
%        dec_1       used in gradient descent for adversarial noise, sign
%                    decision function for class 1
%   Adapted from rCPM script from Siyuan Gao https://github.com/YaleMRRC/CPM

% example input:
% [mdl, feat_loc, dec_1]= cpm_classifier_train(x_train_resampled, phenotype_train_resampled,...
%     seed, 'per_feat', 1, 'use_feat', 1, 'learner', 'svm');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Parse input
p=inputParser;
defaultper_feat=0.01;

addRequired(p,'x',@isnumeric);
addRequired(p,'sex',@isnumeric); % must be n x nsubs
addParameter(p,'per_feat',defaultper_feat,@isnumeric);
addParameter(p,'use_feat', 0, @isnumeric);
addParameter(p, 'learner', 'svm');

parse(p,x,y,varargin{:});

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
    [edges(start:stop, :),y]=cpm_check_errors(x(:, :, :, idx),y, 1);
end
x=edges;
disp(size(x))
%% Train Connectome-Based Predictive Model
[mdl, feat_loc, dec_1]=...
    cpm_cv_svm(x,y,per_feat, use_feat, seed, learner);

end
