function [adv_noise, acc] =...
    adv_noise(mdl, x,y, feat_loc, traintest, adv_noise, dec_1, learner)
% train to add or test application of adversarial noise
% called from run_adv_noise.m

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   REQUIRED INPUTS
%        mdl         trained classifier
%        x           Predictor variable (e.g., connectivity matrix)
%                    Allowed dimensions are 3D (node x node x nsubs x
%                    ntasks)
%        y           variable to be predicted, 1D vector, class 1 and 2
%        feat_loc    feature selection indices
%        traintest   used for determining whether to learn the
%                    adversarial noise ('train') or apply previously 
%                    learned adversarial noise to data ('test')
%        adv_noise   learned adversarial noise pattern, blank if training
%        dec_1       sign of decision function for class 1 of data
%
%   OUTPUTS
%        mdl         trained classifier
%        feat_loc    feature selection indices
%        dec_1       used in gradient descent for adversarial noise, sign
%                    decision function for class 1
%   Adapted from rCPM script from Siyuan Gao https://github.com/YaleMRRC/CPM

% example input:
%  [adv_noise, ~] = adv_noise(mdl, xrest_test, phenotype_test,...
%                                    feat_loc, 'train', [], dec_1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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

if strcmp(traintest, 'train')
    acc = [];
    if ~strcmp(learner, 'ksvm')
    [adv_noise]=...
        generate_adv_noise(mdl, x,y,feat_loc, dec_1);
elseif strcmp(traintest, 'test')
    xfeat = x(feat_loc, :);
    acc.baseline = mean(predict(mdl, xfeat')==y(1));
    acc.adv = mean(predict(mdl, xfeat'+adv_noise)==y(1));
end

end
