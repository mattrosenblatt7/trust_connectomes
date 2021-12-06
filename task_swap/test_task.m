function [acc, acc_class1, acc_class2] =...
    test_task(mdl, x,y, feat_loc)
% train to add adversarial noise
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   REQUIRED INPUTS
%        mdl         trained classifier
%        x           Predictor variable (e.g., connectivity matrix)
%                    Allowed dimensions are 3D (node x node x nsubs x
%                    ntasks)
%        y           variable to be predicted, 1D vector, class 1 and 2
%        feat_loc    feature selection indices

%   OUTPUTS
%        acc         overall accuracy
%        acc_class1  accuracy for class 1
%        acc_class2  accuracy for class 2
%                    decision function for class 1
%   Adapted from rCPM script from Siyuan Gao https://github.com/YaleMRRC/CPM

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


xfeat = x(feat_loc, :);
accv=(predict(mdl, xfeat')==y);
acc = mean(accv);
acc_class1 = mean(accv(y==1));
acc_class2 = mean(accv(y==2));

end
