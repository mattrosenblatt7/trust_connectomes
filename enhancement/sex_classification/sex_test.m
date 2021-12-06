function [acc] =sex_test(mdl, x,sex, feat_loc)
% train to add adversarial noise
% called from run_adv_noise.m
% traintest parameter is used for determining whether to learn the
% adversarial noise ('train') or apply previously learned adversarial noise
% to a new dataset ('test')

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


xfeat = x(feat_loc, :);
acc = mean(predict(mdl, xfeat')==sex);

end
