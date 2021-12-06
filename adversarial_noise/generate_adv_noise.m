function [adv_noise]=...
    generate_adv_noise(mdl, x,y,feat_loc, dec_1)
% train adversarial noise using gradient descent
% note: depending on your application, you may wish to adjust learning rate
% in this code
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   REQUIRED INPUTS
%        mdl         trained classifier
%        x           Predictor variable (e.g., connectivity matrix)
%                    Allowed dimensions are 3D (node x node x nsubs x
%                    ntasks)
%        y           variable to be predicted, 1D vector, class 1 and 2
%        feat_loc    feature selection indices
%        dec_1       sign of decision function for class 1 of data
%
%   OUTPUTS
%        adv_noise   generated adversarial nosie pattern

% example input:
%  [adv_noise]=generate_adv_noise(mdl, x,y,feat_loc, dec_1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Set seed and initialize predicted y variable
nsubs=size(x,2);

% feature location
xfeat = x(feat_loc, :);

% learning rate/step size/l2 regularization
acc_1_2 = [];
mean_abs_1_2 = [];
lr = length(feat_loc)*5e-6;
l2_param = 0;  % has minimal effect on results for linear model

% focus on only a single class at once: changing 1->2
n = zeros(1, size(xfeat', 2));
xfeat_1 = xfeat(:, y==1);
for idx = 1:5000  % stop at 5000 iterations
    
    n_1_2 = n;
    
    xt = xfeat_1 + n';
    ypred = predict(mdl, xt');
    
    % save info about noise each iteration
    max_abs_1_2(idx) = max(abs(n));
    mean_abs_1_2(idx) = mean(abs(n));
    acc_1_2(idx) = mean(ypred==1);
    corr_1_2(idx) = mean(diag(corr(xfeat_1, xt)));  % correlation between true and adversarial features
    
    % if classification is 0% accuracy (i.e., all class1 classified as
    % class2), then break the loop
    if idx>1
        if mean(ypred)==2
            disp('ending')
            n_1_2 = n;
            break
        end
    end
    
    % derivative of loss and noise update
    dLdn = dec_1*mdl.Beta + l2_param*mean((n))';
    n = n - lr*dLdn';
end


% focus on only a single class at once: changing 2->1
n = zeros(1, size(xfeat', 2));
xfeat_2 = xfeat(:, y==2);
for idx = 1:5000
    
    n_2_1 = n;
    
    xt = xfeat_2 + n';
    ypred = predict(mdl, xt');
    
    % save info about noise each iteration
    max_abs_2_1(idx) = max(abs(n));
    mean_abs_2_1(idx) = mean(abs(n));
    acc_2_1(idx) = mean(ypred==2);
    corr_2_1(idx) = mean(diag(corr(xfeat_2, xt)));

    % if classification is 0% accuracy (i.e., all class2 classified as
    % class1), then break the loop
    if idx>1
        if mean(ypred)==1
            disp('ending')
            n_2_1 = n;
            break
        end
    end
    
    % derivative of loss and noise update
    dLdn = -dec_1*mdl.Beta + l2_param*mean((n))';
    n = n - lr*dLdn';
end


% store results in adv_noise variable
adv_noise.max_abs_1_2 = max_abs_1_2;
adv_noise.mean_abs_1_2 = mean_abs_1_2;
adv_noise.acc_1_2 = acc_1_2;
adv_noise.n_1_2 = n_1_2;
adv_noise.corr_1_2 = corr_1_2;

adv_noise.max_abs_2_1 = max_abs_2_1;
adv_noise.mean_abs_2_1 = mean_abs_2_1;
adv_noise.acc_2_1 = acc_2_1;
adv_noise.n_2_1 = n_2_1;
adv_noise.corr_2_1 = corr_2_1;



