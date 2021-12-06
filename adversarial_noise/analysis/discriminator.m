function acc = discriminator(x, fake_real, kfolds)
% classify between real/adversarial connectomes
cv_idx=cv_indices(size(x, 2), kfolds);
fake_pred_correct = zeros(size(fake_real, 1), 1);
for idx_fold = 1:kfolds
    train_idx = find(cv_idx~=idx_fold);
    test_idx = find(cv_idx==idx_fold);
    xtrain = x(:, train_idx, :); xtrain=reshape(xtrain, [size(xtrain, 1), 2*size(xtrain, 2)]);
    xtest = x(:, test_idx, :); xtest=reshape(xtest, [size(xtest, 1), 2*size(xtest, 2)]);
    ytrain = fake_real(train_idx, :); ytrain=reshape(ytrain, [2*size(ytrain,1), 1]);
    
    mdl = fitclinear(xtrain',ytrain, 'Learner', 'svm');  % allow for regularization
    df = reshape(mdl.Beta'*xtest+mdl.Bias, [length(test_idx), 2]);  % decision function
    [~, fake_pred_correct(test_idx)] = max(df, [], 2);
end
fake_pred_correct = fake_pred_correct-1;  %1/2 to 0/1 for incorrect/correct

acc = mean(fake_pred_correct);

