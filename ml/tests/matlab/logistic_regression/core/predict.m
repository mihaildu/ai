% Mihail Dunaev
% May 2017

function ypred = predict(xpred, w)
    % function that predicts values for xpred
    % using a linear logistic regresssion model with params w
    N = size(xpred, 1);
    % if [x 1] * w >= 0 => class 1, 0 otherwise
    ypred = [xpred ones(N, 1)] * w >= 0;
end