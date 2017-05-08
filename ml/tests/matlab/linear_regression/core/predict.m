% Mihail Dunaev
% May 2017

function ypred = predict(xpred, w)
    % function that predicts values for xpred
    % using a linear regresssion model with params w
    N = size(xpred, 1);
    ypred = [xpred ones(N, 1)] * w;
end