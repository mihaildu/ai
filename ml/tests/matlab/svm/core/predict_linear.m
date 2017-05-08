% Mihail Dunaev
% May 2017

function ys = predict_linear(xs, w)
    % function that predicts values for xpred
    % using a linear SVM with params w
    npred = size(xs, 1);
    ys = [xs ones(npred,1)] * w >= 0;
end