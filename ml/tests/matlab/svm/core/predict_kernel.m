% Mihail Dunaev
% May 2017

function ys = predict_kernel(xpred, w, xs)
    % function that predicts values for xpred
    % using a kernel SVM with params w
    npred = size(xpred, 1);
    ys = zeros(npred, 1);
    for i = 1:npred
        f = kernel_vec(xpred(i,:), xs);
        ys(i) = (f * w >= 0);
    end
end