% Mihail Dunaev
% May 2017

function [cost, grad] = cost_with_grad(t, xs, ys)
    % cost & grad for linear logistic regression
    xs0 = xs(ys == 0,:);
    xs1 = xs(ys == 1,:);
    n0 = size(xs0, 1);
    n1 = size(xs1, 1);
    N = n0 + n1;
    cost = (sum(-log(1 ./ (1 + exp(-[xs1 ones(n1,1)] * t)))) + ...
        sum(-log(1 - (1 ./ (1 + exp(-[xs0 ones(n0,1)] * t)))))) / N;
    if nargout > 1
        num_params = size(t, 1);
        grad = zeros(num_params, 1);
        X = [xs ones(N, 1)];
        for i = 1:num_params
            grad(i) = sum(((1 ./ (1 + exp(-[xs ones(N,1)] * t))) - ys) ...
                .*  X(:,i)) / N;
        end
    end
end