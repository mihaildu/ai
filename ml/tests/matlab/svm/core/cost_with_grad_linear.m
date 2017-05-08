% Mihail Dunaev
% May 2017

function [cost, grad] = cost_with_grad_linear(t, xs, ys, C)
    % cost & grad for linear SVM
    xs0 = xs(ys == 0,:);
    xs1 = xs(ys == 1,:);
    n0 = size(xs0, 1);
    n1 = size(xs1, 1);
    N = n0 + n1;
    cost = sum(max((1 - [xs1 ones(n1,1)] * t) * exp(1) / 2, 0)) + ...
        sum(max((1 + [xs0 ones(n0,1)] * t) * exp(1) / 2, 0));
    cost = C * cost + sum(t .^ 2) / 2;

    % lazy implementation for grad
    % TODO it shows ok on check_grad but doesn't work for minimization
    % TODO check sub-gradient
    % TODO check this for ND data points (N > 1)
    if nargout > 1
        grad = zeros(size(t));
        for i = 1:N
            if ys(i) == 0
                if [xs(i) 1] * t >= -1
                    %grad = grad + [(xs(i) ^ 2) * exp(1) / 2; exp(1) / 2];
                    grad = grad + [xs(i) * exp(1) / 2; exp(1) / 2];
                end
            elseif [xs(i) 1] * t <= 1
                grad = grad + [-xs(i) * exp(1) / 2; -exp(1) / 2];
            end
        end
        grad = C * grad + t;
    end
end