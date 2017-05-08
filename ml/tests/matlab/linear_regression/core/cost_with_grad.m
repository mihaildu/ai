% Mihail Dunaev
% May 2017

function [cost, grad] = cost_with_grad(t, xs, ys)
    % cost and gradient function for linear regression
    %   input
    %       t = params values for the model
    %       xs = train features
    %       ys = train output values
    %   output
    %       cost = value of cost function
    %       grad = value of gradient
    N = size(xs, 1);
    X = [xs ones(N, 1)];
    cost = sum(((X * t) - ys) .^ 2) / (2 * N);
    if nargout > 1
        num_params = size(t, 1);
        grad = zeros(num_params, 1);
        for i = 1:num_params
            grad(i) = sum(((X * t) - ys) .* X(:,i)) / N;
        end
    end
end