% Mihail Dunaev
% May 2017

function tf = grad_descent_simple(fun, t)
    % simple implementation of gradient descent
    %   input
    %       fun = function to minimize
    %       t = initial vars for fun
    %   output
    %       tf = values for t that minimize fun
    alpha = 0.001;
    num_iter = 100000;
    for i = 1:num_iter
        [~, grad] = fun(t);
        t = t - (alpha * grad);
    end
    tf = t;
end