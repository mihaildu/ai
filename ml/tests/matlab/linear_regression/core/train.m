% Mihail Dunaev
% May 2017

function w = train(xs, ys, direct_method)
    % function that trains a linear regression model
    % this is just a wrapper for `train_direct` & `train_cost`
    %   input
    %       xs = training features xtrain
    %       ys = training values ytrain
    %       direct_method = boolean value; if true the direct method
    %           will be used (using the design matrix)
    %   output
    %       w = params for the model
    if direct_method == true
        w = train_direct(xs, ys);
    else
        w = train_cost(xs, ys);
    end
end

function w = train_direct(xs, ys)
    % function that trains a linear regression model
    % using the direct method (design matrix)
    % pseudo-inverse of matrix X = inv(X'*X)*X'
    N = size(xs, 1);
    X = [xs ones(N, 1)];
    w = pinv(X) * ys;
end

function w = train_cost(xs, ys)
    % function that trains a linear regression model
    % by minimizing a cost function
    syms t;
    t0 = [1; 1];
    fun = @(t)(cost_with_grad(t, xs, ys));

    % fminsearch (nelder & mead simplex)
    % this won't use the grad
    %[w, ~] = fminsearch(fun, t0);

    % fminunc
    %   no grad: bfgs quasi-newton
    %   grad: trust region
    options = optimset('fminunc');
    options = optimset(options, 'GradObj', 'on');
    [w, ~] = fminunc(fun, t0, options);

    % minimize with own gradient descent
    %w = grad_descent(fun, t0);
end