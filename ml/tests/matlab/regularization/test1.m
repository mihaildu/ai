% Mihail Dunaev
% May 2017
%
% Simple test that shows how regularization works. We are fitting
% a 9th degree polynomial to linear data, and then add regularization.
%
% TODO tweak the cost minimization one a bit/feature scaling

function test1()
    [xs, ys] = read_data();
    
    use_reg = false;
    direct_method = false;
    w = train(xs, ys, use_reg, direct_method);
    plot_all(xs, ys, w);
    
    % predict some values
    %xpred = [12 45]';
    %ypred = pred(xpred);
    %disp(ypred);
end

% wrapper for train
function w = train(xs, ys, use_reg, direct_method)
    if use_reg == true
        % big values are needed since features are not scaled
        %lambda = 1e08;
        %lambda = 1e20;
        lambda = 1e23;
    else
        lambda = 0;
    end
    if direct_method == true
        w = train_direct(xs, ys, lambda);
    else
        w = train_cost(xs, ys, lambda);
    end
end

% train using the pseudo-inverse matrix
function w = train_direct(xs, ys, lambda)
    N = size(xs, 1);
    
    % compute design matrix
    degree_poly = 9;
    X = ones(N, degree_poly+1);
    for i = 1:degree_poly
        X(:,i) = xs .^ (degree_poly + 1 - i);
    end
    
    % no reg
    if lambda == 0
        w = pinv(X) * ys;
        return;
    end
    
    % decay matrix used in pseudo inverse
    dm = eye(degree_poly + 1);
    % don't use reg for bias param
    dm(1,1) = 0;
    w = (X' * X + lambda * dm) \ (X' * ys);
end

function w = train_cost(xs, ys, lambda)
    syms t;
    t0 = [3000 5 1 0 0.0001 0.0001 0 0 0 0]';
    fun = @(t)(cost_with_grad(t, xs, ys, lambda));
    
    options = optimset('fminunc');
    options = optimset(options, 'GradObj', 'on');
    [w, ~] = fminunc(fun, t0, options);
end

function [cost, grad] = cost_with_grad(t, xs, ys, lambda)
    N = size(xs, 1);
    degree_poly = 9;
    X = ones(N, degree_poly+1);
    for i = 1:degree_poly
        X(:,i) = xs .^ (degree_poly + 1 - i);
    end
    
    cost = sum(((X * t) - ys) .^ 2) / (2 * N);
    if lambda > 0
        cost = cost + (lambda * sum(t(2:end) .^ 2)) / (2 * N);
    end

    if nargout > 1
        num_params = size(t, 1);
        grad = zeros(num_params, 1);
        for i = 1:num_params
            grad(i) = sum(((X * t) - ys) .* X(:,i)) / N;
        end
        if lambda > 0
            for i = 1:(num_params-1)
                grad(i) = grad(i) + (lambda * t(i)) / N;
            end
        end
    end
end

function [xs, ys] = read_data()
    data = load('../../input/regularization.in');
    xs = data(:,1);
    ys = data(:,2);
end

function plot_all(xs, ys, w)
    % plot initial data
    figure;
    hold on;
    plot(xs, ys, 'bo');
    xmin = min(xs) - 20;
    xmax = max(xs) + 20;
    ymin = min(ys) - 10;
    ymax = max(ys) + 10;
    xlim([xmin xmax]);
    ylim([ymin ymax]);

    % plot best fit
    delta = 0.1;
    num_pts = (xmax - xmin) / delta + 1;
    wxs = (xmin:delta:xmax)';
    degree_poly = 9;
    W = ones(num_pts, degree_poly+1);
    for i = 1:degree_poly
        W(:,i) = wxs .^ (degree_poly + 1 - i);
    end
    wys = W * w;
    plot(wxs, wys, 'r-');
end