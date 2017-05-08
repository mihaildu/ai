% simple linear regression
% 
% for houses.in
%   cost = 353.3414
%   w train_direct = 10.8619, -22.2128, 1.6422e-04s
%   w train_cost fminsearch = 10.8619, -22.2128, 0.0036s
%   w train_cost fminunc = 10.8619, -22.2128, 0.0082s
%   w train_cost fminunc with grad = 10.8619, -22.2128, 0.0061s
%   own gradient descent 1.3612s

function main()
    read_data();
    train();
    plot_all();

    % predict some values
    xpred = [12 45]';
    ypred = pred(xpred);
    disp(ypred);
end

% function to read data from file
function read_data()
    houses = load('../input/houses.in');
    global xs ys N;
    xs = houses(:,1);
    ys = houses(:,2);
    N = size(xs, 1);
end

% wrapper for train
function train()
    %train_direct();
    train_cost();
end

% train using the pseudo-inverse matrix
function train_direct()
    global xs ys N w;
    
    % compute design matrix
    X = [xs ones(N, 1)];
    
    % w = pseudo-inverse * ys;
    % pseudo-inverse = inv(X'X) * X';
    w = pinv(X) * ys;
end

% train by minimizing a cost function
function train_cost()
    global xs ys N w;
    
    % define cost function
    syms t;
    X = [xs ones(N, 1)];
    cost = @(t)(sum(((X * t) - ys) .^ 2) / (2 * N));
    
    % minimize using fminsearch
    % this uses nelder & mead simplex
    %t0 = [1 1]';
    %[tf,~] = fminsearch(cost, t0);

    % minimize using fminunc
    % bfgs quasi-newton with no gradient
    %t0 = [1 1]';
    %[tf,~] = fminunc(cost, t0);
    
    % trust region algo with grad
    %t0 = [1 1]';
    %tmpfun = @cost_with_grad;
    %options = optimset('fminunc');
    %options = optimset(options, 'GradObj', 'on');
    %[tf,~] = fminunc(tmpfun, t0, options);
    
    % minimize with my own gradient descent
    t0 = [1 1]';
    tmpfun = @cost_with_grad;
    tf = grad_descent2(tmpfun, t0);
    
    % TODO - use minimize - loglike
    %t0 = [1 1]';
    %num_iter = 500;
    %tf, cost_final, iter] = minimize(t0, @neg_lr_loglike, num_iter, xs, ys);
    
    w = tf;
end

% improved version of grad descent
function tf = grad_descent2(tmpfun, t)
    alpha = 0.001;
    eps = 0.0001;
    max_iter = 1000000;
    
    [prev_cost,grad] = tmpfun(t);
    for i = 1:max_iter
        
        % if gradient too small we are close to the minimum
        % convergence also ok if (cost - prev_cost) < 0.001
        if norm(grad) <= eps
            break;
        end
        
        % update one step
        t = t - (alpha * grad);
        [cost,grad] = tmpfun(t);
        
        % if cost goes higher, it overshoots
        % this might not be enough
        if cost > prev_cost
            alpha = alpha / 2;
        end
        
        % if cost decreases too slowly make alpha higher TODO
        prev_cost = cost;
    end

    % possibly not minimum
    if i == max_iter
        disp('Maximum number of iterations reached');
    end
    tf = t;
end

% simplest version of grad descent
function tf = grad_descent(tmpfun, t)
    alpha = 0.001;
    num_iter = 100000;
    for i = 1:num_iter
        [~,grad] = tmpfun(t);
        t = t - (alpha * grad);
    end
    tf = t;
end

% cost function f and gradient g (fminunc)
function [f,g] = cost_with_grad(t)
    global xs ys N;    
    X = [xs ones(N, 1)];
    f = sum(((X * t) - ys) .^ 2) / (2 * N);
    if nargout > 1
        g = [sum(((X * t) - ys) .* xs) / N; sum((X * t) - ys) / N];
    end
end

function plot_all()
    global xs ys w;
    
    % plot initial data
    figure;
    hold on;
    plot(xs, ys, 'bo');
    
    % plot best fit
    delta = 0.1;
    xmin = min(xs);
    xmax = max(xs);
    num_pts = (xmax - xmin) / delta + 1;
    wxs = (xmin:delta:xmax)';
    wys = [wxs ones(num_pts,1)] * w;
    plot(wxs, wys, 'r-');
end

function ys = pred(xs)
    global w;
    npred = size(xs, 1);
    ys = [xs ones(npred,1)] * w;
end