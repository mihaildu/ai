% simple logistic regression
% 
% for tumors.in
%   threshold value for x is at -w(2)/w(1)
%   fminsearch w = 1.9564, -85.8940, cost = 0, x0 = 43.9041
%   fminunc w = 0.8260, -41.5848, cost = 1.0601e-06, x0 = 50.3448
%   fminunc with grad w = 0.6321, -27.3605, cost = 1.0773e-06, x0 = 43.2851
%   own gradient descent w = 1.8705, -70.4174, cost = 2.9976e-14,
%                        x0 = 37.6463 (alpha = 7)
%
% for tumors2.in
%   fminsearch w = 0.1425, -8.6982, cost = 4.399, x0 = 61.04
%   fminunc w = 0.1425, -8.6982, cost = 4.399, x0 = 61.04
%   fminunc with grad w = 0.1425, -8.6982, cost = 4.399, x0 = 61.04
%   own gradient descent w = 0.1436, -8.7625, cost = 4.3991,
%                        x0 = 61.0202 (alpha = 3)
%
% for logistic.in
%   fminsearch w = 1.0e+03 * [0.4683, -0.1113, -1.1665], cost = 0 
%   fminunc w = 16.3325, 2.6081, -67.5492, cost = 7.4372e-06
%   fminunc with grad w = 8.6056, 4.0303, -46.8913, cost = 4.7543e-05,
%   own gradient descent w = 5.3367, 2.3259, -28.1962, cost = 0.00494
%                       (alpha = 3)

function main()
    % some preparation
    global sigmoid;
    syms t;
    sigmoid = @(t)(1 / (1 + exp(-t)));
    
    %tumor_test1();
    %tumor_test2();
    %test3();
    %test4();
    multiclass_test5();
end

% one-vs-all multiclass classification with logreg
function multiclass_test5()
    read_data_multiclass();
    train_multiclass();
    predict_multiclass();
    plot_multiclass();
end

% read data for multiple classes (3)
function read_data_multiclass()
    global xs xs0 xs1 xs2 ys n0 n1 n2 N;
    data = load('../../../input/logistic_multiclass.in');
    xs = data(:,1:end-1);
    ys = data(:,end);
    xs0 = xs(ys == 0,:);
    n0 = size(xs0, 1);
    xs1 = xs(ys == 1,:);
    n1 = size(xs1, 1);
    xs2 = xs(ys == 2,:);
    n2 = size(xs2, 1);
    N = size(xs, 1);
end

% train function for the tumors problem
function train_multiclass()
    global W;

    % W(:,1) = params for class 0, W(:,2) = 1, W(:,3)
    W = zeros(3,3);

    % fminunc with grad
    t0 = [1 1 1]';
    %t0 = [-1 0.5 1.5]';
    fun0 = @cost_with_grad_test5_c0;
    fun1 = @cost_with_grad_test5_c1;
    fun2 = @cost_with_grad_test5_c2;
    options = optimset('fminunc');
    options = optimset(options, 'GradObj', 'on');
    [W(:,1),~] = fminunc(fun0, t0, options);
    [W(:,2),~] = fminunc(fun1, t0, options);
    [W(:,3),~] = fminunc(fun2, t0, options);
    %disp(W);
end

% cost & grad for first class in multiclass logreg
% 1 on the inside, 0 outside
function [cost, grad] = cost_with_grad_test5_c0(t)
    global xs0 xs1 xs2 n1 n0 n2 N xs ys;
    cost = sum(-log(1 ./ (1 + exp(-[xs0 ones(n0,1)] * t)))) + ...
        sum(-log(1 - (1 ./ (1 + exp(-[[xs1;xs2] ones(n1+n2,1)] * t)))));
    % ys = 1 for xs0, 0 for xs1, xs2
    ys0 = ys;
    ys0(ys0 == 1) = 2;
    ys0(ys0 == 0) = 1;
    ys0(ys0 == 2) = 0;
    if nargout > 1
        grad = [
        sum(((1 ./ (1 + exp(-[xs ones(N,1)] * t))) - ys0) .* xs(:,1)) / N;
        sum(((1 ./ (1 + exp(-[xs ones(N,1)] * t))) - ys0) .* xs(:,2)) / N;
        sum((1 ./ (1 + exp(-[xs ones(N,1)] * t))) - ys0) / N];
    end
end

% cost & grad for second class in multiclass logreg
% 1 on the inside, 0 outside
function [cost, grad] = cost_with_grad_test5_c1(t)
    global xs0 xs1 xs2 n1 n0 n2 N xs ys;
    cost = sum(-log(1 ./ (1 + exp(-[xs1 ones(n1,1)] * t)))) + ...
        sum(-log(1 - (1 ./ (1 + exp(-[[xs0;xs2] ones(n0+n2,1)] * t)))));
    % ys = 1 for xs1, 0 for xs0, xs2
    ys1 = ys;
    ys1(ys1 == 2) = 0;
    if nargout > 1
        grad = [
        sum(((1 ./ (1 + exp(-[xs ones(N,1)] * t))) - ys1) .* xs(:,1)) / N;
        sum(((1 ./ (1 + exp(-[xs ones(N,1)] * t))) - ys1) .* xs(:,2)) / N;
        sum((1 ./ (1 + exp(-[xs ones(N,1)] * t))) - ys1) / N];
    end
end

% cost & grad for third class in multiclass logreg
% 1 on the inside, 0 outside
function [cost, grad] = cost_with_grad_test5_c2(t)
    global xs0 xs1 xs2 n1 n0 n2 N xs ys;
    cost = sum(-log(1 ./ (1 + exp(-[xs2 ones(n2,1)] * t)))) + ...
        sum(-log(1 - (1 ./ (1 + exp(-[[xs0;xs1] ones(n0+n1,1)] * t)))));
    % ys = 1 for xs2, 0 for xs0, xs1
    ys2 = ys;
    ys2(ys2 == 1) = 0;
    ys2(ys2 == 2) = 1;
    if nargout > 1
        grad = [
        sum(((1 ./ (1 + exp(-[xs ones(N,1)] * t))) - ys2) .* xs(:,1)) / N;
        sum(((1 ./ (1 + exp(-[xs ones(N,1)] * t))) - ys2) .* xs(:,2)) / N;
        sum((1 ./ (1 + exp(-[xs ones(N,1)] * t))) - ys2) / N];
    end
end

% predict for test 5
function predict_multiclass()
    global xpred ypred;
    xpred = [6 2.1; 4.5 4.5; 6 7; 6 6.1; 3 6; 2 1.5; 5 4];
    ypred = pred_multiclass(xpred);
    disp(ypred);
end

% helper function
function ys = pred_multiclass(xs)
    global W sigmoid;
    npred = size(xs, 1);
    ys = zeros(npred, 1);
    
    % compute probs for each class
    P =  arrayfun(sigmoid,[xs ones(npred,1)] * W);
    for i = 1:npred
        if P(i,1) == P(i,2) == P(i,3)
            ys(i) = -1;
            continue;
        end
        maxp = P(i,1);
        y = 0;
        if P(i,2) > maxp
            maxp = P(i,2);
            y = 1;
        end
        if P(i,3) > maxp
            y = 2;
        end
        ys(i) = y;
    end
end

% plotting for multiclass (test 5)
function plot_multiclass()
    global xs0 xs1 xs2 xs xpred ypred W;
    
    % plot initial data
    figure;
    hold on;
    plot(xs0(:,1), xs0(:,2), 'bo');
    plot(xs1(:,1), xs1(:,2), 'rd');
    plot(xs2(:,1), xs2(:,2), 'gs');
    
    % set range for x/y axis
    xmin = min(xs(:,1));
    xmax = max(xs(:,1));
    ymin = min(xs(:,2));
    ymax = max(xs(:,2));
    xlim([xmin-1 xmax+1]);
    ylim([ymin-1 ymax+1]);
    
    % plot separating line
    x0 = [xmin; xmax];
    y0 = [-(W(3,1)+(W(1,1) * xmin))/W(2,1); -(W(3,1)+(W(1,1) * xmax))/W(2,1)];
    y1 = [-(W(3,2)+(W(1,2) * xmin))/W(2,2); -(W(3,2)+(W(1,2) * xmax))/W(2,2)];
    y2 = [-(W(3,3)+(W(1,3) * xmin))/W(2,3); -(W(3,3)+(W(1,3) * xmax))/W(2,3)];
    plot(x0, y0, 'b-');
    plot(x0, y1, 'r-');
    plot(x0, y2, 'g-');
    
    % plot predicted points
    npred = size(xpred, 1);
    for i = 1:npred
        if ypred(i) == 0
            plot(xpred(i,1), xpred(i,2), 'bx');
        elseif ypred(i) == 1
            plot(xpred(i,1), xpred(i,2), 'rx');
        elseif ypred(i) == 2
            plot(xpred(i,1), xpred(i,2), 'gx');
        else
            plot(xpred(i,1), xpred(i,2), 'kx');
        end
    end
end

% separating between two sets of points in a plane (circle)
function test4()
    read_data_logistic('../../../input/logistic_circle.in');
    train_test4();
    predict_test4();
    plot_test4();
end

% train function for the tumors problem
function train_test4()
    t0 = [1 -1 1 -2 0.5]';
    train(t0, @cost_with_grad_test4);
end

% cost function and gradient for multiple features (circle)
function [cost, grad] = cost_with_grad_test4(t)
    global xs1 xs0 n1 n0 N xs ys;

    % cost function for circle/ellipse separator
    cost = (sum(-log(1 ./ (1 + exp(-[xs1 .^ 2 xs1 ones(n1,1)] * t)))) + ...
        sum(-log(1 - (1 ./ (1 + exp(-[xs0 .^ 2 xs0 ones(n0,1)] * t)))))) / N;

    if nargout > 1
        % grad for ellipse
        grad = [
        sum(((1 ./ (1 + exp(-[xs .^2 xs ones(N,1)] * t))) - ys) .* (xs(:,1) .^ 2)) / N;
        sum(((1 ./ (1 + exp(-[xs .^2 xs ones(N,1)] * t))) - ys) .* (xs(:,2) .^ 2)) / N;
        sum(((1 ./ (1 + exp(-[xs .^2 xs ones(N,1)] * t))) - ys) .* xs(:,1)) / N;
        sum(((1 ./ (1 + exp(-[xs .^2 xs ones(N,1)] * t))) - ys) .* xs(:,2)) / N;
        sum((1 ./ (1 + exp(-[xs .^2 xs ones(N,1)] * t))) - ys) / N];
    end
end

% predict for test 4
function predict_test4()
    global xpred ypred;
    %xpred = [3 4; 6 5; 4 3; 4.2 3];
    xpred = [3.5 4; 4 4.1; 2 6];
    ypred = pred_test4(xpred);
    disp(ypred);
end

% helper function
function ys = pred_test4(xs)
    global w;
    npred = size(xs, 1);
    ys = [xs .^2 xs ones(npred,1)] * w >= 0;
end

% plotting for 2 features data (test 4)
function plot_test4()
    global xs0 xs1 xs xpred ypred w sigmoid;
    
    % plot initial data
    figure;
    hold on;
    plot(xs0(:,1), xs0(:,2), 'bo');
    plot(xs1(:,1), xs1(:,2), 'rx');
    
    % set range for x/y axis
    xmin = min(xs(:,1));
    xmax = max(xs(:,1));
    ymin = min(xs(:,2));
    ymax = max(xs(:,2));
    xlim([xmin-1 xmax+1]);
    ylim([ymin-1 ymax+1]);
    
    % plot best fit (3d)
    delta = 0.1;
    [wxs,wys] = meshgrid(xmin:delta:xmax, ymin:delta:ymax);
    wzs = arrayfun(sigmoid, (wxs .^ 2) .* w(1) + (wys .^ 2) .* w(2) + ...
        wxs .* w(3) + wys .* w(4) + w(5));    
    surf(wxs, wys, wzs);

    % plot predicted points
    %plot3(xpred(:,1), xpred(:,2), ypred, 'go');
    plot(xpred(:,1), xpred(:,2), 'go');
end

% separating between two sets of points in a plane (line)
function test3()
    read_data_logistic('../../../input/logistic.in');
    train_test3();
    predict_test3();
    plot_test3();
end

% train function for the tumors problem
function train_test3()
    t0 = [1 -1 1]';
    train(t0, @cost_with_grad_test3);
end

% cost function and gradient for multiple features (linear)
function [cost, grad] = cost_with_grad_test3(t)
    global xs1 xs0 n1 n0 N xs ys;
    
    % cost function for linear separator
    cost = sum(-log(1 ./ (1 + exp(-[xs1 ones(n1,1)] * t)))) + ...
        sum(-log(1 - (1 ./ (1 + exp(-[xs0 ones(n0,1)] * t)))));

    if nargout > 1
        % grad for linear separator
        grad = [
        sum(((1 ./ (1 + exp(-[xs ones(N,1)] * t))) - ys) .* xs(:,1)) / N;
        sum(((1 ./ (1 + exp(-[xs ones(N,1)] * t))) - ys) .* xs(:,2)) / N;
        sum((1 ./ (1 + exp(-[xs ones(N,1)] * t))) - ys) / N];
    end
end

% predict for test 3
function predict_test3()
    global xpred ypred;
    %xpred = [3 4; 6 5; 4 3; 4.2 3];
    xpred = [3.5 4; 4 4.1; 2 6];
    % pred_tumors still ok
    ypred = pred_tumors(xpred);
end

% plotting for 2 features data (test 3)
function plot_test3()
    global xs0 xs1 xs xpred ypred w sigmoid;
    
    % plot initial data
    figure;
    hold on;
    plot(xs0(:,1), xs0(:,2), 'bo');
    plot(xs1(:,1), xs1(:,2), 'rx');
    
    % set range for x/y axis
    xmin = min(xs(:,1));
    xmax = max(xs(:,1));
    ymin = min(xs(:,2));
    ymax = max(xs(:,2));
    xlim([xmin-1 xmax+1]);
    ylim([ymin-1 ymax+1]);
    
    % plot separating line
    x0 = [xmin; xmax];
    y0 = [-(w(3)+(w(1) * xmin))/w(2); -(w(3)+(w(1) * xmax))/w(2)];
    plot(x0, y0);
    
    % plot best fit (3d)
    delta = 0.1;
    [wxs,wys] = meshgrid(xmin:delta:xmax, ymin:delta:ymax);
    wzs = arrayfun(sigmoid, wxs .* w(1) + wys .* w(2) + w(3));
    surf(wxs, wys, wzs);
    
    % plot predicted points
    %plot3(xpred(:,1), xpred(:,2), ypred, 'go');
    plot(xpred(:,1), xpred(:,2), 'go');
end

% one feature as input (tumor size)
function tumor_test1()
    read_data_logistic('../../../input/tumors.in');
    train_tumors();
    predict_tumors();
    plot_tumors();
end

% one feature as input test 2
function tumor_test2()
    read_data_logistic('../../../input/tumors2.in');
    train_tumors();
    predict_tumors();
    plot_tumors();
    plot_cost();
end

% train function for the tumors problem
function train_tumors()
    %t0 = [1 1]';
    t0 = [0 -5]';
    train(t0, @cost_with_grad_tumors);
end

% cost function and gradient for the tumors problem (one feature)
function [cost, grad] = cost_with_grad_tumors(t)
    global xs1 xs0 n1 n0 N xs ys;
    
    cost = (sum(-log(1 ./ (1 + exp(-[xs1 ones(n1,1)] * t)))) + ...
        sum(-log(1 - (1 ./ (1 + exp(-[xs0 ones(n0,1)] * t)))))) / N;
    % other way of computing this
    % sigmoid = @(t)(1 / (1 + exp(-t)));
    % cost = @(t)((sum(-log(arrayfun(sigmoid,[xs1 ones(n1,1)] * t))) + ...
    % sum(-log(1 - arrayfun(sigmoid,[xs0 ones(n0,1)] * t)))) / N)

    if nargout > 1
        grad = [
        sum(((1 ./ (1 + exp(-[xs ones(N,1)] * t))) - ys) .* xs) / N;
        sum((1 ./ (1 + exp(-[xs ones(N,1)] * t))) - ys) / N];
    end
end

% predict some values
function predict_tumors()
    global xpred ypred;
    xpred = [7 43 44 70]';
    ypred = pred_tumors(xpred);
end

% helper function
function ys = pred_tumors(xs)
    global w;
    npred = size(xs, 1);
    % if [x 1] * w >= 0 => class 1, 0 otherwise
    ys = [xs ones(npred,1)] * w >= 0;
end

% plotting for the tumors problem (1 feature)
function plot_tumors()
    global xs0 xs1 xs xpred ypred w sigmoid;
    
    % plot initial data
    figure;
    hold on;
    plot(xs0, 0, 'bo');
    plot(xs1, 1, 'rx');
    
    % set range for x/y axis
    xmin = min(xs);
    xmax = max(xs);
    xlim([xmin-1 xmax+1]);
    ylim([-0.5 1.5]);

    % plot best fit
    delta = 0.1;
    num_pts = (xmax - xmin) / delta + 1;
    wxs = (xmin:delta:xmax)';
    wys = arrayfun(sigmoid, [wxs ones(num_pts,1)] * w);
    plot(wxs, wys, 'k-');
    
    % plot predicted points
    plot(xpred, ypred, 'go');
end

% generic read data (multiple features)
function read_data_logistic(fname)
    global xs xs0 xs1 ys n0 n1 N;
    data = load(fname);
    xs = data(:,1:end-1);
    ys = data(:,end);
    xs0 = xs(ys == 0,:);
    n0 = size(xs0, 1);
    xs1 = xs(ys == 1,:);
    n1 = size(xs1, 1);
    N = size(xs, 1);
end

% generic train function
function train(t0, fun)
    global w;
    
    % syms t;
    % cost = @(t)(...);
    
    % minimize using fminsearch
    %[tf,~] = fminsearch(cost, t0);
    
    % minimize using fminunc
    %[tf,~] = fminunc(cost, t0);
    
    % minimize with my own gradient descent
    %tf = grad_descent(fun, t0);
    
    % TODO minimize/MLE
    
    % minimize using fminunc with grad
    options = optimset('fminunc');
    options = optimset(options, 'GradObj', 'on', 'OutputFcn', @outfun);
    [tf,~] = fminunc(fun, t0, options);
    w = tf;
end

% grad descent
function tf = grad_descent(tmpfun, t)
    %alpha = 7;
    alpha = 3;
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

    disp(cost);
    % possibly not minimum
    if i == max_iter
        disp('Maximum number of iterations reached');
    end
    tf = t;
end

% function that gets called after every step in fminunc
function stop = outfun(t, optimValues, state)
	% t = current values for thetas
	% optimValues.fval = function value for t
	% optimValues.iteration = iteration number
	% optimValues.procedure = procedure message
    %   what I print in cost_with_grad
	% optimValues.funccount = number of function evals until now
    global cost_vals;
    stop = false;
    if optimValues.iteration == 0
        cost_vals = [];
    end
    cost_vals = [cost_vals; optimValues.fval];
end

% plots evolution of cost function
function plot_cost()
    global cost_vals;
    figure;
    hold on;
    plot((1:size(cost_vals, 1))', cost_vals);
end