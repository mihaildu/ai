% Mihail Dunaev
% May 2017
%
% Simple test that tries to separate 3 sets of 2D data on a plane
%   using one-vs-all linear logistic regression. Data is in 
%   'logistic_multiclass.in'.

function test5_multiclass()
    addpath('core/');
    [xs, ys] = read_data('../../input/logistic_multiclass.in');

    syms t;
    w0 = [1; -1; 1];
    W = zeros(3, 3);
    fun = @(t)(my_cost_with_grad_c0(t, xs, ys));
    W(:,1) = train(w0, fun);
    fun = @(t)(my_cost_with_grad_c1(t, xs, ys));
    W(:,2) = train(w0, fun);
    fun = @(t)(my_cost_with_grad_c2(t, xs, ys));
    W(:,3) = train(w0, fun);
    
    xpred = [6 2.1; 4.5 4.5; 6 7; 6 6.1; 3 6; 2 1.5; 5 4];
    ypred = my_predict(xpred, W);
    plot_all(xs, ys, W, xpred, ypred);
end

function [cost, grad] = my_cost_with_grad_c0(t, xs, ys)
    % cost & grad for linear logistic regression with 3 classes
    % this takes class with y = 0 vs all
    xs0 = xs(ys == 0,:);
    xs1 = xs(ys == 1,:);
    xs2 = xs(ys == 2,:);
    n0 = size(xs0, 1);
    n1 = size(xs1, 1);
    n2 = size(xs2, 1);
    N = n0 + n1 + n2;
    cost = (sum(-log(1 ./ (1 + exp(-[xs0 ones(n0,1)] * t)))) + ...
        sum(-log(1 - (1 ./ (1 + exp(-[[xs1;xs2] ones(n1+n2,1)] * t)))))) / N;
    if nargout > 1
        % ys = 1 for xs0, 0 for xs1, xs2
        ys0 = ys;
        ys0(ys0 == 1) = 2;
        ys0(ys0 == 0) = 1;
        ys0(ys0 == 2) = 0;
        num_params = size(t, 1);
        grad = zeros(num_params, 1);
        X = [xs ones(N, 1)];
        for i = 1:num_params
            grad(i) = sum(((1 ./ (1 + exp(-[xs ones(N,1)] * t))) - ys0) ...
                .*  X(:,i)) / N;
        end
    end
end

function [cost, grad] = my_cost_with_grad_c1(t, xs, ys)
    % cost & grad for linear logistic regression with 3 classes
    % this takes class with y = 1 vs all
    xs0 = xs(ys == 0,:);
    xs1 = xs(ys == 1,:);
    xs2 = xs(ys == 2,:);
    n0 = size(xs0, 1);
    n1 = size(xs1, 1);
    n2 = size(xs2, 1);
    N = n0 + n1 + n2;
    cost = (sum(-log(1 ./ (1 + exp(-[xs1 ones(n1,1)] * t)))) + ...
        sum(-log(1 - (1 ./ (1 + exp(-[[xs0;xs2] ones(n0+n2,1)] * t)))))) / N;
    if nargout > 1
        % ys = 1 for xs1, 0 for xs0, xs2
        ys1 = ys;
        ys1(ys1 == 2) = 0;
        num_params = size(t, 1);
        grad = zeros(num_params, 1);
        X = [xs ones(N, 1)];
        for i = 1:num_params
            grad(i) = sum(((1 ./ (1 + exp(-[xs ones(N,1)] * t))) - ys1) ...
                .*  X(:,i)) / N;
        end
    end
end

function [cost, grad] = my_cost_with_grad_c2(t, xs, ys)
    % cost & grad for linear logistic regression with 3 classes
    % this takes class with y = 2 vs all
    xs0 = xs(ys == 0,:);
    xs1 = xs(ys == 1,:);
    xs2 = xs(ys == 2,:);
    n0 = size(xs0, 1);
    n1 = size(xs1, 1);
    n2 = size(xs2, 1);
    N = n0 + n1 + n2;
    cost = (sum(-log(1 ./ (1 + exp(-[xs2 ones(n2,1)] * t)))) + ...
        sum(-log(1 - (1 ./ (1 + exp(-[[xs0;xs1] ones(n0+n1,1)] * t)))))) / N;
    if nargout > 1
        % ys = 1 for xs2, 0 for xs0, xs1
        ys2 = ys;
        ys2(ys2 == 1) = 0;
        ys2(ys2 == 2) = 1;
        num_params = size(t, 1);
        grad = zeros(num_params, 1);
        X = [xs ones(N, 1)];
        for i = 1:num_params
            grad(i) = sum(((1 ./ (1 + exp(-[xs ones(N,1)] * t))) - ys2) ...
                .*  X(:,i)) / N;
        end
    end
end

function ypred = my_predict(xpred, W)
    % function that predicts class for xpred
    % using a linear one-vs-all logistic regresssion model with params W
    syms t;
    N = size(xpred, 1);
    ypred = zeros(N, 1);
    sigmoid = @(t)(1 / (1 + exp(-t)));

    % compute probs for each class
    P =  arrayfun(sigmoid,[xpred ones(N,1)] * W);
    for i = 1:N
        if P(i,1) == P(i,2) == P(i,3)
            ypred(i) = -1;
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
        ypred(i) = y;
    end
end

function plot_all(xs, ys, W, xpred, ypred)
    xs0 = xs(ys == 0,:);
    xs1 = xs(ys == 1,:);
    xs2 = xs(ys == 2,:);
    
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
    
    % plot separating lines
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
