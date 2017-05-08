% Mihail Dunaev
% May 2017
%
% Simple test that tries to separate 2 sets of 2D data on a plane
%   using polynomial (2nd degree) logistic regression. Data is in 
%   'logistic_circle.in'.

function test4_circle()
    addpath('core/');
    [xs, ys] = read_data('../../input/logistic_circle.in');

    syms t;
    w0 = [1; -1; 1; -2; 0.5];
    fun = @(t)(my_cost_with_grad(t, xs, ys));
    w = train(w0, fun);
    
    xpred = [3.5 4; 4 4.1; 2 6];
    ypred = my_predict(xpred, w);
    plot_all(xs, ys, w, xpred, ypred);
    %plot_cost();
end

function [cost, grad] = my_cost_with_grad(t, xs, ys)
    % cost & grad for circle/ellipse logistic regression
    xs0 = xs(ys == 0,:);
    xs1 = xs(ys == 1,:);
    n0 = size(xs0, 1);
    n1 = size(xs1, 1);
    N = n0 + n1;
    cost = (sum(-log(1 ./ (1 + exp(-[xs1 .^ 2 xs1 ones(n1,1)] * t)))) + ...
        sum(-log(1 - (1 ./ (1 + exp(-[xs0 .^ 2 xs0 ones(n0,1)] * t)))))) / N;
    
    if nargout > 1
        num_params = size(t, 1);
        grad = zeros(num_params, 1);
        X = [xs .^ 2 xs ones(N, 1)];
        for i = 1:num_params
            grad(i) = sum(((1 ./ (1 + exp(-[xs .^ 2 xs ones(N,1)] * t))) ...
                - ys) .*  X(:,i)) / N;
        end
    end
end

function ypred = my_predict(xpred, w)
    % function that predicts values for xpred
    % using a circle/ellipse logistic regresssion model with params w
    N = size(xpred, 1);
    % if [x 1] * w >= 0 => class 1, 0 otherwise
    ypred = [xpred .^2 xpred ones(N, 1)] * w >= 0;
end

function plot_all(xs, ys, w, xpred, ypred)
    xs0 = xs(ys == 0,:);
    xs1 = xs(ys == 1,:);
    
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
    syms t;
    sigmoid = @(t)(1 / (1 + exp(-t)));
    wzs = arrayfun(sigmoid, (wxs .^ 2) .* w(1) + (wys .^ 2) .* w(2) + ...
        wxs .* w(3) + wys .* w(4) + w(5));    
    surf(wxs, wys, wzs);
    
    % plot predicted points
    %plot3(xpred(:,1), xpred(:,2), ypred, 'go');
    plot(xpred(:,1), xpred(:,2), 'go');
end