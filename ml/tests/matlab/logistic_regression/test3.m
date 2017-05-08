% Mihail Dunaev
% May 2017
%
% Simple test that tries to separate 2 sets of 2D data on a plane
%   using linear logistic regression. Data is in 'logistic.in'.
%
% Results
%   fminsearch w = 1.0e+03 * [0.4683, -0.1113, -1.1665], cost = 0 
%   fminunc w = 16.3325, 2.6081, -67.5492, cost = 7.4372e-06
%   fminunc with grad w = 8.6056, 4.0303, -46.8913, cost = 4.7543e-05,
%   own gradient descent w = 5.3367, 2.3259, -28.1962, cost = 0.00494
%                       (alpha = 3)

function test3()
    addpath('core/');
    [xs, ys] = read_data('../../input/logistic.in');

    syms t;
    w0 = [1; -1; 1];
    fun = @(t)(cost_with_grad(t, xs, ys));
    w = train(w0, fun);
    
    xpred = [3.5 4; 4 4.1; 2 6];
    ypred = predict(xpred, w);
    plot_all(xs, ys, w, xpred, ypred);
    %plot_cost();
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
    
    % plot separating line
    x0 = [xmin; xmax];
    y0 = [-(w(3)+(w(1) * xmin))/w(2); -(w(3)+(w(1) * xmax))/w(2)];
    plot(x0, y0);
    
    % plot best fit (3d)
    delta = 0.1;
    [wxs,wys] = meshgrid(xmin:delta:xmax, ymin:delta:ymax);
    syms t;
    sigmoid = @(t)(1 / (1 + exp(-t)));
    wzs = arrayfun(sigmoid, wxs .* w(1) + wys .* w(2) + w(3));
    surf(wxs, wys, wzs);
    
    % plot predicted points
    %plot3(xpred(:,1), xpred(:,2), ypred, 'go');
    plot(xpred(:,1), xpred(:,2), 'go');
end