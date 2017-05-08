% Mihail Dunaev
% May 2017
%
% Simple test that tries to predict if a tumor is malignant (y = 1)
%   or benign (y = 0) based on the tumor size (one feature). It uses
%   linear logistic regression to do that. Data is in 'tumors2.in' in
%   input dir.
%
% Results
%   threshold value for x (x0) is at -w(2)/w(1)
%   fminsearch w = 0.1425, -8.6982, cost = 4.399, x0 = 61.04
%   fminunc w = 0.1425, -8.6982, cost = 4.399, x0 = 61.04
%   fminunc with grad w = 0.1425, -8.6982, cost = 4.399, x0 = 61.04
%   own gradient descent w = 0.1436, -8.7625, cost = 4.3991,
%                        x0 = 61.0202 (alpha = 3)

function test2_tumors()
    addpath('core/');
    [xs, ys] = read_data('../../input/tumors2.in');

    syms t;
    w0 = [0; -5];
    fun = @(t)(cost_with_grad(t, xs, ys));
    w = train(w0, fun);

    % predict some values
    xpred = [7 43 44 70]';
    ypred = predict(xpred, w);
    plot_all(xs, ys, w, xpred, ypred);
    %plot_cost();
end

function plot_all(xs, ys, w, xpred, ypred)
    xs0 = xs(ys == 0,:);
    xs1 = xs(ys == 1,:);
    syms t;
    sigmoid = @(t)(1 / (1 + exp(-t)));
    
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
    wys = arrayfun(sigmoid, [wxs ones(num_pts, 1)] * w);
    plot(wxs, wys, 'k-');
    
    % plot predicted points
    plot(xpred, ypred, 'go');
end