% Mihail Dunaev
% May 2017
%
% Simple test that uses linear SVM to separate between 1D data
% points. Data in 'tumors.in'.


function test1_linear()
    addpath('core/');
    [xs, ys] = read_data('../../input/tumors.in');
    
    % train params
    syms t;
    w0 = [1 0.5]';
    C = 100;
    fun = @(t)(cost_with_grad_linear(t, xs, ys, C));
    w = train(w0, fun);
    
    % TODO check gradient
    %check_grad(@cost_with_grad, w0);
    
    % predict some values
    xpred = [7 43 44 70]';
    ypred = predict_linear(xpred, w);
    
    % plot everything
    plot_all(xs, ys, w, xpred, ypred);
end

function plot_all(xs, ys, w, xpred, ypred)
    xs0 = xs(ys == 0,:);
    xs1 = xs(ys == 1,:);
    
    % plot initial data
    figure;
    hold on;
    plot(xs0, 0, 'bo');
    plot(xs1, 1, 'rx');
    
    % set range for x/y axis
    xmin = min(xs);
    xmax = max(xs);
    xlim([xmin-1 xmax+1]);
    ylim([-1.0 1.5]);
    
    % plot best fit
    delta = 0.1;
    num_pts = (xmax - xmin) / delta + 1;
    wxs = (xmin:delta:xmax)';
    wys = [wxs ones(num_pts,1)] * w;
    plot(wxs, wys, 'k-');
    
    % plot xplus/minus
    xplus = (1 - w(2)) / w(1);
    xminus = -(1 + w(2)) / w(1);
    plot(xplus, 1, 'g+');
    plot(xminus, 0, 'g+');
    
    % plot predicted points
    plot(xpred, ypred, 'go');
end