% Mihail Dunaev
% May 2017
%
% Simple test that uses linear SVM to separate between 2D data
% points. Data in 'logistic.in'.

function test2_linear()
    addpath('core/');
    [xs, ys] = read_data('../../input/logistic.in');
    
    % train params
    syms t;
    w0 = [1 -1 0.5]';
    C = 100;
    fun = @(t)(cost_with_grad_linear(t, xs, ys, C));
    w = train(w0, fun);
    
    plot_all(xs, ys, w);
end

function plot_all(xs, ys, w)
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
    
    % plot separating lines
    x0 = [xmin; xmax];
    y0plus = [(1-(w(3)+w(1)*xmin))/w(2); (1-(w(3)+w(1)*xmax))/w(2)];
    y0minus = [(-1-(w(3)+w(1)*xmin))/w(2); (-1-(w(3)+w(1)*xmax))/w(2)];
    y0normal = [-(w(3)+w(1)*xmin)/w(2); -(w(3)+w(1)*xmax)/w(2)];
    plot(x0, y0plus, 'r-');
    plot(x0, y0minus, 'b-');
    plot(x0, y0normal, 'g-');
end