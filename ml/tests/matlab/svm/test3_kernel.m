% Mihail Dunaev
% May 2017
%
% Simple test that uses kernel SVM to separate between 2D data
% points (non-linear decision boundary). Data in 'logistic_circle.in'.

function test3_kernel()
    addpath('core/');
    [xs, ys] = read_data('../../input/logistic_circle.in');

    % random initial weights
    N = size(xs, 1);
    eps_theta = 0.1;
    w0 = rand(N+1, 1) * (2 * eps_theta) - eps_theta;
    
    syms t;
    C = 100;
    fun = @(t)(cost_with_grad_kernel(t, xs, ys, C));
    w = train(w0, fun);

    % predict some values
    xpred = [3.5 4; 4 4.1; 2 6];
    ypred = predict_kernel(xpred, w, xs);
    
    %figure;
    %hold on;
    %plot_kernels(xs);
    plot_all(xs, ys, xpred, ypred);
end

% plot similarity around each point
function plot_kernels(xs)
    N = size(xs, 1);
    delta = 0.1;
    for k = 1:N
        xmin = xs(k,1) - 1;
        xmax = xs(k,1) + 1;
        ymin = xs(k,2) - 1;
        ymax = xs(k,2) + 1;
        [wxs,wys] = meshgrid(xmin:delta:xmax, ymin:delta:ymax);
        sz = size(wxs);
        total_sz = sz(1)*sz(2);
        cwxs = reshape(wxs, total_sz, 1);
        cwys = reshape(wys, total_sz, 1);
        cwzs = zeros(total_sz, 1);
        for i = 1:total_sz
            cwzs(i,1) = kernel(xs(k,:), [cwxs(i) cwys(i)], 1);
        end
        wzs = reshape(cwzs, sz);
        surf(wxs, wys, wzs, 'EdgeColor', 'none');
    end
    alpha 0.1;
end

function plot_all(xs, ys, xpred, ypred)
    xs0 = xs(ys == 0,:);
    xs1 = xs(ys == 1,:);

    % plot initial data
    figure;
    hold on;
    
    %plot_similarity_test3();
    
    plot(xs0(:,1), xs0(:,2), 'bo');
    plot(xs1(:,1), xs1(:,2), 'rx');

    % set range for x/y axis
    xmin = min(xs(:,1));
    xmax = max(xs(:,1));
    ymin = min(xs(:,2));
    ymax = max(xs(:,2));
    xlim([xmin-1 xmax+1]);
    ylim([ymin-1 ymax+1]);
    
    % plot predicted points
    for i = 1:size(xpred, 1)
        if ypred(i) == 1
            plot(xpred(i,1), xpred(i,2), 'r+');
        elseif ypred(i) == 0
            plot(xpred(i,1), xpred(i,2), 'b+');
        else
            plot(xpred(i,1), xpred(i,2), 'k+');
        end
    end
end