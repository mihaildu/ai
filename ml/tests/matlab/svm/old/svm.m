% svm

function main()
    %linear_svm_test1();
    %linear_svm_test2();
    %kernel_svm_test3();
end

% test for non-linear decision boundary
function kernel_svm_test3()
    read_data('../input/logistic_circle.in');
    train_test3();
    predict_test3()
    plot_data_test3();
end

function ys = pred_test3(xs)
    global w;
    npred = size(xs, 1);
    ys = zeros(npred, 1);
    for i = 1:npred
        f = similarity_vec_test3(xs(i,:));
        ys(i) = (f * w >= 0);
    end
end

function predict_test3()
    global xpred ypred;
    xpred = [3.5 4; 4 4.1; 2 6];
    ypred = pred_test3(xpred);
end

function train_test3()
    global w N;
    
    % random initial weights
    eps_theta = 0.1;
    t0 = rand(N+1, 1) * (2 * eps_theta) - eps_theta;
    
    options = optimset('fminunc');
    %options = optimset(options, 'GradObj', 'on');
    [tf,c] = fminunc(@cost_with_grad_test3, t0, options);
    w = tf;
end

function [cost, grad] = cost_with_grad_test3(t)
    global xs1 xs0 n1 n0 N;
    
    % compute similarity for each training point
    % prob better way to do this
    mat1 = zeros(n1, N+1);
    for k = 1:n1
        mat1(k,:) = similarity_vec_test3(xs1(k,:));
    end
    mat0 = zeros(n0, N+1);
    for k = 1:n0
        mat0(k,:) = similarity_vec_test3(xs0(k,:));
    end
    
    %C = 100;
    cost = sum(max((1 - mat1 * t) * exp(1) / 2, 0)) + ...
        sum(max((1 + mat0 * t) * exp(1) / 2, 0));
    %cost = C * cost + sum(t .^ 2) / 2;

    % TODO grad
    if nargout > 1
        grad = zeros(size(t));
    end
end

% returns [fm(x) ... f2(x) f1(x) 1]
function v = similarity_vec_test3(x)
    global xs N;
    s = 0.5;
    v = ones(1, N+1);
    for i = 1:N
        v(i) = similarity_test3(xs(i,:), x, s);
    end
end

function r = similarity_test3(x1, x2, s)
    r = exp(-sum((x1 - x2) .^ 2) / (2 * (s ^ 2)));
end

function plot_similarity_test3()
    % plot similarity around each point
    global xs N;
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
            cwzs(i,1) = similarity_test3(xs(k,:), [cwxs(i) cwys(i)], 1);
        end
        wzs = reshape(cwzs, sz);
        surf(wxs, wys, wzs, 'EdgeColor', 'none');
    end
    alpha 0.1;
end

function plot_data_test3()
    global xs0 xs1 xs xpred ypred;

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

% test with 2 features
function linear_svm_test2()
    read_data('../input/logistic.in');
    train_test2();
    plot_data_test2();
end

function train_test2()
    global w;
    t0 = [1 -1 0.5]';
    %check_grad(@cost_with_grad_test2, t0);
    %return;

    options = optimset('fminunc');
    %options = optimset(options, 'GradObj', 'on');
    [tf,~] = fminunc(@cost_with_grad_test2, t0, options);
    w = tf;
end

function [cost, grad] = cost_with_grad_test2(t)
    global xs1 xs0 n1 n0;
    C = 100;
    cost = sum(max((1 - [xs1 ones(n1,1)] * t) * exp(1) / 2, 0)) + ...
        sum(max((1 + [xs0 ones(n0,1)] * t) * exp(1) / 2, 0));
    cost = C * cost + sum(t .^ 2) / 2;

    % lazy implementation for grad
    % TODO
    if nargout > 1
        grad = zeros(size(t));
    end
end

function plot_data_test2()
    global xs0 xs1 xs w;

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
    %y0normal = [-(w(3)+w(1)*xmin)/w(2); -(w(3)+w(1)*xmax)/w(2)];
    plot(x0, y0plus, 'r-');
    plot(x0, y0minus, 'b-');
    %plot(x0, y0normal, 'g-');
end

% test with 1 feature
function linear_svm_test1()
    % read data
    read_data('../input/tumors.in');

    % train params
    train_test1();
    
    % for pred just look at theta * x >= 0
    % instead of +/-1
    predict_test1();
    
    % plot data
    plot_data_test1();
end

function predict_test1()
    global xpred ypred;
    xpred = [7 43 44 70]';
    ypred = pred_test1(xpred);
end

function ys = pred_test1(xs)
    global w;
    npred = size(xs, 1);
    ys = [xs ones(npred,1)] * w >= 0;
end

function train_test1()
    global w;
    t0 = [1 0.5]';
    %check_grad(@cost_with_grad_test1, t0);
    %return;
    
    options = optimset('fminunc');
    %options = optimset(options, 'GradObj', 'on');
    [tf,~] = fminunc(@cost_with_grad_test1, t0, options);
    w = tf;
end

function [cost, grad] = cost_with_grad_test1(t)
    global xs1 xs0 n1 n0 N xs ys;
    C = 100;
    cost = sum(max((1 - [xs1 ones(n1,1)] * t) * exp(1) / 2, 0)) + ...
        sum(max((1 + [xs0 ones(n0,1)] * t) * exp(1) / 2, 0));
    cost = C * cost + sum(t .^ 2) / 2;

    % lazy implementation for grad
    % TODO it shows ok on check_grad but doesn't work for minimization
    % TODO check sub-gradient
    if nargout > 1
        grad = zeros(size(t));
        for i = 1:N
            if ys(i) == 0
                if [xs(i) 1] * t >= -1
                    %grad = grad + [(xs(i) ^ 2) * exp(1) / 2; exp(1) / 2];
                    grad = grad + [xs(i) * exp(1) / 2; exp(1) / 2];
                end
            elseif [xs(i) 1] * t <= 1
                grad = grad + [-xs(i) * exp(1) / 2; -exp(1) / 2];
            end
        end
        grad = C * grad + t;
    end
end

function plot_data_test1()
    global xs0 xs1 xs w xpred ypred;
    
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

% compares gradient returned by fgrad with numerical one
function check_grad(fgrad, t0)
    % e.g. usage: check_grad(@cost_with_grad, t0);
    [~, grad] = fgrad(t0);
    sz = size(t0, 1);
    num_grad = zeros(sz, 1);
    eps = 0.0001;
    for i = 1:sz
        t0(i) = t0(i) + eps;
        [fp, ~] = fgrad(t0);
        t0(i) = t0(i) - (2 * eps);
        [fm, ~] = fgrad(t0);
        t0(i) = t0(i) + eps;
        num_grad(i) = (fp - fm) / (2 * eps);
    end
    disp(num_grad);
    disp(grad);
    disp(sum((num_grad - grad) .^ 2));
end

function read_data(fname)
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