% Mihail Dunaev
% May 2017
%
% Simple test that uses a neural network as logistic regression.
%
% Arch
%   1, 1

function test9_lr()
    addpath('core/');
    layers = [1 1];
    nn = build_nn(layers);
    
    % read training data
    [xtrain, ytrain] = read_data(layers, '../../input/tumors2.in');
    
    % train network
    nn = train(xtrain, ytrain, nn);

    % predict values
    xpred = [7 43 44 70]';
    ypred = pred_nn(nn, xpred);
    
    % plot everything
    plot_all(xtrain, ytrain, nn, xpred, ypred);
    
    % plot cost function too
    plot_cost();
end

function nn = train(xtrain, ytrain, nn)
    % set initial weights in a vec (random small values)
    t0 = get_random_thetas(nn.layers, 0.1);

    % fminsearch
    %[tf,~] = fminsearch(@cost_with_grad, t0);

    % fminunc with grad
    fun = @(t)(cost_with_grad(t, 0, xtrain, ytrain, nn));
    options = optimset('fminunc');
    options = optimset(options, 'GradObj', 'on', 'Display', 'off', ...
        'OutputFcn', @outfun);
    [tf, ~] = fminunc(fun, t0, options);
    %[tf, ~] = fmincg(fun, t0, options);

    % set nn weights to tf
    nn = unroll_thetas(tf, nn);
end

function plot_all(xtrain, ytrain, nn, xpred, ypred)
    xs0 = xtrain(ytrain == 0);
    xs1 = xtrain(ytrain == 1);
    
    figure;
    hold on;
    plot(xs0, 0, 'bo');
    plot(xs1, 1, 'rx');
    
    % set range for x/y axis
    xmin = min(xtrain);
    xmax = max(xtrain);
    xlim([xmin-1 xmax+1]);
    ylim([-0.5 1.5]);
    
    % plot best fit
    delta = 0.1;
    wxs = (xmin:delta:xmax)';
    wys = pred_nn(nn, wxs);
    plot(wxs, wys, 'k-');
    
    % plot predicted points
    for k = 1:size(xpred, 1)
        if ypred(k) >= 0.5
            plot(xpred(k), 1, 'go');
        else
            plot(xpred(k), 0, 'go');
        end
    end
    %plot(xpred, ypred, 'go');
end