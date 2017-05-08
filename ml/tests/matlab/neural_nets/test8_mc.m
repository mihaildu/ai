% Mihail Dunaev
% May 2017
%
% Simple test that trains a neural net to separate between
%   4 classes of points in 2D. Data is in 'xnor_mc_nn.in'.
%
% Arch
%   2, 2, 4

function test8_mc()
    addpath('core/');
    layers = [2 2 4];
    %layers = [2 4];
    nn = build_nn(layers);
    
    % read train data
    [xtrain, ytrain] = read_data(layers, '../../input/xnor_mc_nn.in');
    
    % train the nn
    nn = train(xtrain, ytrain, nn);
    
    % predict some values
    %xpred = [0 0; 0 1; 1 0; 1 1];
    xpred = [0.1 0.1; -0.1 0.9; 0.9 0.2; 0.8 1.1; 0.4 0.5; 0.7 0.25;
        0.25 0.55; 0.5 0.5];
    ypred = pred_nn(nn, xpred);
    %disp(ypred);
    
    % class 1, 2, 3 & 4
    % [1 0 0 0] = class 1 / [0, 0]
    % [0 1 0 0] = class 2 / [1, 1]
    % [0 0 1 0] = class 3 / [0, 1]
    % [0 0 0 1] = class 4 / [1, 0]
    for i = 1:size(ypred, 1)
        [~, c] = max(ypred(i,:));
        % maybe do something if max < 0.5
        fprintf('[%f %f] has class %d\n', xpred(i,:), c);
    end
    
    % plot data
    plot_all(xtrain, ytrain, xpred, ypred);
    
    % plot evolution of cost function
    plot_cost();
end

function nn = train(xtrain, ytrain, nn)
    % set initial weights in a vec (random small values)
    t0 = get_random_thetas(nn.layers, 0.1);
    
    % fminsearch
    %[tf,~] = fminsearch(@cost_with_grad, t0);
    
    % fminunc with grad
    fun = @(t)(cost_with_grad(t, 0.0001, xtrain, ytrain, nn));
    options = optimset('fminunc');
    options = optimset(options, 'GradObj', 'on', 'Display', 'off', ...
        'OutputFcn', @outfun);
    [tf, ~] = fminunc(fun, t0, options);
    %[tf,~] = fmincg(fun, t0, options);

    % set nn weights to tf
    nn = unroll_thetas(tf, nn);
end

function plot_all(xtrain, ytrain, xpred, ypred)
    % plot initial data
    figure;
    hold on;
    Ntrain = size(xtrain, 1);
    for i = 1:Ntrain
        if ytrain(i,1) == 1
            plot(xtrain(i,1), xtrain(i,2), 'bo');
        elseif ytrain(i,2) == 1
            plot(xtrain(i,1), xtrain(i,2), 'rs');
        elseif ytrain(i,3) == 1
            plot(xtrain(i,1), xtrain(i,2), 'gd');
        elseif ytrain(i,4) == 1
            plot(xtrain(i,1), xtrain(i,2), 'y^');
        else
            plot(xtrain(i,1), xtrain(i,2), 'kx');
        end
    end

    % set range for x/y axis
    xmin = min(xtrain(:,1));
    xmax = max(xtrain(:,1));
    ymin = min(xtrain(:,2));
    ymax = max(xtrain(:,2));
    xlim([xmin-1 xmax+1]);
    ylim([ymin-1 ymax+1]);
    
    % plot predicted values
    [~, c] = max(ypred, [], 2);
    for i = 1:size(ypred, 1)
        if c(i) == 1
            plot(xpred(i,1), xpred(i,2), 'bx');
        elseif c(i) == 2
            plot(xpred(i,1), xpred(i,2), 'rx');
        elseif c(i) == 3
            plot(xpred(i,1), xpred(i,2), 'gx');
        elseif c(i) == 4
            plot(xpred(i,1), xpred(i,2), 'yx');
        else
            plot(xpred(i,1), xpred(i,2), 'kx');
        end
    end
end