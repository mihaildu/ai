% Mihail Dunaev
% May 2017
%
% Simple test that trains a neural net to perform logical XNOR.
% Arch
%   layer 1: x2 x1
%   layer 2: n1 n2
%   layer 3: n3
%
% Stats for train time (old)
%   maxiter 10 = 2.69s
%   maxiter 1000 = 5 min
%   maxiter 5000 = 21min
%
% Initial weights that performed well:
% t0 = [0.1; -1.2; 3.0; 0.5; -0.25; -1; 0.7; 1.0; -2.0];
% t0 = [70.3062; 80.0590; -105.4745; -9.9347; -13.9525;
%    4.6976; 179.9034; 166.0707; -50.9448];

function test7_xnor2()
    addpath('core/');
    
    % read train data
    [xtrain, ytrain] = my_read_data();
    
    % build nn struct
    layers = [2 2 1];
    nn = build_nn(layers);
    
    % load weights from file
    nn = load_weights_mat('../../input/weights_test2_xnor', nn);
    
    % train it (global var)
    %nn = train(xtrain, ytrain, nn);
    
    % store weights to file
    %store_weights_mat('../../input/weights_test2_xnor', nn);
    
    % predict some values
    xpred = [0 0; 0 1; 1 0; 1 1];
    ypred = pred_nn(nn, xpred);
    disp(ypred);
    
    % plot data
    plot_all(xtrain, ytrain, nn);
    
    % plot cost function
    plot_cost();
end

% own read data func since ytrain is not a vector
% in the input file
function [xtrain, ytrain] = my_read_data()
    data = load('../../input/xnor_nn.in');
    xtrain = data(:,1:end-1);
    ytrain = data(:,end);
end

function nn = train(xtrain, ytrain, nn)
    % perform training several times, starting with diff weights
    maxiter = 5;
    cost_threshold = 0.05;
    for k = 1:maxiter
        % set initial weights in a vec (random small values)
        t0 = get_random_thetas(nn.layers, 0.1);
        
        % fminunc with grad
        syms t;
        % possible vals for reg: 0, 0.0001, 0.001
        fun = @(t)(cost_with_grad(t, 0.0001, xtrain, ytrain, nn));
        options = optimset('fminunc');
        options = optimset(options, 'GradObj', 'on', 'TolFun', ...
            1.0000e-10, 'TolX', 1.0000e-10, 'MaxIter', 1000, ...
            'MaxFunEvals', 1000, 'Display', 'off', 'OutputFcn', @outfun);

        %[tf, c] = fminunc(fun, t0, options);
        [tf, c] = fmincg(fun, t0, options);
        if isnan(c(end))
            break;
        end
        if c(end) < cost_threshold
            break;
        end
        plot_cost();
    end
    
    % set nn weights to tf
    nn = unroll_thetas(tf, nn);
end

function plot_all(xtrain, ytrain, nn)
    xtrain0 = xtrain(ytrain == 0,:);
    xtrain1 = xtrain(ytrain == 1,:);
    
    % plot initial data
    figure;
    hold on;
    plot(xtrain0(:,1), xtrain0(:,2), 'bo');
    plot(xtrain1(:,1), xtrain1(:,2), 'rx');
    
    % set range for x/y axis
    xmin = min(xtrain(:,1));
    xmax = max(xtrain(:,1));
    ymin = min(xtrain(:,2));
    ymax = max(xtrain(:,2));
    xlim([xmin-1 xmax+1]);
    ylim([ymin-1 ymax+1]);
    
    % plot 3d fit
    delta = 0.1;
    [wxs, wys] = meshgrid(xmin:delta:xmax, ymin:delta:ymax);
    sz = size(wxs);
    cwxs = reshape(wxs, sz(1)*sz(2), 1);
    cwys = reshape(wys, sz(1)*sz(2), 1);
    cwzs = pred_nn_it(nn, [cwxs cwys]);
    wzs = reshape(cwzs, sz);
    surf(wxs, wys, wzs);
end