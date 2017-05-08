% Mihail Dunaev
% May 2017
%
% Simple test that tries to predict house prices using linear
%   regression. There is only one feature (house size) and data
%   is stored in 'houses.in' in input dir.
%
% Results
%   w train_direct = 10.8619, -22.2128, 1.6422e-04s
%   w train_cost fminsearch = 10.8619, -22.2128, 0.0036s
%   w train_cost fminunc = 10.8619, -22.2128, 0.0082s
%   w train_cost fminunc with grad = 10.8619, -22.2128, 0.0061s
%
%   min cost reached = 353.3414
%
%   train time with own gradient descent = 1.3612s

function test1()
    addpath('core/');
    [xs, ys] = read_data('../../input/houses.in');
    direct_method = false;
    w = train(xs, ys, direct_method);
    
    plot_all(xs, ys, w);
    
    % predict some values
    xpred = [12 45]';
    ypred = predict(xpred, w);
    disp(ypred);
end

function plot_all(xs, ys, w)
    % plot initial data
    figure;
    hold on;
    plot(xs, ys, 'bo');
    
    % plot best fit
    delta = 0.1;
    xmin = min(xs);
    xmax = max(xs);
    num_pts = (xmax - xmin) / delta + 1;
    wxs = (xmin:delta:xmax)';
    wys = [wxs ones(num_pts,1)] * w;
    plot(wxs, wys, 'r-');
end