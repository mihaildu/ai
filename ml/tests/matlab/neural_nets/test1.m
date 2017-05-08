% Mihail Dunaev
% May 2017
%
% Simple test that builds a neural net and predicts some values.
% Neural net details
%   layer 1: x3 x2 x1 + bias x0 (3 features) - input
%   layer 2: node 1 (n1), node 2 (n2) - hidden
%   layer 3: node 3 (n3) - output

function test1()
    addpath('core/');
    
    % build nn from layers
    layers = [3 2 1];
    nn = build_nn(layers);
    
    % set the weights
    % w(1)*x3 + w(2)*x2 + w(3)*x1 + w(4) etc
    w = [0 2 4 5]';
    nn = set_weights(nn, 1, 1, w);
    w = [-2 1 -1 0.3]';
    nn = set_weights(nn, 1, 2, w);
    w = [-2 1 3]';
    nn = set_weights(nn, 2, 1, w);
    display_weights(nn);
    
    % predict some inputs
    xpred = [1.5 2 3.3; -1 0.5 -2];
    ypred = pred_nn(nn, xpred);
    disp(ypred);
end