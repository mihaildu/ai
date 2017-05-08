% Mihail Dunaev
% May 2017
%
% Simple test that builds an XNOR neural net 
% and predicts some values.
%   layer 1: x2 x1
%   layer 2: n1 n2
%   layer 3: n3
%
% XNOR values
%   0 xnor 0 = 1
%   0 xnor 1 = 0
%   1 xnor 0 = 0
%   1 xnor 1 = 1

function test6_xnor()
    addpath('core/');
    layers = [2 2 1];
    nn = build_nn(layers);
    
    % set weights
    w = [20 20 -30]';
    nn = set_weights(nn, 1, 1, w);
    w = [-20 -20 10]';
    nn = set_weights(nn, 1, 2, w);
    w = [20 20 -10]';
    nn = set_weights(nn, 2, 1, w);
    
    % pred some values
    xpred = [0 0; 0 1; 1 0; 1 1];
    ypred = pred_nn(nn, xpred);
    disp(ypred);
end

