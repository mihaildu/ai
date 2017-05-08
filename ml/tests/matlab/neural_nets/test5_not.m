% Mihail Dunaev
% May 2017
%
% Simple test that builds a NOT neural net 
% and predicts some values.
%   layer 1: x1
%   layer 2: n1 = output

function test5_not()
    addpath('core/');
    layers = [1 1];
    nn = build_nn(layers);
    
    % set weights
    w = [-20 10]';
    nn = set_weights(nn, 1, 1, w);
    
    % pred some values
    xpred = [0; 1];
    ypred = pred_nn(nn, xpred);
    disp(ypred);
end

