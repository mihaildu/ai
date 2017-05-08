% Mihail Dunaev
% May 2017
%
% Simple test that builds an OR neural net 
% and predicts some values.
%   layer 1: x2 x1
%   layer 2: n1 = output

function test4_or()
    addpath('core/');
    layers = [2 1];
    nn = build_nn(layers);
        
    % set weights
    w = [20 20 -10]';
    nn = set_weights(nn, 1, 1, w);
    
    % pred some values
    xpred = [0 0; 0 1; 1 0; 1 1];
    ypred = pred_nn(nn, xpred);
    disp(ypred);
end