% Mihail Dunaev
% May 2017
%
% Simple test that builds a neural net and tests some functions.
% Neural net details
%   layer 1: x2 x1 + bias x0 (2 features) - input
%   layer 2: node 1 (n1), node 2 (n2) - hidden
%   layer 3: node 3 (n3), node 4 (n4) - output

function test2()
    addpath('core/');

    % build nn from layers
    layers = [2 2 2];
    nn = build_nn(layers);
    
    % set the weights
    % w(1)*x3 + w(2)*x2 + w(3)*x1 + w(4) etc
    w = [1 2 3]';
    nn = set_weights(nn, 1, 1, w);
    w = [4 5 6]';
    nn = set_weights(nn, 1, 2, w);
    w = [7 8 9]';
    nn = set_weights(nn, 2, 1, w);
    w = [10 11 12]';
    nn = set_weights(nn, 2, 2, w);
    
    % get output weights
    w = get_output_weights(nn, 1, 2);
    disp(w);
end