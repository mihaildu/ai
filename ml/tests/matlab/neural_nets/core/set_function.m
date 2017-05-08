% Mihail Dunaev
% May 2017

function nn = set_function(nn, layer, num, fname)
    % sets activation function with fname for node
    % in neural net at (layer, num)
    %
    % input
    %   nn = neural network (from build_nn)
    %   layer = layer of the node; counting starts after features
    %           e.g. layer 1 is the first hidden layer/output layer
    %   num = node number in that layer; counting starts at the top
    %   fname = string representing the new activation function for
    %           that node
    %       possible values:
    %           'sigmoid' = 1/(1 + exp(-x))
    %
    % output
    %   nn = neural net with the new functions
    
    % TODO sanity checks on nn, layer, num & f
    
    % nn.F(layer, num) = function for the node
    if nn.allowed_functions.isKey(fname)
        nn.F(layer, num) = {fname};
    else
        nn.F(layer, num) = {nn.default_function};
    end
end