% Mihail Dunaev
% May 2017

function f = get_function(nn, layer, num)
    % returns activation function from node at (layer, num)
    % input
    %   nn = neural network (from build_nn)
    %   layer = layer of the node; counting starts after features
    %           e.g. layer 1 is the first hidden layer/output layer
    %   num = node number in that layer; counting starts at the top
    
    % TODO sanity checks on nn, layer & num
    
    % nn.F(layer, num) = function for the node
    fname = char(nn.F(layer, num));
    if nn.fmap.isKey(fname)
        f = nn.fmap(fname);
    else
        f = nn.fmap(nn.default_function);
    end
end