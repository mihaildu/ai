% Mihail Dunaev
% May 2017

function nn = set_weights(nn, layer, num, w)
    % sets input weights w for node in neural net at (layer, num)
    % 
    % input
    %   nn = neural network (from build_nn)
    %   layer = layer of the node; counting starts after features
    %           e.g. layer 1 is the first hidden layer/output layer
    %   num = node number in that layer; counting starts at the top
    %   w = column vector that represents the new weights
    %
    % output
    %   nn = neural net with the new weights
    
    % TODO sanity checks on nn, layer, num & w
    
    % nn.W(layer,:,:) = matrix for our layer
    % nn.W(layer,:,num) = weight vector for our node
    % weight vector has only `nn.layers(layer) + 1` values
    % nn.W(layer,:,num) can have more than that, so we only
    % take the first `nn.layers(layer) + 1`
    nn.W(layer, 1:(nn.layers(layer)+1), num) = w;
end