% Mihail Dunaev
% May 2017

function w = get_input_weights(nn, layer, num)
    % returns input weights from node at (layer, num)
    % input
    %   nn = neural network (from build_nn_generic)
    %   layer = layer of the node; counting starts after features
    %           e.g. layer 1 is the first hidden layer/output layer
    %   num = node number in that layer; counting starts at the top
    % output
    %   w = weights that go inside node from (layer,num)
    %           as a column vector
    
    % TODO sanity checks on nn, layer & num
    
    % nn.W(layer,:,:) = matrix for our layer
    % nn.W(layer,:,num) = weight vector for our node
    % weight vector has only `nn.layers(layer) + 1` values
    % nn.W(layer,:,num) can have more than that, so we only
    % take the first `nn.layers(layer) + 1`
    w = nn.W(layer, 1:(nn.layers(layer)+1), num)';
end