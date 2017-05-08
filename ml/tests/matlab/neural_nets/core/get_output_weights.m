% Mihail Dunaev
% May 2017

function w = get_output_weights(nn, layer, num)
    % returns output weights from node at (layer, num)
    % input
    %   nn = neural network (from build_nn_generic)
    %   layer = layer of the node; counting starts after features
    %           e.g. layer 1 is the first hidden layer/output layer
    %   num = node number in that layer; counting starts at the top
    % output
    %   w = weights that go outside the node at (layer,num)
    %           as a column vector
    
    % TODO sanity checks on nn, layer & num
    
    % nn.W(layer+1,:,:) = matrix for next layer
    % nn.W(layer,num,:) = weights that go outside from the node
    % weight vector has only `nn.layers(layer+2)` values
    % nn.W(layer,num,:) can have more than that, so we only
    % take the first `nn.layers(layer+2)`
    w = squeeze(nn.W(layer + 1, num, 1:nn.layers(layer+2)));
end