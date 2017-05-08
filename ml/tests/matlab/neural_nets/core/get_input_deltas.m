% Mihail Dunaev
% May 2017

function d = get_input_deltas(layers, deltas, layer, num)
    % similar to get_input_weights
    % this is used internally in backprop
    d = deltas(layer, 1:(layers(layer)+1), num)';
end