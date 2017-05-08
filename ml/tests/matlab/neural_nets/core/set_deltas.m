% Mihail Dunaev
% May 2017

function deltas = set_deltas(layers, deltas, layer, num, d)
    % similar to set_weights
    % this is used internally in backprop
    deltas(layer, 1:(layers(layer)+1), num) = d;
end