% Mihail Dunaev
% May 2017

function val = get_value(nn, xs, layer, num)
    % returns value from node num from specified layer
    % with xs as inputs
    if layer == 1
        val = xs(num);
        return;
    end
    vals = ones(1, nn.layers(layer-1) + 1);
    for i = 1:nn.layers(layer-1)
        vals(i) = get_value(nn, xs, layer-1, i);
    end
    w = get_input_weights(nn, layer-1, num);
    f = get_function(nn, layer-1, num);
    val = f(vals * w);
end