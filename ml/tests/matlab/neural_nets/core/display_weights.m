% Mihail Dunaev
% May 2017

function display_weights(nn)
    % shows all weights for each node of nn on a new line
    % starts from left layer, top node
    for i = 1:(nn.num_layers - 1)
        for j = 1:nn.layers(i+1)
            w = get_input_weights(nn, i, j);
            disp(w');
        end
    end
end