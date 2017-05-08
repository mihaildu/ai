% Mihail Dunaev
% May 2017

function t = roll_thetas(nn)
    % rolls weights (thetas) from nn (cube) to vec
    % input
    %   nn = neural net (from build_nn)
    % output
    %   t = weights in a single column vector
    t = [];
    for i = 1:(nn.num_layers - 1)
        for j = 1:nn.layers(i+1)
            weights = get_input_weights(nn, i, j);
            t = [t; weights];
        end
    end
end