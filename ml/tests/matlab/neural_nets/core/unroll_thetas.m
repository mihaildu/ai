% Mihail Dunaev
% May 2017

function nn = unroll_thetas(t, nn)
    % unrolls weights from array to cube
    % input
    %   t = new weights in a single column vector
    %   nn = neural net
    % output
    %   nn = neural net with the new weights from t
    last_index = 1;
    for i = 1:(nn.num_layers - 1)
        % num_weights is actually num_weights - 1
        num_weights = nn.layers(i);
        for j = 1:nn.layers(i+1)
            weights = t(last_index:last_index + num_weights);
            nn = set_weights(nn, i, j, weights);
            last_index = last_index + num_weights + 1;
        end
    end
end