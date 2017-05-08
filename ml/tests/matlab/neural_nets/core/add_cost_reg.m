% Mihail Dunaev
% May 2017

function cost = add_cost_reg(cost, lambda, nn, Ntrain)
    % add regularization term to cost function
    % TODO remove reg for bias weights
    s = 0;
    for i = 1:(nn.num_layers - 1)
        for j = 1:nn.layers(i+1)
            w = get_input_weights(nn, i, j);
            s = s + sum(w(1:end-1) .^ 2);
        end
    end
    cost = cost + (lambda * s) / (2 * Ntrain);
end