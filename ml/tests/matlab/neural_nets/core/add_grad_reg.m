% Mihail Dunaev
% May 2017

function grad = add_grad_reg(grad, lambda, nn)
    % add regularization term to gradient
    % TODO no reg for bias weights
    gstart = 1;
    for i = 1:(nn.num_layers - 1)
        dindex = nn.layers(i);
        for j = 1:nn.layers(i+1)
            w = get_input_weights(nn, i, j);
            gend = gstart + dindex;
            grad(gstart:gend) = grad(gstart:gend) + lambda * w;
            gstart = gend + 1;
        end
    end
end