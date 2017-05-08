% Mihail Dunaev
% May 2017

function [cost, grad] = cost_with_grad(t, lambda, xtrain, ytrain, nn)
    % cost function and gradient for neural net
    % input
    %   t = weights values as a single column vector
    %   lambda = regularization param
    %   xtrain, ytrain = training set
    %   nn = neural net
    % output
    %   cost & gradient for neural net
    %
    % this should be faster if we precompute xtrain0/1
    Ntrain = size(xtrain, 1);
    nn = unroll_thetas(t, nn);
    ypred = zeros(size(ytrain));
    deltas = zeros(size(nn.W));
    sz = size(xtrain, 1);
    for k = 1:sz
        nn = compute_val(nn, xtrain(k,:));

        % compute error on last layer
        layer = nn.num_layers - 1;
        num_nodes = nn.layers(layer + 1);
        ypred(k,:) = nn.val(layer + 1, 1:num_nodes);

        % skip backprop if we only compute cost
        if nargout <= 1
            continue;
        end
        nn.err(layer, 1:num_nodes) = ypred(k,:) - ytrain(k, :);

        % backprop error
        layer = layer - 1;
        while layer > 0
            num_nodes = nn.layers(layer+1);
            next_num_nodes = nn.layers(layer+2);
            % TODO this can be vectorized
            for num = 1:num_nodes
                w = get_output_weights(nn, layer, num);
                val = nn.val(layer+1,num);
                val = val * (1 - val);
                nn.err(layer, num) = nn.err(layer+1, 1:next_num_nodes) * w * val;
            end
            layer = layer - 1;
        end
        
        % update deltas
        for layer = 1:(nn.num_layers - 1)
            % num_weights is actually num_weights - 1
            for j = 1:nn.layers(layer+1)
                d = get_input_deltas(nn.layers, deltas, layer, j);
                vals = [nn.val(layer, 1:nn.layers(layer)) 1]';
                d = d + vals * nn.err(layer, j);
                deltas = set_deltas(nn.layers, deltas, layer, j, d);
            end
        end
    end

    % compute cost function
    cost = -(sum(sum(ytrain .* log(ypred))) + ...
        sum(sum((1 - ytrain) .* log(1 - ypred)))) / Ntrain;
    if lambda > 0
        cost = add_cost_reg(cost, lambda, nn, Ntrain);
    end
    if nargout <= 1
        return;
    end

    % roll deltas into grad
    k = 1;
    grad = zeros(size(t));
    for layer = 1:(nn.num_layers - 1)
        for j = 1:nn.layers(layer+1)
                d = get_input_deltas(nn.layers, deltas, layer, j) / Ntrain;
                ek = k + size(d, 1) - 1;
                grad(k:ek) = d;
                k = k + size(d, 1);
        end
    end
    if lambda > 0
        grad = add_grad_reg(grad, lambda, nn);
    end
end