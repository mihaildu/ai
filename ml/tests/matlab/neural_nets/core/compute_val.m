% Mihail Dunaev
% May 2017

function nn = compute_val(nn, xpred)
    % computes val matrix and stores it inside nn
    % xpred is a single input (one line of features)
    nn.val(1,1:nn.layers(1)) = xpred;
    for i = 2:nn.num_layers
        for j = 1:nn.layers(i)
            w = get_input_weights(nn, i-1, j);
            f = get_function(nn, i-1, j);
            vals = [nn.val(i-1,1:nn.layers(i-1)) 1];
            nn.val(i,j) = f(vals * w);
        end
    end
end