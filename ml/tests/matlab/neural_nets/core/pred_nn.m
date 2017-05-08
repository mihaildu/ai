% Mihail Dunaev
% May 2017

function ypred = pred_nn(nn, xpred)
    % returns predicted values for xpred (recursive version)
    % use 'pred_nn_it' instead of this for multiple output units
    % this recomputes values
    % input
    %   nn = neural net (from build_nn)
    %   xpred = feature values to be predicted
    %       e.g. if we have 2 feature as inputs xpred will look like this
    %       xpred = [1.5 2; -1 0.5; ...];
    % output
    %   ypred = predicted values for each example, as a vector
    %       e.g. if we have 3 output units (3 classes)
    %       ypred = [0.6 0.2 0.1; 0.05 0.78 0.22; ...]
    sz = size(xpred, 1);
    ypred = zeros(sz, nn.layers(end));
    for i = 1:sz
        for j = 1:nn.layers(end)
            ypred(i,j) = get_value(nn, xpred(i,:), nn.num_layers, j);
        end
    end
end