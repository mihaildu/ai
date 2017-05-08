% Mihail Dunaev
% May 2017

function nn = build_nn(layers)
    % builds a neural net with the architecture specified in layers array
    %
    % input
    %   layers = list of number of nodes for each layer
    %   e.g. layers = [3 2 1]
    %       3 features on layers 1
    %       2 nodes on layer 2
    %       1 node on layer 3
    %
    % output
    %   nn = matlab structure with multiple fields
    %       nn.layers = layers array (array of int)
    %       nn.num_features = how many features it supports (int)
    %       nn.num_layers = number of layers (int)
    %       nn.W = weights cube (use set_weights/get_weights)
    %       nn.fmap = name (string) - function mapping
    %       nn.allowed_functions = <string>:<bool>; if function name is
    %           not in this dictionary, it's not supported
    %       nn.default_function = use this as activation if no other
    %           function is specified
    %       nn.F = cube of functions (strings) for each node
    %       nn.err = used in backprop to temp store the error
    %       nn.val = used to temp store output values for each node for
    %           some input xs(i,:)
    %
    % weights and functions can be set with other functions
    % set_weights(layer, num_node, w);
    % set_function(layer, num_node, f);
    % WARNING they are not set by default (all zeros)

    % sanity checks on layers TODO
    nn.layers = layers;
    nn.num_features = nn.layers(1);
    nn.num_layers = size(nn.layers, 2);

    % use cube for weights
    sz1 = nn.num_layers - 1;
    sz2 = max(nn.layers(1:end-1)) + 1;
    sz3 = max(nn.layers(2:end));
    nn.W = zeros(sz1, sz2, sz3);

    % set weights to some initial value ones()/zeros()
    %for i = 1:(size(nn.layers, 2) - 1)
    %    default_weight = ones(nn.layers(i) + 1,1);
    %    for j = 1:nn.layers(i+1)
    %        nn = set_weights(nn, i, j, default_weight);
    %    end
    %end
    
    % set possible functions & default one
    syms t;
    nn.fmap = containers.Map;
    nn.allowed_functions = containers.Map;
    
    % add multiple functions here
    nn.sigmoid = @(t)(1 / (1 + exp(-t)));
    nn.fmap('sigmoid') = nn.sigmoid;
    nn.allowed_functions('sigmoid') = true;
    nn.default_function = 'sigmoid';

    % set activation functions to default function
    sz1 = nn.num_layers - 1;
    sz2 = max(nn.layers(2:end));
    nn.F = cell(sz1, sz2);
    for i = 1:sz1
        for j = 1:nn.layers(i+1)
            nn = set_function(nn, i, j, nn.default_function);
        end
    end

    % backpropagation error
    nn.err = zeros(sz1, sz2);

    % initialize value matrix (or node output values)
    sz1 = nn.num_layers;
    sz2 = max(nn.layers);
    nn.val = zeros(sz1, sz2);
end