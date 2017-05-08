% Mihail Dunaev
% May 2017

function nn = load_weights_mat(fname, nn)
    % reads weights values from fname (.mat) to nn
    % same as a pretrained neural net
    % input
    %   fname = file name (must be .mat, created by 
    %       a previous call to 'store_weights_mat'
    %   nn = neural net
    % output
    %   nn = neural net with the new weights
    weights = load(fname);
    nn.W = weights.W;
end