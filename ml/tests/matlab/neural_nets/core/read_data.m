% Mihail Dunaev
% May 2017

function [xs, ys] = read_data(layers, fname)
    % reads train data based on neural net architecture
    % we only need first & last layer (inputs & outputs)
    % input
    %   layers = layers array
    %   fname = filename; e.g. 'xnor_nn.in'
    % output
    %   xs = feature values from file
    %   ys = classes for xs (as a vector)
    data = load(fname);
    delim = layers(1);
    xs = data(:,1:delim);
    % assume correct input file
    ys = data(:,delim+1:end);
end