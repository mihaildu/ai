% Mihail Dunaev
% May 2017

function [xs, ys] = read_data(fname)
    % function that reads data for multiple input features (xtrain)
    %       and one output (ytrain).
    %   input
    %       fname = input file name (e.g. '../../input/tumors.in')
    %   output
    %       xs = xtrain values
    %       ys = ytrain values
    data = load(fname);
    xs = data(:,1:end-1);
    ys = data(:,end);
end