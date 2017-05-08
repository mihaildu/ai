% Mihail Dunaev
% May 2017

function [xs, N] = read_data(fname)
    xs = load(fname);
    N = size(xs, 1);
end