% Mihail Dunaev
% May 2017

function r = kernel(x1, x2, s)
    % returns the kernel/similarity between x1 and x2
    % this is a gaussian kernel with std = s
    r = exp(-sum((x1 - x2) .^ 2) / (2 * (s ^ 2)));
end