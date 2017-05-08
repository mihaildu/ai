% Mihail Dunaev
% May 2017

function zs = proj(xs, mat)
    % projects xs (n dimensions) to space (k dimensions)
    % determined by mat (column vectors)
    zs = xs * mat;
end