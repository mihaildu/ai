% Mihail Dunaev
% May 2017

function nxs = proc_input(xs)
    % preprocessing = mean norm + feature scaling
    
    % mean normalization
    nxs = xs - repmat(mean(xs), size(xs,1), 1);

    % feature scaling TODO [a,b] -> [c,d]
    % (x - a) * (d - c) / (b - a) + c
    % [xmin,xmax] -> [-1,1] | xs(:,1)
    % [ymin,ymax] -> [-1,1] | xs(:,2)

    % xmin = min(xs(:,1));
    % xmax = max(xs(:,1));
    % ymin = min(xs(:,2));
    % ymax = max(xs(:,2));
    % xs(i,1) = (xs(i,1) - xmin) * 2 / (xmax - xmin) - 1;
    % xs(i,2) = (xs(i,2) - ymin) * 2 / (ymax - ymin) - 1;
end