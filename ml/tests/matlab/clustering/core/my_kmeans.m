% Mihail Dunaev
% May 2017

function [ks, cs] = my_kmeans(xs, K, it)
    % tries to find K clusters in xs data
    % input
    %   xs = ND data
    %   K = number of clusters (int)
    %   it = optional param; if true,
    %       iterative plotting is enabled
    % output
    %   ks = centers for each cluster
    %   cs = class for each point in xs
    
    N = size(xs, 1);
    num_dim = size(xs, 2);
    
    % start with random points (old)
    %kmin = min(min(xs));
    %kmax = max(max(xs));
    %ks = rand(K, num_dim) * (kmax - kmin) + kmin;
    
    % start with K random points from xs
    is = randperm(N);
    ks = zeros(K, num_dim);
    for i = 1:K
        index = is(i);
        ks(i,:) = xs(index,:);
    end
    cs = zeros(N, 1);
    ds = zeros(K, 1);
    
    % stop when cost doesn't get modified or centers don't move anymore
    while true
        new_vals = false;
        % group points by distance
        for i = 1:N
            for j = 1:K
                ds(j) = norm(xs(i,:) - ks(j,:));
            end
            [~, c] = min(ds);
            if new_vals == false
                if cs(i) ~= c
                    new_vals = true;
                end
            end
            cs(i) = c;
        end

        if new_vals == false
            break;
        end

        % iterative plotting
        if nargin > 2
            if it == true
                plot_data(xs, ks, cs, N);
                waitforbuttonpress;
                cla;                
            end
        end

        % move centers to mean
        for j = 1:K
            ks(j,:) = mean(xs(cs == j, :));
        end

        % plot again
        if nargin > 2
            if it == true
                plot_data(xs, ks, cs, N);
                waitforbuttonpress;
                cla;
            end
        end
    end
end