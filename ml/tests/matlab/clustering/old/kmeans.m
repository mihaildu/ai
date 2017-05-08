% simple implementation of kmeans
% this is already implemented in matlab
% some conventions
%   first class = red/value 1
%   second class = blue/value 2

function main()
    %test1();
    test2();
end

% same as test1, using dbscan
function test2()
    [xs, N] = read_data('../input/kmeans.in');
    
    figure;
    hold on;
    
    % [la, type] = dbscan(new_obj_pts,5,[]);
    [cs, ~] = dbscan(xs, 2, []);
    ks = zeros(2,2);
    plot_data_test1(xs, ks, cs, N);
end

function test1()
    [xs, N] = read_data('../input/kmeans.in');

    % init plotting
    figure;
    hold on;

    % matlab version [IDX, C] = kmeans(X, K)
    % idx is cs and c is ks
    [ks, cs] = my_kmeans(xs, 2);
    plot_data_test1(xs, ks, cs, N);
end

function [ks, cs] = my_kmeans(xs, K)
    % start with some random k pts
    N = size(xs, 1);
    num_dim = size(xs, 2);
    %kmin = min(min(xs));
    %kmax = max(max(xs));
    %ks = rand(K, num_dim) * (kmax - kmin) + kmin;
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

        % plot
        plot_data_test1(xs, ks, cs, N);
        waitforbuttonpress;
        cla;

        % move centers to mean
        for j = 1:K
            ks(j,:) = mean(xs(cs == j, :));
        end

        % plot again
        plot_data_test1(xs, ks, cs, N);
        waitforbuttonpress;
        cla;
    end
end

function plot_data_test1(xs, ks, cs, N)
    % hardcoded colors: 
    %   first class = red/1
    %   second class = blue/2
    for i = 1:N
        if cs(i) == 1
            plot(xs(i,1), xs(i,2), 'ro');
        elseif cs(i) == 2
            plot(xs(i,1), xs(i,2), 'bo');
        else
            plot(xs(i,1), xs(i,2), 'ko');
        end
    end
    
    % set range for x/y axis
    xmin = min(xs(:,1));
    xmax = max(xs(:,1));
    ymin = min(xs(:,2));
    ymax = max(xs(:,2));
    xlim([xmin-1 xmax+1]);
    ylim([ymin-1 ymax+1]);
    
    % plot ks
    plot(ks(1,1), ks(1,2), 'rx');
    plot(ks(2,1), ks(2,2), 'bx');
end

function [xs, N] = read_data(fname)
    xs = load(fname);
    N = size(xs, 1);
end