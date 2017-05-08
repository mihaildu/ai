% Mihail Dunaev
% May 2017
%
% Simple test that finds 2 clusters in 2D data using kmeans and
% dbscan. Data is in 'kmeans.in'.
%
% Some conventions:
%   first class = red/value 1
%   second class = blue/value 2

function test1()
    % wrapper for run_test
    iterative_plotting = true;
    use_dbscan = false;
    run_test(iterative_plotting, use_dbscan);
end

function run_test(iterative_plotting, use_dbscan)
    addpath('core/');
    [xs, N] = read_data('../../input/kmeans.in');
    
    % init plotting
    if iterative_plotting == true
        figure;
        hold on;
    end
    
    if use_dbscan == true
        [cs, ~] = dbscan(xs, 2, []);
        ks = zeros(2,2);
    else
        [ks, cs] = my_kmeans(xs, 2, iterative_plotting);
        % matlab version
        %[cs, ks] = kmeans(xs, 2);
    end
    
    plot_data(xs, ks, cs, N);
end