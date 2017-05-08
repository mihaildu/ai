% Mihail Dunaev
% May 2017

function plot_cost()
    % plots evolution of cost function
    global cost_vals;
    figure;
    hold on;
    plot((1:size(cost_vals, 1))', cost_vals);
end