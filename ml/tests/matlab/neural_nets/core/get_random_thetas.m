% Mihail Dunaev
% May 2017

function t0 = get_random_thetas(layers, eps_theta)
    % returns random initial weights (thetas) for nn
    % weights are in [-eps_theta, eps_theta]
    t0 = [];
    for i = 1:(size(layers, 2) - 1)
        for j = 1:layers(i+1)
            weights = rand(layers(i) + 1, 1) * ...
                    (2 * eps_theta) - eps_theta;
            t0 = [t0; weights];
        end
    end
end