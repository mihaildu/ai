% Mihail Dunaev
% May 2017

function stop = outfun(t, optimValues, state)
    % function that gets called after every step in fminunc
    % it stores the cost values in a global var
    %
    % input
    %   t = current values for thetas
    %   optimValues.fval = function value for t
    %   optimValues.iteration = iteration number
    %   optimValues.procedure = procedure message
    %   optimValues.funccount = number of function evals until now
    %
    % output
    %   stop = if set to true, fminunc stops
    global cost_vals;
    stop = false;
    if optimValues.iteration == 0
        cost_vals = [];
    end
    cost_vals = [cost_vals; optimValues.fval];
end