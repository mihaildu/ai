% Mihail Dunaev
% May 2017

function tf = grad_descent(fun, t)
    % simple implementation of gradient descent
    % slightly improved version of `grad_descent_simple`
    %   input
    %       fun = function to minimize
    %       t = initial vars for fun
    %   output
    %       tf = values for t that minimize fun
    alpha = 0.001;
    eps = 0.0001;
    max_iter = 1000000;
    
    [prev_cost, grad] = fun(t);
    for i = 1:max_iter
        
        % if gradient too small we are close to the minimum
        % convergence also ok if (cost - prev_cost) < 0.001
        if norm(grad) <= eps
            break;
        end
        
        % update one step
        t = t - (alpha * grad);
        [cost, grad] = fun(t);
        
        % if cost goes higher, it overshoots
        % this might not be enough
        if cost > prev_cost
            alpha = alpha / 2;
        end
        
        % if cost decreases too slowly make alpha higher TODO
        prev_cost = cost;
    end

    % possibly not minimum
    if i == max_iter
        disp('Maximum number of iterations reached');
    end
    tf = t;
end