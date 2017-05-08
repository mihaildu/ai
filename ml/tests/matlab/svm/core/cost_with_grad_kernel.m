% Mihail Dunaev
% May 2017

function [cost, grad] = cost_with_grad_kernel(t, xs, ys, C)
    % cost & grad for kernel SVM
    xs0 = xs(ys == 0,:);
    xs1 = xs(ys == 1,:);
    n0 = size(xs0, 1);
    n1 = size(xs1, 1);
    N = n0 + n1;
    
    % compute similarity for each training point
    % prob better way to do this
    mat1 = zeros(n1, N+1);
    for k = 1:n1
        mat1(k,:) = kernel_vec(xs1(k,:), xs);
    end
    mat0 = zeros(n0, N+1);
    for k = 1:n0
        mat0(k,:) = kernel_vec(xs0(k,:), xs);
    end
    
    %C = 100;
    cost = sum(max((1 - mat1 * t) * exp(1) / 2, 0)) + ...
        sum(max((1 + mat0 * t) * exp(1) / 2, 0));
    cost = C * cost + sum(t .^ 2) / 2;
    
    % TODO grad
    if nargout > 1
        grad = zeros(size(t));
    end
end