% Mihail Dunaev
% May 2017

function check_grad(fgrad, t0)
    % compares gradient returned by fgrad with numerical one
    % e.g. usage: check_grad(@cost_with_grad, t0);
    % fgrad: [cost, grad] = fgrad(t)
    [~, grad] = fgrad(t0);
    sz = size(t0, 1);
    num_grad = zeros(sz, 1);
    eps = 0.0001;
    for i = 1:sz
        t0(i) = t0(i) + eps;
        [fp, ~] = fgrad(t0);
        t0(i) = t0(i) - (2 * eps);
        [fm, ~] = fgrad(t0);
        t0(i) = t0(i) + eps;
        num_grad(i) = (fp - fm) / (2 * eps);
    end
    disp(num_grad);
    disp(grad);
    disp(sum((num_grad - grad) .^ 2));
end