% Mihail Dunaev
% May 2017

function v = kernel_vec(x, xs)
    % returns [fm(x) ... f2(x) f1(x) 1]
    % that is the kernel between each element in xs and x
    % adds 1 at the end
    N = size(xs, 1);
    s = 0.5;
    v = ones(1, N+1);
    for i = 1:N
        v(i) = kernel(xs(i,:), x, s);
    end
end