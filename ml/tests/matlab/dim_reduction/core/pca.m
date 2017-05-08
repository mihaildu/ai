% Mihail Dunaev
% May 2017

function pca_mat = pca(xs, k)
    % performs pca on xs (n -> k)
    % initial dimension n = size(xs, 2)
    % returns projection matrix
    N = size(xs, 1);
    n = size(xs, 2);

    % TODO
    if k >= n
        return;
    end

    % cov matrix
    % since we performed mean norm, we don't need to subtract the mean
    % otherwise xs - mean(xs)
    sig = (1/N) * (xs' * xs);
    
    % eigenvectors
    [u, s, ~] = svd(sig);
    
    % first k column vectors from u are the ones onto which we proj
    pca_mat = u(:,1:k);
    
    % variance retained = s()
    s1 = 0;
    for i = 1:k
        s1 = s1 + s(i,i);
    end
    s2 = 0;
    for i = 1:n
        s2 = s2 + s(i,i);
    end
    vr = (s1 * 100) / s2;
    fprintf('Variance retained: %.2f%%\n', vr);
end