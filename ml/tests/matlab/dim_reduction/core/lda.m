% Mihail Dunaev
% May 2017

function lda_mat = lda(xs, ys, k)
    % performs lda on xs (n -> k) based on classes in ys
    % initial dimension n = size(xs, 2)
    % returns projection matrix
    
    % split xs after class
    xs0 = xs(ys == 0,:);
    xs1 = xs(ys == 1,:);
    n0 = size(xs0, 1);
    n1 = size(xs1, 1);
    
    % compute means
    m0 = mean(xs0);
    m1 = mean(xs1);
    
    % mean norm before this
    %m = mean(xs);
    %ms = [m0 - m; m1 - m];
    ms = [m0; m1];
    
    % cov of class means/mean diff
    num_classes = 2;
    sb = (1 / num_classes) * (ms' * ms);
    
    % sum of scatter within classes
    % TODO we might need to subtract mean here
    % doesn't seem to make a diff
    %nxs0 = xs0 - repmat(m0, size(xs0,1), 1);
    %nxs1 = xs1 - repmat(m1, size(xs1,1), 1);
    %sig0 = (1 / n0) * (nxs0' * nxs0);
    %sig1 = (1 / n1) * (nxs1' * nxs1);
    sig0 = (1 / n0) * (xs0' * xs0);
    sig1 = (1 / n1) * (xs1' * xs1);
    sig = (1 / 2) * (sig0 + sig1);
    
    % eigenvectors
    [u, ~, ~] = svd(pinv(sig) * sb);
    
    % first k column vectors from u are the ones onto which we proj
    lda_mat = u(:,1:k);
end