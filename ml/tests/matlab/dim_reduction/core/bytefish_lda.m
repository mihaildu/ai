% Mihail Dunaev
% May 2017

function lda_mat = bytefish_lda(xs, ys, k)
    % lda with code taken from bytefish.de
    num_classes = 2;
    xs0 = xs(ys == 0,:);
    xs1 = xs(ys == 1,:);
    mu_total = mean(xs);
    mu = [mean(xs0); mean(xs1)];
    c = ys + 1;
    sw = (xs - mu(c,:))' * (xs - mu(c,:));
    sb = (ones(num_classes,1) * mu_total - mu)' * ...
        (ones(num_classes,1) * mu_total - mu);

    %[V, D] = eig(sw\sb)
    %[D, i] = sort(diag(D), 'descend');
    %V = V(:,i);

    % svd should work better than eig
    [u, ~, ~] = svd(pinv(sw) * sb);
    lda_mat = u(:,1:k);
end