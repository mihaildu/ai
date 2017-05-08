% dimensionality reduction algos

function main()
    %pca_test();
    lda_test();
end

% perform lda on data
function lda_test()
    global xs ys;
    read_data('../input/lda.in');
    
    % feature scaling & mean norm
    nxs = proc_input(xs);
    
    % compute proj matrix
    lda_mat = lda(nxs, ys, 1);
    %lda_mat = bytefish_lda(nxs, ys, 1);
    
    % proj the input/any would work xs/nxs
    zs = proj(nxs, lda_mat);
    
    % plot stuff
    plot_data_test2(nxs, ys, zs, lda_mat);
end

% performs lda on xs (n -> k)
% returns the proj matrix
function lda_mat = lda(xs, ys, k)
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

% lda with code taken from bytefish.de
function lda_mat = bytefish_lda(xs, ys, k)
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

% plotting for test 2/lda
function plot_data_test2(xs, ys, zs, lda_mat)
	figure;
    hold on;
    
    xs0 = xs(ys == 0,:);
    xs1 = xs(ys == 1,:);
    plot(xs0(:,1), xs0(:,2), 'ro');
    plot(xs1(:,1), xs1(:,2), 'bo');

    % set range for x/y axis
    xmin = min(xs(:,1));
    xmax = max(xs(:,1));
    ymin = min(xs(:,2));
    ymax = max(xs(:,2));
    xlim([xmin-1 xmax+1]);
    ylim([ymin-1 ymax+1]);
    
    % plot lda line
    slope = lda_mat(2) / lda_mat(1);
    delta = 0.1;
    wxs = (xmin:delta:xmax)';
    wys = wxs * slope;
    plot(wxs, wys, 'g-');

    % plot projected points
    pxs = zs * lda_mat';
    pxs0 = pxs(ys == 0,:);
    pxs1 = pxs(ys == 1,:);
    plot(pxs0(:,1), pxs0(:,2), 'ro');
    plot(pxs1(:,1), pxs1(:,2), 'bo');
end

% perform pca on data
% pca() should be in matlab newest version
function pca_test()
    global xs;
    read_data('../input/logistic.in');

    % perform feature scaling and mean norm on input
    nxs = proc_input(xs);

    % compute proj matrix
    pca_mat = pca(nxs, 1);

    % proj the input/any would work xs/nxs
    zs = proj(nxs, pca_mat);

    % plot stuff
    plot_data_test1(nxs, zs, pca_mat);
end

% preprocessing = mean norm + feature scaling
function nxs = proc_input(xs)
    % mean normalization
    nxs = xs - repmat(mean(xs), size(xs,1), 1);

    % feature scaling TODO [a,b] -> [c,d]
    % (x - a) * (d - c) / (b - a) + c
    % [xmin,xmax] -> [-1,1] | xs(:,1)
    % [ymin,ymax] -> [-1,1] | xs(:,2)

    % xmin = min(xs(:,1));
    % xmax = max(xs(:,1));
    % ymin = min(xs(:,2));
    % ymax = max(xs(:,2));
    % xs(i,1) = (xs(i,1) - xmin) * 2 / (xmax - xmin) - 1;
    % xs(i,2) = (xs(i,2) - ymin) * 2 / (ymax - ymin) - 1;
end

% performs pca on xs (n -> k)
% returns the proj matrix
function pca_mat = pca(xs, k)
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

% proj xs (n dim) to space (k dim) det by mat (col vectors)
function zs = proj(xs, mat)
    zs = xs * mat;
end

% plotting for test 1/pca
function plot_data_test1(xs, zs, pca_mat)
	figure;
    hold on;
    plot(xs(:,1), xs(:,2), 'ko');

    % set range for x/y axis
    xmin = min(xs(:,1));
    xmax = max(xs(:,1));
    ymin = min(xs(:,2));
    ymax = max(xs(:,2));
    xlim([xmin-1 xmax+1]);
    ylim([ymin-1 ymax+1]);
    
    % plot pca line
    slope = pca_mat(2) / pca_mat(1);
    delta = 0.1;
    wxs = (xmin:delta:xmax)';
    wys = wxs * slope;
    plot(wxs, wys, 'g-');
    
    % plot projected points
    pxs = zs * pca_mat';
    plot(pxs(:,1), pxs(:,2), 'ro');
end

% read data from file
function read_data(fname)
    global xs xs0 xs1 ys n0 n1 N;
    data = load(fname);
    xs = data(:,1:end-1);
    ys = data(:,end);
    xs0 = xs(ys == 0,:);
    n0 = size(xs0, 1);
    xs1 = xs(ys == 1,:);
    n1 = size(xs1, 1);
    N = size(xs, 1);
end