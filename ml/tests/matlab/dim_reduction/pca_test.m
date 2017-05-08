% Mihail Dunaev
% May 2017
%
% Simple test that performs Principal Component Analysis (PCA)
% on 2D data (2D -> 1D). Input in 'logistic.in'.
% pca() should be implemented in newest matlab version.

function pca_test()
    addpath('core/');
    [xs, ~] = read_data('../../input/logistic.in');

    % perform feature scaling and mean norm on input
    nxs = proc_input(xs);

    % compute proj matrix
    pca_mat = pca(nxs, 1);

    % proj the input/any would work xs/nxs
    zs = proj(nxs, pca_mat);

    % plot stuff
    plot_data_test1(nxs, zs, pca_mat);
end

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