% Mihail Dunaev
% May 2017
%
% Simple test that performs Linear Discriminant Analysis (LDA)
% on 2D data (2D -> 1D). Input in 'logistic.in'.

function lda_test()
    addpath('core/');
    [xs, ys] = read_data('../../input/lda.in');
    
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