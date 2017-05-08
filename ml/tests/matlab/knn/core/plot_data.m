% Mihail Dunaev
% May 2017

function plot_data(xs, ys, xpred, ypred)
    % plots 2D data for 3 classes
    xs0 = xs(ys == 0,:);
    xs1 = xs(ys == 1,:);
    xs2 = xs(ys == 2,:);
    
    figure;
    hold on;
    plot(xs0(:,1), xs0(:,2), 'bo');
    plot(xs1(:,1), xs1(:,2), 'rd');
    plot(xs2(:,1), xs2(:,2), 'gs');
    xmin = min(xs(:,1));
    xmax = max(xs(:,1));
    ymin = min(xs(:,2));
    ymax = max(xs(:,2));
    xlim([xmin-1 xmax+1]);
    ylim([ymin-1 ymax+1]);
    
    for i = 1:size(xpred, 1)
        if ypred(i) == 0
            plot(xpred(i,1), xpred(i,2), 'bx');
        elseif ypred(i) == 1
            plot(xpred(i,1), xpred(i,2), 'rx');
        elseif ypred(i) == 2
            plot(xpred(i,1), xpred(i,2), 'gx');
        else
            plot(xpred(i,1), xpred(i,2), 'kx');
        end 
    end
end