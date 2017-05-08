% Mihail Dunaev
% May 2017

function plot_data_it(xs, ys, xpred, is, cls)
    % iterative plotting function used in knn
    xs0 = xs(ys == 0,:);
    xs1 = xs(ys == 1,:);
    xs2 = xs(ys == 2,:);
    
    plot(xs0(:,1), xs0(:,2), 'bo');
    plot(xs1(:,1), xs1(:,2), 'rd');
    plot(xs2(:,1), xs2(:,2), 'gs');
    xmin = min(xs(:,1));
    xmax = max(xs(:,1));
    ymin = min(xs(:,2));
    ymax = max(xs(:,2));
    xlim([xmin-1 xmax+1]);
    ylim([ymin-1 ymax+1]);
    
    if cls == 1
        plot(xpred(:,1), xpred(:,2), 'bx');
    elseif cls == 2
        plot(xpred(:,1), xpred(:,2), 'rx');
    elseif cls == 3
        plot(xpred(:,1), xpred(:,2), 'gx');
    else
        plot(xpred(:,1), xpred(:,2), 'kx');
    end
    
    plot(xs(is,1), xs(is,2), 'co');
end