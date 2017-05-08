% Mihail Dunaev
% May 2017

function plot_data(xs, ks, cs, N)
    % plots clustered xs points with colors (for 2 classes)
    % inputs
    %   xs = data points
    %   ks = center values (for each cluster)
    %   cs = class for each data point
    %   N = num points
    % hardcoded colors:
    %   first class = red/1
    %   second class = blue/2
    for i = 1:N
        if cs(i) == 1
            plot(xs(i,1), xs(i,2), 'ro');
        elseif cs(i) == 2
            plot(xs(i,1), xs(i,2), 'bo');
        else
            plot(xs(i,1), xs(i,2), 'ko');
        end
    end
    
    % set range for x/y axis
    xmin = min(xs(:,1));
    xmax = max(xs(:,1));
    ymin = min(xs(:,2));
    ymax = max(xs(:,2));
    xlim([xmin-1 xmax+1]);
    ylim([ymin-1 ymax+1]);
    
    % plot ks
    plot(ks(1,1), ks(1,2), 'rx');
    plot(ks(2,1), ks(2,2), 'bx');
end