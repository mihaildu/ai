% Mihail Dunaev
% May 2017

function ypred = knn(xpred, xs, ys, k, it)
    % performs k-nearest neighbors on xpred using xs
    % ys = classes for xs (0/1...)
    % xpred = [xpred1; xpred2; ...]
    % it = iterative plots (optional)
    % TODO make this look a little bit nicer
    
    ypred = zeros(size(xpred,1), 1);
    num_classes = max(ys) + 1;
    
    if nargin > 4
        if it == true
            figure;
            hold on;
        end
    end
    
    % look at all the points
    for i = 1:size(xpred, 1)
        ds = zeros(k, 1);
        is = zeros(k, 1);
        cs = zeros(num_classes, 1);
        
        next_index = 1;
        maxd = -1;
        hd = -1;
        for j = 1:size(xs, 1)
            d = norm(xs(j,:) - xpred(i,:));
            
            % if there is room in ds, add it
            if next_index <= k
                ds(next_index) = d;
                is(next_index) = j;
                cls = ys(j) + 1;
                cs(cls) = cs(cls) + 1;
                if d > maxd
                    maxd = d;
                    hd = next_index;
                end
                next_index = next_index + 1;
            % else replace only if greater than maximum
            elseif d < maxd
                % subtract class counter first
                old_index = is(hd);
                cls = ys(old_index) + 1;
                cs(cls) = cs(cls) - 1;
                
                % update with new values
                ds(hd) = d;
                is(hd) = j;
                cls = ys(j) + 1;
                cs(cls) = cs(cls) + 1;
                
                % update hd & maxd
                maxd = -1;
                for h = 1:k
                    if ds(h) > maxd
                        maxd = ds(h);
                        hd = h;
                    end
                end
            end
        end
        [~, cls] = max(cs);
        ypred(i) = cls - 1;
        
        if nargin > 4
            if it == true
                plot_data_it(xs, ys, xpred(i,:), is, cls);
                waitforbuttonpress;
                cla;
            end
        end
    end
end