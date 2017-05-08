% Mihail Dunaev
% May 2017
%
% Simple test that performs k-nearest neighbors (knn) on 2D data
% from 'logistic_multiclass.in'.

function test1()
    [xs, ys] = read_data('../../input/logistic_multiclass.in');
    
    k = 3;
    plot_everything = true;
    xpred = [6 2.1; 4.5 4.5; 6 7; 6 6.1; 3 6; 2 1.5; 5 4];
    ypred = knn(xpred, xs, ys, k, plot_everything);
    plot_data(xs, ys, xpred, ypred);
end