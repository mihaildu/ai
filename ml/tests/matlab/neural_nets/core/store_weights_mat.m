% Mihail Dunaev
% May 2017

function store_weights_mat(fname, nn)
    % stores weights from nn to a .mat file
    % input
    %   fname = file name, without extension (e.g. 'weights')
    %       the '.mat' will be added automatically
    %   nn = neural net
    W = nn.W;
    save(fname, 'W');
end