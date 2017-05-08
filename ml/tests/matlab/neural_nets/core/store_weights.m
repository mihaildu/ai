% Mihail Dunaev
% May 2017

function store_weights(fname, nn)
    % stores weights from nn to a file
    % input
    %   fname = file name (e.g. 'weights.out')
    %   nn = neural net
    fid = fopen(fname, 'w');
    for i = 1:(nn.num_layers - 1)
        for j = 1:nn.layers(i+1)
            w = get_input_weights(nn, i, j);
            for k = 1:size(w)
                fprintf(fid, '%d ', w(k));
            end
            fprintf(fid, '\n');
        end
    end
    fclose(fid);
end