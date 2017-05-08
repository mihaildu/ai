% Mihail Dunaev
% May 2017
%
% Use NN to recognize digit in image.
%
% Info
%   input/xtrain
%       5000 images 20x20 pixels, grayscale
%       each image has a digit drawn
%       500 images for each digit
%
%   output/ytrain
%       class for the digit in the image (10 classes)
%       class 10 = digit 0
%
%   stats
%       pred_nn_it takes 0.0270 for one input => 135s for all
%       cost_with_grad once: 96.565s (no reg)
%       it takes 28.29min to train for all input with fmincg (nn w/ 25)
%
%   results
%       nn - 99.48% (weights are in weights_test6_nn.mat)
%       logreg - 94.42% (weights_test6_log.mat)
%       nn with 70%/30% - 91.33%
%
% fminunc gets stuck; I used fmincg instead
% I didn't use regularization

function test10_digits()
    addpath('core/');
    %layers = [400 10];
    layers = [400 25 10];
    nn = build_nn(layers);
    
    % read training data
    %[xtrain, ytrain, xtest, ytest] = my_read_data2('../../input/digits_nn.mat');
    [xtrain, ytrain] = my_read_data1('../../input/digits_nn.mat');
    %[xtrain, ytrain] = read_data(layers, '../../input/digits_nn.in');
    
    % train nn
    %nn = train(xtrain, ytrain, nn);
    
    % store weights to file
    %store_weights_mat('../../input/weights_test6_nn_70', nn);
    
    % load weights from file
    nn = load_weights_mat('../../input/weights_test6_nn.mat', nn);
    %nn = load_weights_mat('../../input/weights_test6_log.mat', nn);
    %nn = load_weights_mat('../../input/weights_test6_nn_70.mat', nn);
    
    % test prediction
    %predict1(xtrain, ytrain, nn);
    %predict2(xtest, ytest, nn);
    
    % display weights for first layer as images
    %display_weights_imgs(nn);
    
    % display some images with the prediction
    display_with_pred(xtrain, nn);
    
    % display some input images
    %display_data1(xtrain);
    %display_data2(xtrain, ytrain, xtest, ytest);
    
    % convert input file from .mat to .in
    %write_data('../../input/digits_nn.in', xtrain, ytrain);
    
    % load official weights (old; needs flipping)
    %nn = my_load_weights('../../input/digits_weights_nn.mat', nn)
end

% display weights for each node of layer 2 as images
% layer 2 is the first one with nodes
function display_weights_imgs(nn)
    layer = 2;
    colormap(gray);
    for j = 1:nn.layers(layer)
        w = get_input_weights(nn, layer - 1, j);
        % ignore last element - w(1:end-1);
        % scale values in [-1 1]
        w(1:end-1) = w(1:end-1) / max(abs(w(1:end-1)));
        img = reshape(w(1:end-1), 20, 20);
        imagesc(img, [-1 1]);
        k = waitforbuttonpress;
    end 
end

% computes accuracy for all xtrain
function predict1(xtrain, ytrain, nn)
    ypred = zeros(size(ytrain));
    num_total = size(ytrain, 1);
    num_correct = 0;
    for k = 1:num_total
        % fliplr for official weights
        %ypred(k,:) = pred_nn_it(nn_global, fliplr(xtrain(k,:)));
        ypred(k,:) = pred_nn_it(nn, xtrain(k,:));
        [~, cpred] = max(ypred(k,:));
        [~, ctrain] = max(ytrain(k,:));
        if cpred == ctrain
            num_correct = num_correct + 1;
        end
    end
    %disp(num_correct);
    fprintf('Accuracy: %.2f%%\n', num_correct * 100 / num_total);
end

% computes accuracy for xtest
function predict2(xtest, ytest, nn)
    ypred = zeros(size(ytest));
    num_total = size(ytest, 1);
    num_correct = 0;
    for k = 1:num_total
        % fliplr for official weights
        %ypred(k,:) = pred_nn_it(nn_global, fliplr(xtrain(k,:)));
        ypred(k,:) = pred_nn_it(nn, xtest(k,:));
        [~, cpred] = max(ypred(k,:));
        [~, ctrain] = max(ytest(k,:));
        if cpred == ctrain
            num_correct = num_correct + 1;
        end
    end
    %disp(num_correct);
    fprintf('Accuracy: %.2f%%\n', num_correct * 100 / num_total);
end

% load official weights (andrew ng's course) to nn
% fliplr is required because I use the bias at the end (even in pred)
% TODO still doesn't seem to work, only get 22% accuracy
function nn = my_load_weights(fname, nn)
    data = load(fname);
    for i = 1:25
        %w = circshift(data.Theta1(i,:), [0 1])';
        w = fliplr(data.Theta1(i,:))';
        nn = set_weights(nn, 1, i, w);
    end
    for i = 1:10
        %w = circshift(data.Theta2(i,:), [0 1])';
        w = fliplr(data.Theta2(i,:))';
        nn = set_weights(nn, 2, i, w);
    end
end

% converts input file (xtrain, ytrain) 
% from .mat to .in to make it more portable
function write_data(fname, xtrain, ytrain)
    Ntrain = size(xtrain, 1);
    fid = fopen(fname, 'w');
    num_xs = size(xtrain, 2);
    num_ys = size(ytrain, 2);
    for i = 1:Ntrain
        for j = 1:num_xs
            fprintf(fid, '%d ', xtrain(i,j));
        end
        for j = 1:num_ys
            fprintf(fid, '%d ', ytrain(i,j));
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
end

% shows some random images with the predicted output
function display_with_pred(xtrain, nn)
    Ntrain = size(xtrain, 1);
    is = randperm(Ntrain);
    num_images = 10;
    colormap(gray);
    for k = 1:num_images
        index = is(k);
        img = reshape(xtrain(index,:), 20, 20);
        imagesc(img, [-1 1]);
        ypred = pred_nn_it(nn, xtrain(index,:));
        [~, cpred] = max(ypred);
        if cpred == 10
            fprintf('Predicted digit is 0\n');
        else
            fprintf('Predicted digit is %d\n', cpred);
        end
        k = waitforbuttonpress;
    end
end

% shows 5 images for each digit
function display_data1(xtrain)
    Ntrain = size(xtrain, 1);
    colormap(gray);
    num_images = 5;
    delta_images = 500;
    for i = 1:delta_images:Ntrain
        for j = 0:(num_images-1)
            img = reshape(xtrain(i+j,:), 20, 20);
            imagesc(img, [-1 1]);
            k = waitforbuttonpress;
        end
    end
end

% shows 5 images for each digit, for both train and test
function display_data2(xtrain, ytrain, xtest, ytest)
    Ntrain = size(xtrain, 1);
    Ntest = size(xtest, 1);
    colormap(gray);
    num_images = 5;
    
    % show train
    disp('Showing train images');
    delta_images_train = 350;
    for i = 1:delta_images_train:Ntrain
        for j = 0:(num_images-1)
            img = reshape(xtrain(i+j,:), 20, 20);
            imagesc(img, [-1 1]);
            disp(ytrain(i+j,:));
            k = waitforbuttonpress;
        end
    end
    
    % show test
    disp('Showing test images');
    delta_images_test = 150;
    for i = 1:delta_images_test:Ntest
        for j = 0:(num_images-1)
            img = reshape(xtest(i+j,:), 20, 20);
            imagesc(img, [-1 1]);
            disp(ytest(i+j,:));
            k = waitforbuttonpress;
        end
    end
end

% reads xtrain & ytrain from .mat file (train + test)
% TODO this can be split into read_data_train/test
function [xtrain, ytrain, xtest, ytest] = my_read_data2(fname)
    % 70% = 3500 images
    % 30% = 1500 images
    Ntrain = 3500;
    Ntest = 1500;
    num_classes = 10;
    num_features = 400;
    dtrain = Ntrain / num_classes;
    dtest = Ntest / num_classes;
    xtrain = zeros(Ntrain, num_features);
    ytrain = zeros(Ntrain, num_classes);
    xtest = zeros(Ntest, num_features);
    ytest = zeros(Ntest, num_classes);
    data = load(fname);
    for i = 1:num_classes
        sit = 500 * (i - 1) + 1;
        eit = sit + dtrain - 1;
        site = eit + 1;
        eite = site + dtest - 1;
        sitx = dtrain * (i - 1) + 1;
        eitx = sitx + dtrain - 1;
        sitxe = dtest * (i - 1) + 1;
        eitxe = sitxe + dtest - 1;
        xtrain(sitx:eitx, :) = data.X(sit:eit, :);
        j = sit;
        for k = sitx:eitx
            l = data.y(j);
            ytrain(k, l) = 1;
            j = j + 1;
        end
        xtest(sitxe:eitxe, :) = data.X(site:eite, :);
        j = site;
        for k = sitxe:eitxe
            l = data.y(j);
            ytest(k, l) = 1;
            j = j + 1;
        end
    end
end

% reads xtrain & ytrain from .mat file
function [xtrain, ytrain] = my_read_data1(fname)
    data = load(fname);
    xtrain = data.X;
    num_classes = 10;
    ytrain = zeros(size(xtrain, 1), num_classes);
    for i = 1:size(ytrain, 1)
        j = data.y(i);
        ytrain(i, j) = 1;
    end
end

% train function for test 6
function nn = train(xtrain, ytrain, nn)
    % set initial weights in a vec (random small values)
    t0 = get_random_thetas(nn.layers, 0.1);

    % fminunc doesn't work well, use fmincg
    fun = @(t)(cost_with_grad(t, 0, xtrain, ytrain, nn));
    options = optimset('fminunc');
    options = optimset(options, 'GradObj', 'on', 'MaxIter', 100);
    [tf, ~] = fmincg(fun, t0, options);

    % set nn weights to tf
    nn = unroll_thetas(tf, nn);
end