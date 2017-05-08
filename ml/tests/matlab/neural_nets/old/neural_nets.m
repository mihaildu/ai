% neural nets

function main()
    %simple_test1();
    %simple_test3();
    %and_nn();
    %or_nn();
    %not_nn();
    %xnor_nn();
    simple_test2();
    %simple_test4();
    %simple_test5();
    %digits_test6();
end

% use NN to recognize digit in image
function digits_test6()
    % info
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

    global nn_global;
    %layers = [400 10];
    layers = [400 25 10];
    %layers = [2 1 1];
    nn_global = build_nn(layers);
    
    % read training data
    %read_data_test6_v2('../input/digits_nn.mat');
    %read_data_test6('../input/digits_nn.mat');
    %read_data(layers, '../input/digits_nn.in');
    
    % train nn
    %train_test6();
    
    % store weights to file
    %store_weights_mat('../input/weights_test6_nn_70');
    
    % load weights from file
    %load_weights_mat('../input/weights_test6_nn.mat');
    %load_weights_mat('../input/weights_test6_log.mat');
    load_weights_mat('../input/weights_test6_nn_70.mat');
    
    % test prediction
    %predict_test6();
    %predict_test6_v2();
    
    % display weights for first layer as images
    %display_weights_imgs_test6();
    
    % display some images with the prediction
    %display_with_pred_test6();
    
    % display some input images
    %display_data_test6();
    %display_data_test6_v2();
    
    % convert input file from .mat to .in
    %write_data_test6('../input/digits_nn.in');
    
    % load_weights from .mat/.in (old)
    %load_weights_test6('../input/digits_weights_nn.mat');
    %load_weights('../input/weights_test6_nn.in');
    
    % store weights
    %t0 = [1;2;3;4;5;6;7;8;9];
    %t0 = [1;2;3;4;7];
    %unroll_thetas(t0);
    %display_weights();
    %store_weights('../input/weights_test6_nn.in');
    %store_weights_mat('../input/weights_test6_nn');
end

% display weights for each node of layer 2 as images
% layer 2 is the first one with nodes
function display_weights_imgs_test6()
    global nn_global;
    layer = 2;
    colormap(gray);
    for j = 1:nn_global.layers(layer)
        w = get_input_weights(nn_global, layer - 1, j);
        % ignore last element - w(1:end-1);
        % scale values in [-1 1]
        w(1:end-1) = w(1:end-1) / max(abs(w(1:end-1)));
        img = reshape(w(1:end-1), 20, 20);
        imagesc(img, [-1 1]);
        k = waitforbuttonpress;
    end 
end

% computes accuracy for all xtrain
function predict_test6()
    global nn_global xtrain ytrain;
    ypred = zeros(size(ytrain));
    num_total = size(ytrain, 1);
    num_correct = 0;
    for k = 1:num_total
        % fliplr for official weights
        %ypred(k,:) = pred_nn_it(nn_global, fliplr(xtrain(k,:)));
        ypred(k,:) = pred_nn_it(nn_global, xtrain(k,:));
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
function predict_test6_v2()
    global nn_global xtest ytest;
    ypred = zeros(size(ytest));
    num_total = size(ytest, 1);
    num_correct = 0;
    for k = 1:num_total
        % fliplr for official weights
        %ypred(k,:) = pred_nn_it(nn_global, fliplr(xtrain(k,:)));
        ypred(k,:) = pred_nn_it(nn_global, xtest(k,:));
        [~, cpred] = max(ypred(k,:));
        [~, ctrain] = max(ytest(k,:));
        if cpred == ctrain
            num_correct = num_correct + 1;
        end
    end
    %disp(num_correct);
    fprintf('Accuracy: %.2f%%\n', num_correct * 100 / num_total);
end

% load weights into nn_global for test 6 from .mat
% this is used to load the official weights from andrew ng's course
% fliplr is required because I use the bias at the end (even in pred)
% TODO still doesn't seem to work, only get 22% accuracy
function load_weights_test6(fname)
    global nn_global;
    data = load(fname);
    for i = 1:25
        %w = circshift(data.Theta1(i,:), [0 1])';
        w = fliplr(data.Theta1(i,:))';
        nn_global = set_weights(nn_global, 1, i, w);
    end
    for i = 1:10
        %w = circshift(data.Theta2(i,:), [0 1])';
        w = fliplr(data.Theta2(i,:))';
        nn_global = set_weights(nn_global, 2, i, w);
    end
end

% converts input file (xtrain, ytrain) 
% from .mat to .in to make it more portable
function write_data_test6(fname)
    global xtrain ytrain Ntrain;
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

% shows some random images with the predicted output for test 6
function display_with_pred_test6()
    global nn_global Ntrain xtrain;
    is = randperm(Ntrain);
    num_images = 10;
    colormap(gray);
    for k = 1:num_images
        index = is(k);
        img = reshape(xtrain(index,:), 20, 20);
        imagesc(img, [-1 1]);
        ypred = pred_nn_it(nn_global, xtrain(index,:));
        [~, cpred] = max(ypred);
        if cpred == 10
            fprintf('Predicted digit is 0\n');
        else
            fprintf('Predicted digit is %d\n', cpred);
        end
        k = waitforbuttonpress;
    end
end

% shows some input images with digits for test 6
function display_data_test6()
    global xtrain Ntrain;
    % show 5 images for each digit
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

% shows some input images with digits for test 6
% TODO split this into train/test
% also, this is more a debugging method
function display_data_test6_v2()
    global xtrain ytrain Ntrain xtest ytest Ntest;
    % show 5 images for each digit, for both train and test
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

% read train data for test 6 (xtrain + xtest)
% TODO this can be split into read_data_train/test
function read_data_test6_v2(fname)
    % 70% = 3500 images
    % 30% = 1500 images
    global xtrain ytrain xtest ytest Ntrain Ntest;
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

% read train data for test 6
function read_data_test6(fname)
    global xtrain ytrain Ntrain;
    data = load(fname);
    xtrain = data.X;
    num_classes = 10;
    ytrain = zeros(size(xtrain, 1), num_classes);
    for i = 1:size(ytrain, 1)
        j = data.y(i);
        ytrain(i, j) = 1;
    end
    Ntrain = size(xtrain, 1);
end

% logistic regression as a NN with one layer
function simple_test5()
    global nn_global;
    
    % build nn
    layers = [1 1];
    nn_global = build_nn(layers);
    
    % read training data
    read_data(layers, '../input/tumors2.in');
    
    % train network
    train_test5();

    % predict values
    xpred = [7 43 44 70]';
    ypred = pred_nn(nn_global, xpred);
    
    % plot everything
    plot_test5(xpred, ypred);
    
    % plot cost function too
    plot_cost();
end

% train a nn with multiple output neurons
function simple_test4()
    global nn_global;
    layers = [2 2 4];
    %layers = [2 4];
    nn_global = build_nn(layers);
    
    % read train data
    read_data(layers, '../input/xnor_mc_nn.in');

    % train the nn
    train_test4();

    % predict some values
    %xpred = [0 0; 0 1; 1 0; 1 1];
    xpred = [0.1 0.1; -0.1 0.9; 0.9 0.2; 0.8 1.1; 0.4 0.5; 0.7 0.25;
        0.25 0.55; 0.5 0.5];
    ypred = pred_nn(nn_global, xpred);
    %disp(ypred);

    % class 1, 2, 3 & 4
    % [1 0 0 0] = class 1 / [0, 0]
    % [0 1 0 0] = class 2 / [1, 1]
    % [0 0 1 0] = class 3 / [0, 1]
    % [0 0 0 1] = class 4 / [1, 0]
    for i = 1:size(ypred, 1)
        [~, c] = max(ypred(i,:));
        % maybe do something if max < 0.5
        fprintf('[%f %f] has class %d\n', xpred(i,:), c);
    end
    
    % plot data
    plot_test4(xpred, ypred);
    
    % plot evolution of cost function
    plot_cost();
end

% train a nn to perform xnor
function simple_test2()
    % train time:
    % maxiter 10 = 2.69s (1000 => 5min??)
    % maxiter 1000 = 5 min (301.340s)
    % maxiter 5000 = 1262.6s = 21min
    %t0 = [0.1; -1.2; 3.0; 0.5; -0.25; -1; 0.7; 1.0; -2.0;
    %    0.2; -1.1; 2.7; -0.5; 0.6; -0.2; 0.3; 0.75; -2.1];
    global nn_global;

    % possible sol to always train this right:
    % start multiple times at random (eventually with eps >= 1)
    % when you get a low cost, stop

    % read train data
    %read_data_test2();

    % build nn struct
    layers = [2 2 1];
    nn_global = build_nn(layers);

    % load weights from file
    load_weights_mat('../../../input/weights_test2_xnor');
    
    % train it (global var)
    %train_xnor_test2();

    % store weights to file
    %store_weights_mat('../input/weights_test2_xnor');

    % predict some values
    xpred = [0 0; 0 1; 1 0; 1 1];
    ypred = pred_nn(nn_global, xpred);
    disp(ypred);

    % plot data
    %plot_all_test2();
    
    % plot cost function
    %plot_cost();
end

% neural net that performs logical XNOR
function xnor_nn()
    % 0 xnor 0 = 1;
    % 0 xnor 1 = 0;
    % 1 xnor 0 = 0;
    % 1 xnor 1 = 1;
    % layer 1: x2 x1
    % layer 2: n1 n2
    % layer 3: n3
    layers = [2 2 1];
    nn = build_nn(layers);

    % set weights
    w = [20 20 -30]';
    nn = set_weights(nn, 1, 1, w);
    w = [-20 -20 10]';
    nn = set_weights(nn, 1, 2, w);
    w = [20 20 -10]';
    nn = set_weights(nn, 2, 1, w);
    
    % pred some values
    xpred = [0 0; 0 1; 1 0; 1 1];
    ypred = pred_nn(nn, xpred);
    disp(ypred);
end

% neural net that performs logical NOT
function not_nn()
    % layer 1: x1
    % layer 2: n1 = output
    layers = [1 1];
    nn = build_nn(layers);
        
    % set weights
    w = [-20 10]';
    nn = set_weights(nn, 1, 1, w);
    
    % pred some values
    xpred = [0; 1];
    ypred = pred_nn(nn, xpred);
    disp(ypred);
end

% neural net that performs logical OR
function or_nn()
    % layer 1: x2 x1
    % layer 2: n1 = output
    layers = [2 1];
    nn = build_nn(layers);
        
    % set weights
    w = [20 20 -10]';
    nn = set_weights(nn, 1, 1, w);
    
    % pred some values
    xpred = [0 0; 0 1; 1 0; 1 1];
    ypred = pred_nn(nn, xpred);
    disp(ypred);
end

% neural net that performs logical AND
function and_nn()
    % layer 1: x2 x1
    % layer 2: n1 = output
    layers = [2 1];
    nn = build_nn(layers);
    
    % set weights
    w = [20 20 -30]';
    nn = set_weights(nn, 1, 1, w);
    
    % pred some values
    xpred = [0 0; 0 1; 1 0; 1 1];
    ypred = pred_nn(nn, xpred);
    disp(ypred);
end

% simple example of predicting values for a nn
function simple_test1()
    % layer 1: x3 x2 x1 + bias x0 (3 features) - input
    % layer 2: node 1 (n1), node 2 (n2) - hidden
    % layer 3: node 3 (n3) - output
    % build nn from layers
    layers = [3 2 1];
    nn = build_nn(layers);
    
    % set the weights
    % w(1)*x3 + w(2)*x2 + w(3)*x1 + w(4) etc
    w = [0 2 4 5]';
    nn = set_weights(nn, 1, 1, w);
    w = [-2 1 -1 0.3]';
    nn = set_weights(nn, 1, 2, w);
    w = [-2 1 3]';
    nn = set_weights(nn, 2, 1, w);
    
    % predict some inputs
    xpred = [1.5 2 3.3; -1 0.5 -2];
    ypred = pred_nn(nn, xpred);
    disp(ypred);
end

% test functions to modify nn
function simple_test3()
    % layer 1: x2 x1 + bias x0 (2 features) - input
    % layer 2: node 1 (n1), node 2 (n2) - hidden
    % layer 3: node 3 (n3), node 4 (n4) - output
    % build nn from layers
    layers = [2 2 2];
    nn = build_nn(layers);
    
    % set the weights
    % w(1)*x3 + w(2)*x2 + w(3)*x1 + w(4) etc
    w = [1 2 3]';
    nn = set_weights(nn, 1, 1, w);
    w = [4 5 6]';
    nn = set_weights(nn, 1, 2, w);
    w = [7 8 9]';
    nn = set_weights(nn, 2, 1, w);
    w = [10 11 12]';
    nn = set_weights(nn, 2, 2, w);
    
    % get output weights
    w = get_output_weights(nn, 1, 2);
    disp(w);
end

% read train data (more generic)
function read_data(layers, fname)
    global xtrain ytrain Ntrain;
    data = load(fname);
    delim = layers(1);
    xtrain = data(:,1:delim);
    ytrain = data(:,delim+1:end);
    Ntrain = size(xtrain, 1);
end

% read train data
function read_data_test2()
    global xtrain xtrain0 xtrain1 ytrain n0 n1 Ntrain;
    data = load('../input/xnor_nn.in');
    xtrain = data(:,1:end-1);
    ytrain = data(:,end);
    xtrain0 = xtrain(ytrain == 0,:);
    n0 = size(xtrain0, 1);
    xtrain1 = xtrain(ytrain == 1,:);
    n1 = size(xtrain1, 1);
    Ntrain = size(xtrain, 1);
end

% returns random initial thetas for nn_global
function t0 = get_random_thetas(eps_theta)
    global nn_global;
    t0 = [];
    for i = 1:(size(nn_global.layers, 2) - 1)
        for j = 1:nn_global.layers(i+1)
            weights = rand(nn_global.layers(i) + 1, 1) * ...
                    (2 * eps_theta) - eps_theta;
            t0 = [t0; weights];
        end
    end
end

% train by minimizing a cost function
function train_xnor_test2()
    % perform training several times, starting with diff weights
    maxiter = 5;
    cost_threshold = 0.05;
    for k = 1:maxiter
        % set initial weights in a vec (random small values)
        t0 = get_random_thetas(0.1);
        %t0 = [0.1; -1.2; 3.0; 0.5; -0.25; -1; 0.7; 1.0; -2.0];
        %t0 = [70.3062; 80.0590; -105.4745; -9.9347; -13.9525;
        %    4.6976; 179.9034; 166.0707; -50.9448];

        % fminunc with grad
        syms t;
        %tmpfun = @(t)(cost_with_grad(t, 0.001));
        tmpfun = @(t)(cost_with_grad(t, 0.0001));
        %tmpfun = @(t)(cost_with_grad(t));
        options = optimset('fminunc');
        options = optimset(options, 'GradObj', 'on', 'TolFun', ...
            1.0000e-10, 'TolX', 1.0000e-10, 'MaxIter', 1000, ...
            'MaxFunEvals', 1000, 'Display', 'off', 'OutputFcn', @outfun);

        %[tf, c] = fminunc(tmpfun, t0, options);
        [tf, c] = fmincg(tmpfun, t0, options);
        %if c < cost_threshold
        if c(end) < cost_threshold
            break;
        end
        plot_cost();
    end
    % set nn weights to tf
    unroll_thetas(tf);
end

% train function for test 4
function train_test4()
    % set initial weights in a vec (random small values)
    t0 = get_random_thetas(0.1);

    % fminsearch
    %[tf,~] = fminsearch(@cost_with_grad, t0);

    % fminunc with grad
    tmpfun = @(t)(cost_with_grad(t, 0.0001));
    options = optimset('fminunc');
    options = optimset(options, 'GradObj', 'on', 'Display', 'off', ...
        'OutputFcn', @outfun);
    [tf,~] = fminunc(tmpfun, t0, options);
    %[tf,~] = fmincg(tmpfun, t0, options);

    % set nn weights to tf
    unroll_thetas(tf);
end

% train function for test 4
% this is similar to test4 but will be moved to sep file
function train_test5()
    % set initial weights in a vec (random small values)
    t0 = get_random_thetas(0.1);

    % fminsearch
    %[tf,~] = fminsearch(@cost_with_grad, t0);

    % fminunc with grad
    tmpfun = @(t)(cost_with_grad(t));
    options = optimset('fminunc');
    options = optimset(options, 'GradObj', 'on', 'Display', 'off', ...
        'OutputFcn', @outfun);
    [tf,~] = fminunc(tmpfun, t0, options);
    %[tf,~] = fmincg(tmpfun, t0, options);

    % set nn weights to tf
    unroll_thetas(tf);
end

% train function for test 6
function train_test6()
    % set initial weights in a vec (random small values)
    t0 = get_random_thetas(0.1);

    % fminunc doesn't work well, use fmincg
    tmpfun = @(t)(cost_with_grad(t));
    options = optimset('fminunc');
    options = optimset(options, 'GradObj', 'on', 'OutputFcn', @outfun, 'MaxIter', 100);
    [tf, ~] = fmincg(tmpfun, t0, options);

    % set nn weights to tf
    unroll_thetas(tf);
end

% cost function and gradient for nn_global
function [cost, grad] = cost_with_grad(t, lambda)
    % lambda is used for regularization
    
    global xtrain Ntrain ytrain nn_global;
    unroll_thetas(t);
    % old cost functions (1 output node only)
    % it's faster when we precompute xtrain0/1 etc
    %cost = (sum(-log(pred_nn(nn_global, xtrain1))) + ...
    %   sum(-log(1 - pred_nn(nn_global, xtrain0)))) / Ntrain;
    
    ypred = zeros(size(ytrain));
    deltas = zeros(size(nn_global.W));
    sz = size(xtrain, 1);
    for k = 1:sz
        nn_global = compute_val(nn_global, xtrain(k,:));

        % compute error on last layer
        layer = nn_global.num_layers - 1;
        num_nodes = nn_global.layers(layer + 1);
        ypred(k,:) = nn_global.val(layer + 1, 1:num_nodes);

        % skip backprop if we only compute cost
        if nargout <= 1
            continue;
        end
        nn_global.err(layer, 1:num_nodes) = ypred(k,:) - ytrain(k, :);

        % backprop error
        layer = layer - 1;
        while layer > 0
            num_nodes = nn_global.layers(layer+1);
            next_num_nodes = nn_global.layers(layer+2);
            % TODO this can be vectorized
            for num = 1:num_nodes
                w = get_output_weights(nn_global, layer, num);
                val = nn_global.val(layer+1,num);
                val = val * (1 - val);
                nn_global.err(layer, num) = nn_global.err(layer+1, 1:next_num_nodes) * w * val;
            end
            layer = layer - 1;
        end
        
        % update deltas
        for layer = 1:(nn_global.num_layers - 1)
            % num_weights is actually num_weights - 1
            for j = 1:nn_global.layers(layer+1)
                d = get_input_deltas(nn_global, deltas, layer, j);
                vals = [nn_global.val(layer, 1:nn_global.layers(layer)) 1]';
                d = d + vals * nn_global.err(layer, j);
                deltas = set_deltas(nn_global, deltas, layer, j, d);
            end
        end
    end

    % compute cost function
    cost = -(sum(sum(ytrain .* log(ypred))) + ...
        sum(sum((1 - ytrain) .* log(1 - ypred)))) / Ntrain;
    if nargin >= 2
        cost = add_cost_reg(cost, lambda);
    end
    if nargout <= 1
        return;
    end

    % roll deltas into grad
    grad = [];
    for layer = 1:(nn_global.num_layers - 1)
        for j = 1:nn_global.layers(layer+1)
                d = get_input_deltas(nn_global, deltas, layer, j) / Ntrain;
                grad = [grad; d];
        end
    end
    if nargin >= 2
        grad = add_grad_reg(grad, lambda);
    end
end

% grad with regularization
function grad = add_grad_reg(grad, lambda)
    global nn_global;
    gstart = 1;
    for i = 1:(nn_global.num_layers - 1)
        dindex = nn_global.layers(i);
        for j = 1:nn_global.layers(i+1)
            w = get_input_weights(nn_global, i, j);
            gend = gstart + dindex;
            grad(gstart:gend) = grad(gstart:gend) + lambda * w;
            gstart = gend + 1;
        end
    end
end

% cost with regularization
function cost = add_cost_reg(cost, lambda)
    global nn_global Ntrain;
    s = 0;
    for i = 1:(nn_global.num_layers - 1)
        for j = 1:nn_global.layers(i+1)
            w = get_input_weights(nn_global, i, j);
            s = s + sum(w(1:end-1) .^ 2);
        end
    end
    cost = cost + (lambda * s) / (2 * Ntrain);
end

% unrolls thetas from vec to cube
function unroll_thetas(t)
    global nn_global;
    last_index = 1;
    for i = 1:(nn_global.num_layers - 1)
        % num_weights is actually num_weights - 1
        num_weights = nn_global.layers(i);
        for j = 1:nn_global.layers(i+1)
            weights = t(last_index:last_index + num_weights);
            nn_global = set_weights(nn_global, i, j, weights);
            last_index = last_index + num_weights + 1;
        end
    end
end

% rolls thetas from cube to vec
function t = roll_thetas()
    global nn_global;
    t = [];
    for i = 1:(nn_global.num_layers - 1)
        for j = 1:nn_global.layers(i+1)
            weights = get_input_weights(nn_global, i, j);
            t = [t; weights];
        end
    end
end

% function that gets called after every step in fminunc
function stop = outfun(t, optimValues, state)
	% t = current values for thetas
	% optimValues.fval = function value for t
	% optimValues.iteration = iteration number
	% optimValues.procedure = procedure message
    %   what I print in cost_with_grad
	% optimValues.funccount = number of function evals until now
    global cost_vals;
    stop = false;
    if optimValues.iteration == 0
        cost_vals = [];
    end
    cost_vals = [cost_vals; optimValues.fval];
end

% compares gradient returned by fgrad with numerical one
function check_grad(fgrad, t0)
    % e.g. usage: check_grad(@cost_with_grad, t0);
    [~, grad] = fgrad(t0);
    sz = size(t0, 1);
    num_grad = zeros(sz, 1);
    eps = 0.0001;
    for i = 1:sz
        t0(i) = t0(i) + eps;
        [fp, ~] = fgrad(t0);
        t0(i) = t0(i) - (2 * eps);
        [fm, ~] = fgrad(t0);
        t0(i) = t0(i) + eps;
        num_grad(i) = (fp - fm) / (2 * eps);
    end
    disp(num_grad);
    disp(grad);
    disp(sum((num_grad - grad) .^ 2));
end

% function that builds a nn based on layers array
function nn = build_nn(layers)
    % function contract
    %   layers = list of number of nodes for each layer
    %   e.g. layers = [3 2 1]
    %       3 features on layers 1
    %       2 nodes on layer 2
    %       1 node on layer 3
    %
    % weights and functions can be set with other functions
    % set_weights(layer, num_node, w);
    % set_function(layer, num_node, f);
    % WARNING they are not set by default (all zeros)

    % sanity checks on layers TODO
    nn.layers = layers;
    nn.num_features = nn.layers(1);
    nn.num_layers = size(nn.layers, 2);

    % use cube for weights
    sz1 = nn.num_layers - 1;
    sz2 = max(nn.layers(1:end-1)) + 1;
    sz3 = max(nn.layers(2:end));
    nn.W = zeros(sz1, sz2, sz3);

    % set weights to some initial value ones()/zeros()
    %for i = 1:(size(nn.layers, 2) - 1)
    %    default_weight = ones(nn.layers(i) + 1,1);
    %    for j = 1:nn.layers(i+1)
    %        nn = set_weights(nn, i, j, default_weight);
    %    end
    %end
    
    % set possible functions & default one
    syms t;
    nn.fmap = containers.Map;
    nn.allowed_functions = containers.Map;
    
    % add multiple functions here
    nn.sigmoid = @(t)(1 / (1 + exp(-t)));
    nn.fmap('sigmoid') = nn.sigmoid;
    nn.allowed_functions('sigmoid') = true;
    nn.default_function = 'sigmoid';

    % set activation functions to default function
    sz1 = nn.num_layers - 1;
    sz2 = max(nn.layers(2:end));
    nn.F = cell(sz1, sz2);
    for i = 1:sz1
        for j = 1:nn.layers(i+1)
            nn = set_function(nn, i, j, nn.default_function);
        end
    end

    % backpropagation error
    nn.err = zeros(sz1, sz2);

    % initialize value matrix (or node output values)
    sz1 = nn.num_layers;
    sz2 = max(nn.layers);
    nn.val = zeros(sz1, sz2);
end

function nn = set_weights(nn, layer, num, w)
    % function contract
    %   nn = neural network (from build_nn_generic)
    %   layer = layer of the node; counting starts after features
    %           e.g. layer 1 is the first hidden layer/output layer
    %   num = node number in that layer; counting starts at the top
    %   w = column vector that represents the new weights
    
    % TODO sanity checks on nn, layer, num & w
    
    % nn.W(layer,:,:) = matrix for our layer
    % nn.W(layer,:,num) = weight vector for our node
    % weight vector has only `nn.layers(layer) + 1` values
    % nn.W(layer,:,num) can have more than that, so we only
    % take the first `nn.layers(layer) + 1`
    nn.W(layer, 1:(nn.layers(layer)+1), num) = w;
end

function w = get_input_weights(nn, layer, num)
    % function contract
    %   input
    %       nn = neural network (from build_nn_generic)
    %       layer = layer of the node; counting starts after features
    %           e.g. layer 1 is the first hidden layer/output layer
    %       num = node number in that layer; counting starts at the top
    %   output
    %       w = weights that go inside node from (layer,num)
    %           as a column vector
    
    % TODO sanity checks on nn, layer & num
    
    % nn.W(layer,:,:) = matrix for our layer
    % nn.W(layer,:,num) = weight vector for our node
    % weight vector has only `nn.layers(layer) + 1` values
    % nn.W(layer,:,num) can have more than that, so we only
    % take the first `nn.layers(layer) + 1`
    % TODO explain transpose (some weird matlab thing maybe)
    w = nn.W(layer, 1:(nn.layers(layer)+1), num)';
end

function w = get_output_weights(nn, layer, num)
    % function contract
    %   input
    %       nn = neural network (from build_nn_generic)
    %       layer = layer of the node; counting starts after features
    %           e.g. layer 1 is the first hidden layer/output layer
    %       num = node number in that layer; counting starts at the top
    %   output
    %       w = weights that go outside the node at (layer,num)
    %           as a column vector
    
    % TODO sanity checks on nn, layer & num
    
    % nn.W(layer+1,:,:) = matrix for next layer
    % nn.W(layer,num,:) = weights that go outside from the node
    % weight vector has only `nn.layers(layer+2)` values
    % nn.W(layer,num,:) can have more than that, so we only
    % take the first `nn.layers(layer+2)`
    w = squeeze(nn.W(layer + 1, num, 1:nn.layers(layer+2)));
end

function d = get_input_deltas(nn, deltas, layer, num)
    % similar to get_input_weights
    d = deltas(layer, 1:(nn.layers(layer)+1), num)';
end

function deltas = set_deltas(nn, deltas, layer, num, d)
    % similar to set_weights
    deltas(layer, 1:(nn.layers(layer)+1), num) = d;
end

% function that shows all weights for nn_global
function display_weights()
    % all the weights will be displayed one under another
    % starting from left layer, top node
    global nn_global;
    for i = 1:(nn_global.num_layers - 1)
        for j = 1:nn_global.layers(i+1)
            w = get_input_weights(nn_global, i, j);
            disp(w');
        end
    end
end

% reads weights values from fname (.mat) to nn_global
function load_weights_mat(fname)
    global nn_global;
    weights = load(fname);
    nn_global.W = weights.W;
end

% reads weights values from fname (.in) to nn_global
% TODO implement this
function load_weights(fname)
    global nn_global;
    %data = load(fname);
    %disp(data);
end

% stores weights/params from nn to a .mat file
function store_weights_mat(fname)
    % fname shouldn't have an extension
    % e.g. store_weights_mat(layers, 'my_weights.in');
    global nn_global;
    W = nn_global.W;
    save(fname, 'W');
end

% stores weights/params from nn to a file
function store_weights(fname)
    global nn_global;
    fid = fopen(fname, 'w');
    for i = 1:(nn_global.num_layers - 1)
        for j = 1:nn_global.layers(i+1)
            w = get_input_weights(nn_global, i, j);
            for k = 1:size(w)
                fprintf(fid, '%d ', w(k));
            end
            fprintf(fid, '\n');
        end
    end
    fclose(fid);
end

function nn = set_function(nn, layer, num, fname)
    % function contract
    %   nn = neural network (from build_nn_generic)
    %   layer = layer of the node; counting starts after features
    %           e.g. layer 1 is the first hidden layer/output layer
    %   num = node number in that layer; counting starts at the top
    %   fname = string representing the new activation function for
    %           that node
    %       possible values:
    %           'sigmoid' = 1/(1 + exp(-x))
    
    % TODO sanity checks on nn, layer, num & f

    % nn.F(layer, num) = function for the node
    if nn.allowed_functions.isKey(fname)
        nn.F(layer, num) = {fname};
    else
        nn.F(layer, num) = {nn.default_function};
    end
end

function f = get_function(nn, layer, num)
    % function contract
    %   nn = neural network (from build_nn_generic)
    %   layer = layer of the node; counting starts after features
    %           e.g. layer 1 is the first hidden layer/output layer
    %   num = node number in that layer; counting starts at the top
    
    % TODO sanity checks on nn, layer & num
    
    % nn.F(layer, num) = function for the node
    fname = char(nn.F(layer, num));
    if nn.fmap.isKey(fname)
        f = nn.fmap(fname);
    else
        f = nn.fmap(nn.default_function);
    end
end

% returns value from node num from specified layer,
% with xs as inputs
function val = get_value(nn, xs, layer, num)
    if layer == 1
        val = xs(num);
        return;
    end
    vals = ones(1, nn.layers(layer-1) + 1);
    for i = 1:nn.layers(layer-1)
        vals(i) = get_value(nn, xs, layer-1, i);
    end
    w = get_input_weights(nn, layer-1, num);
    f = get_function(nn, layer-1, num);
    val = f(vals * w);
end

% returns nn values for xpred inputs
% this is much slower than pred_nn_it
% for multiple output layers it recomputes values
function ypred = pred_nn(nn, xpred)
    sz = size(xpred, 1);
    ypred = zeros(sz, nn.layers(end));
    for i = 1:sz
        for j = 1:nn.layers(end)
            ypred(i,j) = get_value(nn, xpred(i,:), nn.num_layers, j);
        end
    end
end

% computes val matrix and stores it inside nn
function nn = compute_val(nn, xpred)
    % xpred is a single input (one line of features)
    nn.val(1,1:nn.layers(1)) = xpred;
    for i = 2:nn.num_layers
        for j = 1:nn.layers(i)
            w = get_input_weights(nn, i-1, j);
            f = get_function(nn, i-1, j);
            vals = [nn.val(i-1,1:nn.layers(i-1)) 1];
            nn.val(i,j) = f(vals * w);
        end
    end
end

% returns nn values for xpred inputs (iterative version)
function ypred = pred_nn_it(nn, xpred)
    sz = size(xpred, 1);
    ypred = zeros(sz, nn.layers(end));
    for k = 1:sz
        nn = compute_val(nn, xpred(k,:));
        for j = 1:nn.layers(end)
            ypred(k,j) = nn.val(nn.num_layers, j);
        end
    end
end

% plots evolution of cost function
function plot_cost()
    global cost_vals;
    figure;
    hold on;
    plot((1:size(cost_vals, 1))', cost_vals);
end

% plot function for test 2
function plot_all_test2()
    global xtrain0 xtrain1 xtrain nn_global; %xpred ypred w sigmoid;

    % plot initial data
    figure;
    hold on;
    plot(xtrain0(:,1), xtrain0(:,2), 'bo');
    plot(xtrain1(:,1), xtrain1(:,2), 'rx');

    % set range for x/y axis
    xmin = min(xtrain(:,1));
    xmax = max(xtrain(:,1));
    ymin = min(xtrain(:,2));
    ymax = max(xtrain(:,2));
    xlim([xmin-1 xmax+1]);
    ylim([ymin-1 ymax+1]);
    
    % plot 3d fit
    delta = 0.1;
    [wxs,wys] = meshgrid(xmin:delta:xmax, ymin:delta:ymax);
    sz = size(wxs);
    cwxs = reshape(wxs, sz(1)*sz(2), 1);
    cwys = reshape(wys, sz(1)*sz(2), 1);
    cwzs = pred_nn_it(nn_global, [cwxs cwys]);
    wzs = reshape(cwzs, sz);
    %wzs = arrayfun(sigmoid, wxs .* w(1) + wys .* w(2) + w(3));
    surf(wxs, wys, wzs);
end

% plot function for test 4
function plot_test4(xpred, ypred)
    global xtrain ytrain Ntrain;

    % plot initial data
    figure;
    hold on;
    for i = 1:Ntrain
        if ytrain(i,1) == 1
            plot(xtrain(i,1), xtrain(i,2), 'bo');
        elseif ytrain(i,2) == 1
            plot(xtrain(i,1), xtrain(i,2), 'rs');
        elseif ytrain(i,3) == 1
            plot(xtrain(i,1), xtrain(i,2), 'gd');
        elseif ytrain(i,4) == 1
            plot(xtrain(i,1), xtrain(i,2), 'y^');
        else
            plot(xtrain(i,1), xtrain(i,2), 'kx');
        end
    end

    % set range for x/y axis
    xmin = min(xtrain(:,1));
    xmax = max(xtrain(:,1));
    ymin = min(xtrain(:,2));
    ymax = max(xtrain(:,2));
    xlim([xmin-1 xmax+1]);
    ylim([ymin-1 ymax+1]);
    
    % plot predicted values
    [~, c] = max(ypred, [], 2);
    for i = 1:size(ypred, 1)
        if c(i) == 1
            plot(xpred(i,1), xpred(i,2), 'bx');
        elseif c(i) == 2
            plot(xpred(i,1), xpred(i,2), 'rx');
        elseif c(i) == 3
            plot(xpred(i,1), xpred(i,2), 'gx');
        elseif c(i) == 4
            plot(xpred(i,1), xpred(i,2), 'yx');
        else
            plot(xpred(i,1), xpred(i,2), 'kx');
        end
    end
end

% plot function for test 5
function plot_test5(xpred, ypred)
    global xtrain ytrain nn_global;
    xs0 = xtrain(ytrain == 0);
    xs1 = xtrain(ytrain == 1);
    
    figure;
    hold on;
    plot(xs0, 0, 'bo');
    plot(xs1, 1, 'rx');
    
    % set range for x/y axis
    xmin = min(xtrain);
    xmax = max(xtrain);
    xlim([xmin-1 xmax+1]);
    ylim([-0.5 1.5]);
    
    % plot best fit
    delta = 0.1;
    wxs = (xmin:delta:xmax)';
    wys = pred_nn(nn_global, wxs);
    plot(wxs, wys, 'k-');
    
    % plot predicted points
    for k = 1:size(xpred, 1)
        if ypred(k) >= 0.5
            plot(xpred(k), 1, 'go');
        else
            plot(xpred(k), 0, 'go');
        end
    end
    %plot(xpred, ypred, 'go');
end