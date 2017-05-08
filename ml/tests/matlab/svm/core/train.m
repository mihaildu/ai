% Mihail Dunaev
% May 2017

function w = train(w0, fun)
    % generic training by minimizing 'fun'
    %   input
    %       w0 = initial params for 'fun'
    %       fun = function that returns a cost & grad value
    %   output
    %       w = params for the model
    options = optimset('fminunc');
    %options = optimset(options, 'GradObj', 'on');
    [w, ~] = fminunc(fun, w0, options);
end