% Mihail Dunaev
% May 2017

function w = train(w0, fun)
    % function that trains a linear regression model
    %   input
    %       w0 = initial values for the params
    %       fun = function that returns a cost & grad value
    %   output
    %       w = params for the model
    options = optimset('fminunc');
    options = optimset(options, 'GradObj', 'on', 'OutputFcn', @outfun);
    [w, ~] = fminunc(fun, w0, options);
end