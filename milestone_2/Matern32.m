%this quote is modified from Roman Garnett at https://gist.github.com/rmgarnett/e43632647d529a949ddd26bdb7d7349c
% uses GPML toolkit: http://www.gaussianprocess.org/gpml/code/matlab/doc/


% condition on data
data = csvread('day.csv', 2, 3);
x = data(:,1:10);
y = data(:, end-1);
%x = rand(10, 1) * 10;
%y = exp(sin(x)) + 0.1 * randn(size(x));

% sample from a GP on a grid of 1000 points in the interval [0, 10]
mean_x = mean(x); 
x_star = linspace(0, 12, 1e3)';



% set mean function, see help meanFunctions for a list

% ?(x) = 0
mean_function = {@meanZero};
theta.mean = []; % no hyperparameters


% set covariance function, see help covFunctions for a list

% K(x, x') = ?²exp(-½|x - x'|²/?²)
covariance_function = {@covMaterniso, 3};
theta.cov = [log(1); log(1)]; % hyperparameters are [log ?; log ?]


% convenience function handles
mu = @(varargin) feval(      mean_function{:}, theta.mean, varargin{:});
K  = @(varargin) feval(covariance_function{:}, theta.cov,  varargin{:});

% prior for f(X*)
prior_mean       = mu(x_star);
prior_covariance =  1/2*(K(x_star, x_star) + transpose(K(x_star, x_star))); 
%diag(ones(1,size(prior_covariance,2))*(10.^-6))


% observation model, see help likFunctions for a list

% default is p(y | f) = N(y; f, ?²)
theta.lik = log(0.5);

% learn hyperparameters by maximizing log p(y | X, ?)
theta = minimize(theta, @gp, -100, [], mean_function, covariance_function, [], ...
                 x, y);

[predictive_mean, predictive_variance] = ...
    gp(theta, [], mean_function, covariance_function, [], x, y, x_star);

predictive_std = sqrt(predictive_variance);

figure(3);
clf;
hold('on');
fill([x_star; flipud(x_star)], ...
     [predictive_mean + 2 * predictive_std; ...
      flipud(predictive_mean - 2 * predictive_std)], ...
     [166, 206, 227] / 255);
plot(x, y, '.', 'markersize', 5);
plot(x_star, predictive_mean, ...
     'color', [31, 120, 180] / 255);
xlabel('x: different weather element: temp, humid,...')
ylabel('y: bike sharing count');
title('GP using Matern 3/2 Kernel on number of bikes sharing depended on the different weather')
fprintf('this model gave log marginal likelihood = %0.3f\n', ...
        gp(theta, [], mean_function, covariance_function, [], x, y));
    