data = csvread('forestfires2.csv', 1);
x = data(:,1:9);
y = data(:, end);

C = cov(x);
[coeff, score, latent] = pca(x);

n = size(data, 1);

xstar = data(1:272,1:8);
ystar = data(1:272, 10);
testx = data(:, 1:8);

lm = fitlm(xstar, ystar, 'linear');

lm2 = fitlm(score(1:272,1:4), ystar, 'linear');

meanfunc = {@meanLinear};            
  
covfunc = {@covSEiso};             
% covfunc = {@covMaterniso, 1};             
% covfunc = {@covMaterniso, 3};             
% covfunc = {@covNoise};             
  
likfunc = @likGauss;
%hyp_mean = [6.7305e-11; -1.0414; 1.6118; 1.6143; -1.1843; 0.00015347; -2.7006e-05];
%hyp_mean = [3.2039e-08] ;
hyp_mean = lm.Coefficients.Estimate(2:end);

hyp = struct('mean', hyp_mean, 'cov', [log(1/sqrt(2)) log(1)], 'lik', log(1e-4));

hyp = minimize(hyp, @gp, -100, [], meanfunc, covfunc, [], ...
                 xstar, ystar);
             
[mu s2] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, xstar, ystar, testx);

figure
f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
fill([testx(:,1); flipdim(testx(:,1),1)], f, [7 7 7]/8)
hold on; plot(testx(:,1), mu); plot(x(:,1), y, '+')
%axis([x(200,1) x(end,1) 8 14.5])
title('Predictive mean and 95% credibal interval');
legend('credibal interval','u(x)','observation','Location', 'northeastoutside');
xlabel('Recorded Timestamp in second');
ylabel('Weighted Price');

    
    

