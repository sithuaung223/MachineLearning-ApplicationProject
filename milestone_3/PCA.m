data = csvread('forestfires2.csv', 1);
x = data(:,1:9);
y = data(:, end);

C = cov(x);
[coeff, score, latent] = pca(x);
x1 = score(:,1);
x2 = score(:, 2);
classes = y;

figure
scatter(x1(classes == 1), x2(classes == 1), 10,  'r', '*');
hold on;
scatter(x1(classes == 0), x2(classes == 0), 10,  'b', '+');
title('Big fire (red) vs small fire (blue)');
xlabel('Principal component 1');
ylabel('Principal component 2');

%plot(score(:,1), y , '+');
%axis([-1000 500 0 ]);

n = size(latent, 1);
proVar = [];
for i=1:n
    v = latent(i)/sum(latent);
    proVar = [proVar; [i v]];
end

figure
plot(proVar(:,1), proVar(:,2));
title('Proportion of variance as a function of number of principal component');
xlabel('Proportion of variance');
ylabel('Number of principal component');

xstar = data(:,1:8);
ystar = data(:, 10);

lm = fitlm(xstar, ystar, 'linear');

lm2 = fitlm(score(:,1:4), ystar, 'linear');

