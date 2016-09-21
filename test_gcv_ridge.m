clc;clear all;close all
n = 100;p = 100;
mu = 2.*randn(3,3);sd = [1,0.5 0.1];
beta = zeros(p,3);
grid = linspace(-10,10,100) ;
for it1 = 1:3
for it = 1:3
beta(:,it1) = beta(:,it1) + normpdf(grid,mu(it,it1),sd(it))';
end
end
% beta = randn(p,3);
X = randn(n,p);
s = 1;
e = s.*randn(n,3);
Y = X*beta +  e;
lambda_array = logspace(-2,1,100);
% lambda_array(:) = 1e-5;
%%
[ beta_gcv,GCV] = ridge_gcv(X,Y,lambda_array,eye(p),1);
figure('Color',[1,1,1])
subplot(221)
plot(lambda_array,GCV,'o-','LineWidth',2)
subplot(222),
plot(beta(:,1),'LineWidth',2)
hold on;plot(beta_gcv(:,1),'r-','LineWidth',2)
subplot(223),
plot(beta(:,2),'LineWidth',2)
hold on;plot(beta_gcv(:,2),'r-','LineWidth',2)
subplot(224),
plot(beta(:,3),'LineWidth',2)
hold on;plot(beta_gcv(:,3),'r-','LineWidth',2)