% This script compares the massive Gaussian Naive Bayes (massive_gnb.m) 
% with the matlab 'classify' function with diagonal covariance matrix. 
% written by Marlis Ontivero-Ortega,   Agustin Lage-Castellanos, Mitchell Valdes-Sosa: (Cuban Center for Neuroscience) 
% and        Giancarlo Valente, Rainer Goebel             (Department of Cognitve Neuroscience, Maastricht University)
clc; clear all;close all
disp('Running test......................................');
load('testdata.mat')
%% parameters
n = size(X,1);       % sample size
ntr = 80;            % train sample size 
nte = n-ntr;         % test sample size 
% depending of time available: (matlab classify function is slow)
s = size(neigs,2);  %  if you want to test for all of searchlights 
% s = 5000;            % if only a subset of searchlights enter into the analysis because the matlab classify function is slow
type = 'diagLinear'; % options: 'diagQuadratic' or 'diagLinear'
%% massive Gaussian Naive Bayes
tic;
[cHat, cita] = massive_gnb(X(ntr+1:end,:), X(1:ntr,:), Y(1:ntr), neigs(:,1:s), type);
time_lda = toc;
%% comparing with Matlab implementation (matlab 'classify' function)
tic;
cHat_it = zeros(nte,s);
for itlda = 1:s
    neigs_it = find(neigs(:,itlda));
    X_it = X(:,neigs_it);
    cHat_it(:,itlda) = classify(X_it(ntr+1:end,:), X_it(1:ntr,:), Y(1:ntr,:), type);
end
time_lda_it = toc;
%% Compare labels
disp(['Number of searchlights: ' num2str(s)]);
disp(['massive    diagonal-GNB CPU time: ' num2str(time_lda)    ' seconds']);
disp(['Sequential diagonal-GNB CPU time: ' num2str(time_lda_it) ' seconds']);
alls=numel(cHat);
eqs=sum(cHat(:)~=cHat_it(:));
perc=eqs/alls*100;
disp(['The massive and the sequential ' type '-GNB differs in ' num2str(perc) '% of the ' num2str(alls) ' predicted labels']);