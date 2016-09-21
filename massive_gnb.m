% Description: massive Gaussian Naive Bayes (pGNB) for all searchligths, 
%              equivalent to 'classify' matlab function with 'diagLinear'
%              option and uniform priors for each class. 
%
% Input:
%   Xtest   Test data (matrix nSamplesTest x nVox).
%   X       Training data (matrix nSamplesTrain x nVox).
%   Y       Labels (vector nSamples  x 1): Y should be in the set [-1,1].
%   neigs   Searchlight sparse structure (size: nVox x nSearchlights).
%           Sij = 1: if voxel 'i' is present in searchlight 'j'.  
%   type    Type of discriminant function: 'diagLinear' or 'diagQuadratic'. 
%
% Output:
%   cHat    Predicted labels for the test set (matrix nSamplesTest x
%           nSearchlights).
%   LINX    Discriminative function score for the test data for each class 
%           (nclass x nSamplesTest x nSearchlights). 
%

% Cuban Center for Neuroscience and the Department of Cognitve Neuroscience, Maastricht University
% 2015, version 1.0
% written by Marlis Ontivero,   Agustin Lage-Castellanos, Mitchell Valdes-Sosa: (Cuban Center for Neuroscience) 
% and       Giancarlo Valente, Rainer Goebel (Department of Cognitve
% Neuroscience, Maastricht University)


function [cHat, LINX, MU,S1] = massive_gnb(Xtest, X, Y, neigs, type)

    [n,p] = size(X);

    L = dummyvar(Y);
    c = size(L,2);
    nc = sum(L);
    
    % individual voxel contributions (mean and std)
    MU = (diag(1./nc)*L')*X; % mean for each voxel
    Xc = X - MU(Y,:); % centering data, mean=0
    
    switch type
        case 'diagLinear'
            S = sqrt(sum(Xc.^2)./(n-c)); % standard deviation
        case 'diagQuadratic'
            S = zeros(c,p);
            for i = 1:c
                S(i,:) = sqrt(sum(Xc(logical(L(:,i)),:).^2)./(nc(i)-1)); % standard deviation
            end
        otherwise
            error([type ' is not a valid value for the TYPE argument.']);
    end
           
    % Classification
    logDetSigma = 2*log(S)*neigs; % avoid over/underflow    
    ntest = size(Xtest,1);   
    ndisk = size(neigs,2);
    S1 = 1./S;
    
    LINX = zeros(ntest*c, p); % discriminant function (Xtest - MU)/SD for each voxel for each class
    switch type
        case 'diagLinear'
            for i = 1:c
                LINX(i:c:end,:) = bsxfun(@times, bsxfun(@minus, Xtest, MU(i, :)), S1);
            end
            LINX = (LINX.*LINX)*neigs; % sum across searchlights
            LINX =  .5 * bsxfun(@plus, LINX, logDetSigma);  
        case 'diagQuadratic'
            for i = 1:c
                LINX(i:c:end,:) = bsxfun(@times, bsxfun(@minus, Xtest, MU(i, :)), S1(i,:));
            end
            LINX = (LINX.*LINX)*neigs; % sum across searchlights
            for i = 1:c
                LINX(i:c:end,:) = .5 * bsxfun(@plus,  LINX(i:c:end,:), logDetSigma(i,:));  
            end
    end
        
    LINX = log(1/c)-LINX;  %uniform prior for each class
    LINX = reshape(LINX, c, ntest, ndisk);
    [kk, cHat] = max(LINX, [], 1); % assign classes
    cHat = squeeze(cHat); % class labels for the test set
end
