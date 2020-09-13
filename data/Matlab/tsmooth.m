function S = tsmooth(I,lambda,sigma,sharpness,maxIter)
%tsmooth - Structure Extraction from Texture via Relative Total Variation
%   S = tsmooth(I, lambda, sigma, maxIter) extracts structure S from
%   structure+texture input I, with smoothness weight lambda, scale
%   parameter sigma and iteration number maxIter. 
%   
%   Paras: 
%   @I         : Input UINT8 image, both grayscale and color images are acceptable.
%   @lambda    : Parameter controlling the degree of smooth.  
%                Range (0, 0.05], 0.01 by default.
%   @sigma     : Parameter specifying the maximum size of texture elements.
%                Range (0, 6], 3 by defalut.                       
%   @sharpness : Parameter controlling the sharpness of the final results,
%                which corresponds to \epsilon_s in the paper [1]. The smaller the value, the sharper the result. 
%                Range (1e-3, 0.03], 0.02 by defalut.   
%   @maxIter   : Number of itearations, 4 by default.
%            
%   Example
%   ==========
%   I  = imread('Bishapur_zan.jpg');
%   S  = tsmooth(I); % Default Parameters (lambda = 0.01, sigma = 3, sharpness = 0.02, maxIter = 4)
%   figure, imshow(I), figure, imshow(S);
%
%   ==========
%   The Code is created based on the method described in the following paper 
%   [1] "Structure Extraction from Texture via Relative Total Variation", Li Xu, Qiong Yan, Yang Xia, Jiaya Jia, ACM Transactions on Graphics, 
%   (SIGGRAPH Asia 2012), 2012. 
%   The code and the algorithm are for non-comercial use only.
%  
%   Author: Li Xu (xuli@cse.cuhk.edu.hk)
%   Date  : 08/25/2012
%   Version : 1.0 
%   Copyright 2012, The Chinese University of Hong Kong.
% 

    if (~exist('lambda','var'))
       lambda=0.01;
    end   
    if (~exist('sigma','var'))
       sigma=3.0;
    end 
    if (~exist('sharpness','var'))
        sharpness = 0.02;
    end
    if (~exist('maxIter','var'))
       maxIter=4;
    end    
    I = im2double(I);
    x = I;
    sigma_iter = sigma;
    lambda = lambda/2.0;
    dec=2.0;
    for iter = 1:maxIter
        [wx, wy] = computeTextureWeights(x, sigma_iter, sharpness);
        x = solveLinearEquation(I, wx, wy, lambda);
        sigma_iter = sigma_iter/dec;
        if sigma_iter < 0.5
            sigma_iter = 0.5;
        end
    end
    S = x;      
end

function [retx, rety] = computeTextureWeights(fin, sigma,sharpness)

   fx = diff(fin,1,2);
   fx = padarray(fx, [0 1 0], 'post');
   fy = diff(fin,1,1);
   fy = padarray(fy, [1 0 0], 'post');
      
   vareps_s = sharpness;
   vareps = 0.001;

   wto = max(sum(sqrt(fx.^2+fy.^2),3)/size(fin,3),vareps_s).^(-1); 
   fbin = lpfilter(fin, sigma);
   gfx = diff(fbin,1,2);
   gfx = padarray(gfx, [0 1], 'post');
   gfy = diff(fbin,1,1);
   gfy = padarray(gfy, [1 0], 'post');     
   wtbx = max(sum(abs(gfx),3)/size(fin,3),vareps).^(-1); 
   wtby = max(sum(abs(gfy),3)/size(fin,3),vareps).^(-1);   
   retx = wtbx.*wto;
   rety = wtby.*wto;

   retx(:,end) = 0;
   rety(end,:) = 0;
   
end

function ret = conv2_sep(im, sigma)
  ksize = bitor(round(5*sigma),1);
  g = fspecial('gaussian', [1,ksize], sigma); 
  ret = conv2(im,g,'same');
  ret = conv2(ret,g','same');  
end

function FBImg = lpfilter(FImg, sigma)     
    FBImg = FImg;
    for ic = 1:size(FBImg,3)
        FBImg(:,:,ic) = conv2_sep(FImg(:,:,ic), sigma);
    end   
end

function OUT = solveLinearEquation(IN, wx, wy, lambda)
% 
% The code for constructing inhomogenious Laplacian is adapted from 
% the implementaion of the wlsFilter. 
% 
% For color images, we enforce wx and wy be same for three channels
% and thus the pre-conditionar only need to be computed once. 
% 
    [r,c,ch] = size(IN);
    k = r*c;
    dx = -lambda*wx(:);
    dy = -lambda*wy(:);
    B(:,1) = dx;
    B(:,2) = dy;
    d = [-r,-1];
    A = spdiags(B,d,k,k);
    e = dx;
    w = padarray(dx, r, 'pre'); w = w(1:end-r);
    s = dy;
    n = padarray(dy, 1, 'pre'); n = n(1:end-1);
    D = 1-(e+w+s+n);
    A = A + A' + spdiags(D, 0, k, k); 
    if exist('ichol','builtin')
        L = ichol(A,struct('michol','on'));    
        OUT = IN;
        for ii=1:ch
            tin = IN(:,:,ii);
            [tout, flag] = pcg(A, tin(:),0.1,100, L, L'); 
            OUT(:,:,ii) = reshape(tout, r, c);
        end    
    else
        OUT = IN;
        for ii=1:ch
            tin = IN(:,:,ii);
            tout = A\tin(:);
            OUT(:,:,ii) = reshape(tout, r, c);
        end    
    end
        
end