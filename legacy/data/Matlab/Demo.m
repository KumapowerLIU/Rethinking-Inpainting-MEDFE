% Demo script
% Uncomment each case to see the results 

I = (imread('imgs/Bishapur_zan.jpg'));
S = tsmooth(I,0.015,3);
figure, imshow(I), figure, imshow(S);

% I = (imread('imgs/graffiti.jpg'));
% S = tsmooth(I,0.015,3);
% figure, imshow(I), figure, imshow(S);

% I = (imread('imgs/crossstitch.jpg'));
% S = tsmooth(I,0.015,3);
% figure, imshow(I), figure, imshow(S);


% I = (imread('imgs/mosaicfloor.jpg'));
% S = tsmooth(I, 0.01, 3, 0.02, 5); 
% figure, imshow(I), figure, imshow(S);






