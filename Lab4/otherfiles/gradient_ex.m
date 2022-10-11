%% 
% Find the gradient magnitude of IMG3.
h2x = [-1 -2 -1 ;  0  0  0 ;  1  2  1]
h2y = [-1  0  1 ; -2  0  2 ; -1  0  1]
resX = conv2(double(img1), h2x); % NOTE: DOUBLE type conversion
resY = conv2(double(img1), h2y);
resXY = sqrt(resX.^2 + resY.^2);

% Display the gradient magnitude, but not like this:
figure, imshow(resXY);
title('Not like this');
% because the assumed range of the DOUBLE type is [0,1],
% but e.g. one of these ways:
figure, imshow(resXY, []);
title('But like this,');
figure, imshow(resXY, [min(resXY(:)) max(resXY(:))]);
title('this')
figure, imshow(resXY/max(resXY(:)));
title('this or')
figure, imshow(uint8(resXY/max(resXY(:)).*255));
title('this. They are all equal.');


%%%%%%%%%%%%%%%%