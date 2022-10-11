scene = imread('waldoskiing.jpg');
waldo = imcrop(scene, [834.5 261.5 120 124]);
montage({waldo,scene})

%Perform correlation and display the result as surface
c = imfilter(double(rgb2gray(scene)), double(rgb2gray(waldo)));
figure, surf(c)
shading flat

%Find the peak in cross-correlation.
[y,x] = find(c==max(c(:)));

yoffset = y-size(waldo,1)/2;
xoffset = x-size(waldo,2)/2;
%Highlight waldo's location
figure, imshow(scene)
drawrectangle(gca,'Position',[xoffset,yoffset,size(waldo,2),size(waldo,1)], ...
    'FaceAlpha',0, 'Color','b');

%% EXERCISE 4 
%%%%%%%%%%%%%%%%

% Normalize resXY such that max(resXY(:)) = 255 and min(resXY(:)) = 0.
% Threshold the result with T = 100.
%
% What would you do if you wanted to obtain an image containing
% only the seam, and the entire seam, of the the ball?
% post_id = 213; %delete this line to force new post;
% permaLink = http://inf4300.olemarius.net/2015/09/02/week-1-introduction-to-matlab-this-weeks-assignment-is-a-quick-introduction-to-matlab-for-image-analysis-the-original-file-is-found-under-the-undervisningsplan-on-the-course-web-site-the-s/;
