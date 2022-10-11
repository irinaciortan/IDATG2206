%% Image filtering - 1
% Author: Irina Ciortan

% This week we'll start with basic image filtering and image enhancements.
% We'll cover mainly point transformations.

%% Gamma correction 
% Is a nonlinear operation, that can be defined by the power
% law expression, where gamma is the power we raise the intensity values
% to. If gamma is higher than unity, then the mapping is weighted to
% lower (darker) values. If gamma is lower than unity, then the mapping is
% weighted to higher (brighter) values.
img = imread('pout.tif');
figure, imshow(img);
figure, imhist(img); % the histogram shows low contrast because all values 
%are stacked in the middle


img2 = im2double(img).^1.5;
img3 = im2double(img).^0.5;
figure
subplot(131), imshow(img), title('Original')
subplot(132), imshow(img2), title('Gamma =1.5')
subplot(133), imshow(img3), title('Gamma =0.5')

%% Adjust limits of the intensity values with linear scaling 
% One way to do this is to use Matlab interface using function imtool.
% When the window opens, click on the "Adjust contrast" icon (the black and
% white circle). A new window opens, that shows the histogram. Draw the red
% lines to the center, surrounding the stack of values in the histogram.
% Then click "Adjust data" and watch the result.
imtool(img)


% Or use imadjust function.
img4 = imadjust(img); % By default, imadjust saturates the bottom 1% and the top 1% of all pixel values.
figure, imshow(img4), title('Linear scaling');

img5 = imadjust(img,[0.3 0.7],[]); % user-specified range of input-output values.
figure, imshow(img5), title('Linear scaling user-specified limits');

% Now use imadjust function for color image
imgc = imread('car1.jpg');
figure, imshow(imgc);

imgc_enhanced = imadjust(imgc,[.1 .2 .1; .9 .8 .9],[]);
figure
imshow(imgc_enhanced), title ('Enhanced car');

%% Histogram equalization 
% Thinking of the histogram as the probability distribution of an image, then 
% to improve the contrast, we'd want the histogram to be more evenly distributed.
% So we spread the most frequent pixel intensity values.
[imgceq, m] = histeq(imgc);
figure
subplot(221), imshow(imgc), title('Original');
subplot(222), imshow(imgceq), title('Equalized');
subplot(223), imhist(imgc), title('Histogram original image');
subplot(224), imhist(imgceq), title('Histogram equalized image');

[imgeq, m] = histeq(img);
figure
subplot(221), imshow(img), title('Original');
subplot(222), imshow(imgeq), title('Equalized');
subplot(223), imhist(img), title('Histogram original image');
subplot(224), imhist(imgeq), title('Histogram equalized image');

plane = imread('liftingbody.png');

[planeq, m] = histeq(plane);
figure
subplot(131), imshow(plane), title('Original');
subplot(132), plot(m), title('Cumulative distribution function');
subplot(133), imshow(planeq), title('Histogram equalized image');

% Visualize the difference between the histogram of the image before and
% after histogram equalization. Notice how the equalized histogram is more
% "spread" along all intensity spectrum instead of centered around the
% middle.
figure, imhist(plane), title('Original histogram');
figure, imhist(planeq), title('Equalized histogram');
% Now we enhanced pixels in the washed-out background/sky at the expense of
% the pixels on the plane, that look all saturated.
% Histogram equalization (histeq) works globally. Instead, we can use a
% local mapping, called "adaptive histogram equalization". 
planeqa = adapthisteq(plane);
figure, imshow(planeqa), title('Adaptive Histogram Equalization');
% This way, contrast is preserved for both foreground and background.
figure, imhist(planeqa), title('Histogram - Adaptive equalization');
% We can see that the histogram looks less flattened, but nonetheless
% evenly distributed.

%% Exercise: Make your own histogram equalization function
% Based on the helper function file myhisteq.m, you need to fill in
% correspondingly (places indicated) in order to achieve the computation of the histogram
% equalization.
% After you write the function, display the ouputs and compare them with
% those provided by matlab.
% Plot as well the cumulative distribution function (yours and the one
% given by the matlab function) using plot and "hold on" to plot both curves.
% Do the equalized histograms (your implementation and Matlab's) look the same? 
% Search in Matlab documentation of 'histeq' and try to understand what might 
% be the reason for the difference.
