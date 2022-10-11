%% Week 4: Types of noise and denoising filters. Edge detection.
% Author: Irina Ciortan
%% Gaussian white noise
% Gaussian noise is an additive noise, with a normal probability distribution function.
% Additive refers to the fact that the noise might be intrinsic to the information system.
% White refers to the idea that it has uniform power across the frequency band for the imaging system. 
% It is an analogy to the color white which has uniform emissions at all frequencies in the visible spectrum.
% Principal sources of Gaussian noise in digital images arise during acquisition 
% e.g. sensor noise caused by poor illumination and/or high temperature, 
% and/or transmission e.g. electronic circuit noise
% Gaussian noise can be best resolved with average filter and Gaussian
% smoothing. 


% Plot a standard normal distribution, with parameters
% mean(μ) equal to 0 and standard deviation(σ) equal to 1.
x = [-3:.1:3];
y = normpdf(x,0,1);
figure, plot(x, y), title('Gaussian function, μ=0, σ=1');

y1 =normpdf(x,0,3);
figure, plot(x, y1), title('Gaussian function, μ=0, σ=3');

% Read an image
img = imread('dogchick.jpg');
% imnoise function is used to apply various types of noise in Matlab.
noisyim = imnoise(img, 'gaussian');
noisyim2 =  imnoise(img, 'gaussian', 0.05);
noisyim3 =  imnoise(img, 'gaussian', 0.1);
noisyim4 =  imnoise(img, 'gaussian', 0.2);
figure, 
subplot(221), imshow(noisyim), title('Default Variance 0.01')
subplot(222), imshow(noisyim2), title('Variance 0.05')
subplot(223), imshow(noisyim3), title('Variance 0.1')
subplot(224), imshow(noisyim4), title('Variance 0.2')

%% Salt and Pepper noise
% Also known as impulse noise. This noise can be caused by sharp and sudden disturbances in the image signal. 
% In grayscale images, it presents itself as randomly occurring white and black (extreme) pixels.

spim = imnoise(rgb2gray(img), 'salt & pepper', 0.02);
spim2 = imnoise(rgb2gray(img), 'salt & pepper', 0.05);

figure, subplot(121), imshow(spim), title('S&P density 0.02')
subplot(122), imshow(spim2), title('S&P density 0.05')

% In color images, it presents as randomly occuring extreme pixels for each
% color channel (pure red, green, blue, black, white pixels)

spim_c = imnoise(img, 'salt & pepper', 0.02);
spim_c2 = imnoise(img, 'salt & pepper', 0.05);

figure, subplot(121), imshow(spim_c), title('S&P density 0.02')
subplot(122), imshow(spim_c2), title('S&P density 0.05')
%% Gaussian smoothing
% Gaussian filters are linear filters weigh neighboring pixels
% according to a bell-curve around the center pixel. This means that farther
% pixels get lower weights. Mathematically, applying a Gaussian blur to an image is the same as convolving
% the image with a Gaussian function. Gaussian function is also called normal distribution function.
% Gaussian smoothing is effective for reducing the Gaussian noise.

% When you are creating a Gaussian filter, the most important parameters you
% have to set are the σ and the filter size.
% The higher the sigma, the more spread the Gaussian curve is and the
% higher the blur the effect. Similarly, the higher the filter size, the
% higher the blur. You can read more about the computation process of
% Gaussian filter here: http://dev.theomader.com/gaussian-kernel-calculator/

% Create a Gaussian filter using the fspecial function.
% fspecial is a Matlab in-built function that implements several types of
% filters.
g1 = fspecial('gaussian', 5, 1)
g2 = fspecial('gaussian', 31, 1);

% Apply the Gaussian smoohting with imfilter to img
img1 = imfilter(img, g1);
img2 = imfilter(img, g2);

figure
subplot(221), imshow(img), title('Original')
subplot(222), imshow(img1), title('Filter size 5')
subplot(223), imshow(img2), title('Filter size 31')
subplot(224), imshow(img2-img, []), title('Difference')
% We notice that having a stable sigma and changing the filtersize only,
% the blur effect is not significantly higher. There is however, difference on the
% edges.

g1 = fspecial('gaussian', 5, 3);
g2 = fspecial('gaussian', 5, 5);
img1 = imfilter(img, g1);
img2 = imfilter(img, g2);
figure
subplot(221), imshow(img), title('Original')
subplot(222), imshow(img1), title('Sigma 3')
subplot(223), imshow(img2), title('Sigma 5')
subplot(224), imshow(img2-img, []), title('Difference')

%The increase in sigma affects more the blur effect.

%You can also apply the gaussian blur using the imgaussfilt functions.
% This functions creates the filter and runs the convolution operation, all
% in one take.
img1 = imgaussfilt(img,  5, 'FilterSize', 5);
img2 = imgaussfilt(img,  7, 'FilterSize', 5);
img3 = imgaussfilt(img,  7, 'FilterSize', 11);
figure
subplot(221), imshow(img), title('Original')
subplot(222), imshow(img1), title('Sigma 5 Size 5')
subplot(223), imshow(img2), title('Sigma 7 Size 5')
subplot(224), imshow(img3), title('Sigma 7 Size 11')

% Try average filtering and Gaussian filtering for various types of noise

restored_gs = imgaussfilt(noisyim2, 2);%restore with Gaussian smoothing
%restore with average filtering
avgfilt = fspecial('average', 5)
restored_avg = imfilter(noisyim2, avgfilt);
figure, subplot(221), imshow(img), title('Original')
subplot(222), imshow(noisyim2), title('Gaussian noise var 0.05')
subplot(223), imshow(restored_gs), title('Gaussian smoothing σ=2')
subplot(224), imshow(restored_avg), title('Average filtering 5x5')


restored1 = imgaussfilt(noisyim, 2);
figure, imshowpair(noisyim, restored1, 'montage'), title('Gaussian smoothing, low noise')

restored2 = imgaussfilt(noisyim4, 2);
figure,imshowpair(noisyim, restored2, 'montage'), title('Gaussian smoothing, high noise')

% As we can see, especially when the noise is high, we can't completely restore
% original information. We can only make the image less disturbing to view.

% What about salt and pepper noise restored with Gaussian smoothing?
spim_restoredgs = imgaussfilt(spim2, 2);
figure, imshowpair(spim2, spim_restoredgs, 'montage'), title('Gaussian smoothing for s&p')

spim_restoredgs_c = imgaussfilt(spim_c2, 2);
figure, imshowpair(spim_c2, spim_restoredgs_c, 'montage'), title('Gaussian smoothing for s&p')

%% Median filtering
% Median filtering is a nonlinear method used to remove noise from images.
% Being non-linear, it is different from any convolution or correlation
% operation.
% It is widely used as it is very effective at removing noise while preserving edges.
% It is particularly effective at removing ‘salt and pepper’ type noise.
% The median filter works by moving through the image pixel by pixel,
% replacing each value with the median value of neighbouring pixels.
% The median is calculated by first sorting all the pixel values from the window in numerical order,
% and then replacing the pixel being considered with the middle (median) pixel value.

% The function to perform median filtering on grayscale images in matlab is medfilt2.
med = medfilt2(rgb2gray(img)); % the default filter is square and of size 3x3.
med1 = medfilt2(rgb2gray(img), [7 11]); %the filter size can be changed to [m n] as the second argument.

figure, subplot(131), imshow(rgb2gray(img)), title('Original'),
subplot(132), imshow(med), title('Default median filtering 3x3');
subplot(133), imshow(med1), title('Median filtering 7x11'); %in this case, we have an anisotropic (not symmetric) filter

%For color images, we could use medfilt3 function that operates on 3D
%data. In this case the median value is extracted from a voxel. But this
%function is meant for volumetric data, not color images and it actually
%might change colors. The right way to apply median filtering for color
%images is to apply medfilt2 for each color channel (to do as TASK). 

med3 = medfilt3(img);
figure, imshowpair(img, med3, 'montage'), title('Medfilt3 on color images');

%Restore noise-corrupted images with median filtering
gausnois = imnoise(rgb2gray(img),'gaussian',  0.05);

noisyim_med = medfilt2(gausnois); %restore gaussian noise
spim_med = medfilt2(spim); %restore s&p noise

figure, subplot(221), imshow(gausnois), title('Gaussian noise var 0.05')
subplot(222), imshow(noisyim_med), title('Median filtering gaussian noise') 
subplot(223), imshow(spim), title('S&P density=0.05')
subplot(224), imshow(spim_med), title('Median filtering S&P')
%% Wiener filtering
% A type of linear filter, applied locally to images. The Wiener filter tailors 
% itself to the local image variance. Where the variance is large, it performs little smoothing. 
% Where the variance is small, it performs more smoothing.
% The adaptive filter is more selective than a comparable global linear filter, preserving edges and other high-frequency parts of an image.

% The function in matlab that applies wiener filtering is wiener2. It only
% works for grayscale images.


gausnois = imnoise(rgb2gray(img),'gaussian',  0.05);
gs = imgaussfilt(gausnois, 2);
wiena = wiener2(gausnois, [5 5]);

figure, subplot(131), imshow(wiena), title('Wiener filtering')
subplot(132), imshow(gausnois), title('Gaussian noise var 0.05');
subplot(133), imshow(gs),title('Gaussian smoothing σ=2');

wienb = wiener2(spim2, [5 5]);
spim2_med = medfilt2(spim2);
figure, subplot(131), imshow(wienb), title('Wiener filtering')
subplot(132), imshow(spim2), title('S&P noise density=0.05');
subplot(133), imshow(spim2_med),title('Median filtering');

%% Edge Detection.
% A directional filter (such as the Prewitt and Sobel filter) is an edge detector that can be used 
% to compute the first derivatives of an image. The first derivatives (or slopes) are 
% most evident when a large change occurs between adjacent pixel values.
% A Laplacian filter is an edge detector used to compute the second derivatives of an image, 
% measuring the rate at which the first derivatives change. This determines if a change in 
% adjacent pixel values is from an edge or continuous progression. 
% Laplacian filter kernels usually contain negative values in a cross pattern, centered within the array. 
% The corners are either zero or positive values. The center value can be either negative or positive.

% We can create most edge detector filters using fspecial function. 
% These are all linear filters.

op_prewitt = fspecial('prewitt')
op_sobel = fspecial('sobel') %sobel gives more weight to the adjacent pixels
op_laplacian = fspecial('laplacian')
edge_p = imfilter(rgb2gray(img), op_prewitt);
edge_s = imfilter(rgb2gray(img), op_sobel);
edge_lap = imfilter(rgb2gray(img), op_laplacian);
figure, subplot(221), imshow(rgb2gray(img)), title('Original')
subplot(222), imshow(edge_p), title('Prewitt Horizontal')
subplot(223), imshow(edge_s), title('Sobel Horizontal')
subplot(224), imshow(edge_lap), title('Laplacian')

edge_pv = imfilter(rgb2gray(img), op_prewitt');
edge_sv = imfilter(rgb2gray(img), op_sobel');
figure, imshowpair(edge_pv, edge_sv, 'montage'), title('Prewitt and Sobel vertical edges');

% You can also use edge function in matlab, that implements all the above
% methods and in addition Canny edge detection method.
% Moreover edge function computes Sobel and Prewitt in both vertical and
% horizontal directions. And the output is binary not grayscale.
ed_s =edge(rgb2gray(img), 'sobel');
ed_p = edge(rgb2gray(img), 'prewitt');
figure, imshowpair(ed_s, ed_p,'montage'), title('Sobel and Prewitt with edge function')

% Canny finds edges by looking for local maxima of the gradient of the image. 
% The edge function calculates the gradient using the derivative of a Gaussian filter. 
% This method uses two thresholds to detect strong and weak edges, including weak edges 
% in the output if they are connected to strong edges. By using two thresholds, the Canny method
% is less likely than the other methods to be fooled by noise, and more likely to detect true weak edges.

edge_canny = edge(rgb2gray(img), 'canny');
figure, imshow(edge_canny), title('Canny edge detection')

% We can improve the result (discard noise) by increasing the sigma for
% Gaussian filtering.
edge_canny2 = edge(rgb2gray(img),'canny', [] , 4);
figure, imshow(edge_canny2), title('Canny edge detection')
% We can display the edges overlaid on the image.
figure, imshowpair(img, edge_canny2, 'blend')


%% Task
% Adapt the median filtering and the wiener filtering to work for color
% images as input. Name the functions medfilt_color and wienerfilt_color.
% Do this by applying medfilt2 and wiener2 for each color channel and then 
% concatenate back the modified color channels to the output images. 