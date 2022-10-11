%% Week 3: Basic image filtering in the spatial domain.
% Author: Irina Ciortan
%% Convolution and correlation
% At the core of signal filtering, there are two main operations:
% Convolution and Correlation.
% Application-wise, correlation is used to find similarity between two
% signals, while convolution is used for filtering and seeing the impact of
% one signal on another signal. 
%   * Matlab function to perform 1D correlation: FILTER
%   * Matlab function to perfom 1D convolution: CONV
%   * Matlab function to perform 2D correlation (works for 1D as well): FILTER2
%   * Matlab function to perfom 2D convolution (works for 1D as well): CONV2
%   * With IMFILTER function you can choose either 2D correlation (default) and 2D convolution, and it also has some boundary options.
% Search these functions in the Matlab documentation using "help" function.

%% Correlation and convolution for 1-D signals
%Let's consider the following 10-element vector:
inputv = [5, 4, 2, 3, 7, 4,  6, 5, 3, 6]
%Let's assume that based on this input vector, we want to produce a new
%vector, where every element is replaced by the average between the element
%itself, its left neighbour and its right neighbour.
%What about the first and last elements that don't have all neighbours?
%Let's first solve this ourselves, excluding the first and last elements:
c=1;
for i = 2:9
    outputv_1(c) = (inputv(i-1)+inputv(i) + inputv(i+1))/3; 
    c = c+1;
end
%Notice the size of the input and output vectors:
size(inputv)
size(outputv_1)

% Let's now solve the problem using a window/filter:
f = [1/3, 1/3, 1/3]
%And apply in-built correlation function:
outputv_2 = filter2(f, inputv)
%Notice the size of the input and output vectors:
size(inputv)
size(outputv_2)

%What about the first and last elements who don't have all neighbours?
% Boundary problem -- solution padding (filter2 function includes this).
% Default padding: add one 0 to both ends.
outputv_3 = filter2(f, inputv, 'full') %adds two zeros to both ends
outputv_4 = filter2(f, inputv, 'valid') %returns the same as outputv_1

%What if we use conv2 with the same filter?
outputv_conv = conv2(inputv, f) %returns the same as *filter2* with option 'full'
outputv_conv2 =conv2(inputv, f, 'same') %returns the same as *filter2* with default option

%Other padding options?
outputv_imfilt = imfilter(inputv, f, 'replicate') %replicates first element and last element

%% Asymmetric filter

fa = [-1 0 1]

assym_conv = conv2(inputv, fa, 'same')
assym_corr = filter2(fa, inputv, 'same')

fa = [3 7 4]

assym_conv = conv2(inputv, fa, 'same')
assym_corr = filter2(fa, inputv, 'same')
% Convolution is a flipped correlation.
% When filters are symmetric, there's no difference.
%% Correlation and convolution for 2-D signals
%Read cameraman image.
img1 = imread('cameraman.tif');

% Filter IMG3 using a 5x5 mean filter.
h1 = ones(5,5) / 25;
img2 = imfilter(img1,h1); % symmetric filter => correlation = convolution
figure, imshow(img1), title('Original image');
figure, imshow(img2), title('Filtered image')


%% Apply correlation for template-matching
%Read an image with Waldo skiing.
scene = imread('waldoskiing.jpg');
%Crop waldo from the scence image.
waldo = imcrop(scene, [834.5 261.5 120 124]);
%Display the template and the scene
montage({waldo,scene})
%%Perform normalized cross-correlation and display the result as surface.
c = normxcorr2(rgb2gray(waldo),rgb2gray(scene));
figure, surf(c)
shading flat

%Find the peak in cross-correlation.
[ypeak,xpeak] = find(c==max(c(:)));

%
yoffSet = ypeak-size(waldo,1);
xoffSet = xpeak-size(waldo,2);


%Highlight waldo's location
figure, imshow(scene)
drawrectangle(gca,'Position',[xoffSet,yoffSet,size(waldo,2),size(waldo,1)], ...
    'FaceAlpha',0, 'Color','green');
%% EXERCISE 1 
%%%%%%%%%%%%%%%%

% Below, we load the image 'football.jpg', convert it to a greyscale
% image and apply a 5x5 mean filter, by using the commands:
img1 = imread('football.jpg');
img3 = rgb2gray(img1);

h1 = ones(5,5) / 25;
img4 = imfilter(img3,h1);
figure,imshow(img3), title('Original image');
figure, imshow(img4), title('Filtered image')
% The filtered image have a two pixel wide blackish frame. This effect is
% more visible is you visualize the images using imtool and zooming in.
% a) Use indexing techniques to remove these.
img5 = img4(3:end-2,3:end-2);
imtool(img5)


% b) Use FILTER2 and CONV2 with the option 'valid' to exclude these.
%    Hint 1: Type convert IMG3 to DOUBLE.
%    Hint 2: The image should be the second argument to FILTER2,
%            not the first as it is to CONV2 and IMFILTER.

img6 = filter2(h1, double(img3), 'valid');
img7 = conv2(double(img3), h1, 'valid');
figure, imshowpair(img6, img7, 'montage');
% c) Use the boundary option of IMFILTER to get a same-sized image without
%    the black frame.
img8 = imfilter(img3, h1, 'replicate');
figure, imshow(img8);
img9 = imfilter(img3, h1, 'circular');
figure, imshow(img9);
%%%%%%%%%%%%%%%%

%% EXERCISE 2 
%%%%%%%%%%%%%%%%

% Similar to how we implemented correlation for 1D signal, create a
% function that computes the correlation for 2D signals. The input of the function 
% are a grayscale image and the filter/sliding window; the output is the filtered image.
% The function should work for any filter size and for any filter content. However, in the
% beginning, you can try it out with a specific numeric example.
% a) At first you can try without padding the input image.
% b) Then, add symmetric (in both dimensions) padding with 0s.
% c) Test the function on grayscale version of "peppers.png" found in Matlab library.
% d) Test for various sizes of a mean filter: 3x3, 5x5, 7x7, and display
% the images; display as well the difference between them.
%
%% EXERCISE 3 
%%%%%%%%%%%%%%%%
% Extend function from exercise 3 to work for color images.


