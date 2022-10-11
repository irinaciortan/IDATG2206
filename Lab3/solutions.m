%% Exercise 1
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

%% Calling the functions from Exercise 2 and 3
close all
img10 = mycorr_nopadding(img3,h1);
img11 = mycorr(img3, h1);
img12 = mycorr_color(img1, h1);
figure, subplot(131), imshow(img10), title('Mycorr, no padding');
subplot(132), imshow(img11), title('Mycorr, padding');
subplot(133), imshow(img12), title('Mycorr, padding + color');
