%The purpose of this exercise is to try out simple arithmetic operators to
%segment objects in an image.
%For example, let's say we want to separate only the coins in the following
%image.
img2 = imread('coins.png');
figure, imshow(img2)
%Then we empirically choose a grayscale value that separates the coins from
%the background.
t = img2>100; % t will be a matrix with the same size of img2
%However it will be a matrix of "logical" type values, containing true/1 at those
%positions where img2 has values above 100 and false/0 otherwise.
%If we display t, it will be "black and white"/binary image.
figure, imshow(t), title('Img > 100');
%The following threshold will segment the background.
t1 = img2<=100;
figure, imshow(t1), title('Img <= 100');
%You cna also visualize the original iamge and thresholded images side by
%side
figure, imshowpair(img2, t, 'montage'), title('Side by side');

%Feel free to try out values other than 100.

%The function finding automatically the threshold in an image based on Otsu algorithm is
%graythresh
otsu_thresh = graythresh(img2);
%get binary image with function im2bw
imbin = im2bw(img2, otsu_thresh);
figure, imshow(imbin), title('Otsu threshold')
figure, imshow(t-imbin), title('Difference Otsu threshold and my threshold');
%Actually our randomly manually chosen threshold segments better one of the coins.

%The function "imbinarize" transforms a grayscale image into a
%binary one, using Otsu method. In addition, it has the "adaptive" option
%that computes the adaptive Otsu threshold).

imbin_adaptive = imbinarize(img2, 'adaptive');
figure, imshow(imbin_adaptive), title('Adaptive Otsu threshold')

%The results are still not perfect. However, in a later alb, we will see
%that we can improve the segmentation results using morphological
%operations.
