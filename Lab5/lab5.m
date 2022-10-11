%% Week 5: Binary image processing. Morphological operations. Gray Scale Image Segmentation
% A binary image is an image that has only 2 values: 0s and 1s. It is
% considered of logical type, where 0s are considered "false" and 1s "true"
% values. By convention, the 1s represent the foreground and the 0s the
% background. 
% Binary images can also be of type uint8 (where 0s stay zero and 1s become 255). 
% Binary images can be obtained as a result of image segmentation
% operation (thresholding, edge detection, etc) and other feature detection
% operations. Sometimes we might need to label, count and measure shape properties of the white
% regions in the binary images. In addition, we might want to refine the
% binary image, by removing small artifacts or by fillying holes. This
% latter task can be solved with morphological operations. 



%% Logical operations on binary images
% Let's build small binary images with the functions false and true.
a = false(100,100);
b = false(100,100);
a(20:70, 30:60)= true;
b(40:100, 60:100) = true;

% NOT a; Get the negative image: white becomes black and vice versa.
% ~ operator and imcomplement, both output the complement image.
a_neg = ~a; 
a_neg2 = imcomplement(a);  % imcomplement works on grayscale image as well (it means mainly 255-input image)

% a AND b; returns only overlapping white areas in both a and b.
intersectioni = a&b;

%a OR b; returns regions that are white in a, as well as regions that are
%white in b
unioni = a|b;

%a MINUS b; returns areas that are white only in a and not in b.
minusi = a-b;

figure, subplot(161), imshow(a), title('a')
subplot(162), imshow(b), title('b')
subplot(163), imshow(a_neg), title('NOT a')
subplot(164), imshow(intersectioni), title('a AND b');
subplot(165), imshow(unioni), title('a OR b');
subplot(166), imshow(minusi), title('a - b');

%% Extract connected components and label binary images
% Two pixels are connected in a binary image if there is path consisting of
% white pixels between them. A connected component is made of pairs of
% connected pixels considering a 4-neighborhood or 8-neighborhood
% adjacency. A connected component is also called "blob". Labeling is the process of assigning the same label number to
% every pixel in a connected component.  

% First, let's read agrayscal image with coffee beans
img = rgb2gray(imread('nordwood-themes-unsplash.jpg')); 

% Thresold the grayscale image to get a binary image. Use Otsu's algorithm
% for computing the (global) threshold. imbinarize does that for you.
bw = imbinarize(img);
bwn =~bw; %we complement the image to color white the regions for the coffee beans.
figure, imshowpair(img, bwn, 'montage');

% Find the 8-connected components using bwconncomp.
CC = bwconncomp(bwn);

% Then label each connected component using labelmatrix function.
% labelmatrix function outputs an image, where for each different conenct
% component, a different number is given.
L = labelmatrix(CC);
% Since we have few labels, we need to scale the range of the values displayed.
% So add [] as argument to imshow. 
figure, imshow(L,[]), title('Grayscale labels')

% Another way to get the labels of connected components is when applying
% watershed segmentation, as we will see in a later example.

% For better visualization, we can pseudocolor the label image using
% label2rgb function, that converts each labeled connected component to a
% random color.
% In the example below, we use the 'spring' color palette to assign colors to labels random order and make background
% black.
RGB = label2rgb(L, 'spring','k','shuffle');
figure, imshow(RGB), title('Color labels');


%% Region properties
% Once connected components are found we sometimes want to quantify them  with basic statistics.
% For example, automatically count them, or measure their area, or know
% their position and orientation.
% There is a function in matlab that gives us such statistics and it is
% called *regionprops*. The input to regionprops is the connected components structure
% or the labeled image and the output is a *structure*. In Matlab, a
% structure is a data type that is an array with named "fields". Access to
% each of this field is given by "." indexing: structurename.fieldname

% The default properties returned by regionprops are Area, Centroid (x, y location)  and Bounding
% Box (the rectangle that completely encloses the blob). But there are more
% other properties that can be displayed.
close all %closes all the windows except the main window and imtool; just to clean up the environment a bit.
blobs = regionprops(CC)

% Q: How many coffee beans do we have in the image?
% A: As many as the size of the structure returned by the regionprops
nrbeans = size(blobs,1)
%So there are 19 cofee beans in the image. 
blobs(1)
blobs(1).Area

% You can access all properties by using 'all' as argument or specific
% properties by indicating their name.
stats = regionprops(CC, 'Circularity')
%Find the indices of the blobs that have a Circularity index higher than
%0.97
idx = find([stats.Circularity] > 0.97);
% Then use the ismember function to create a new black and white image with
% only the circular coffee beans from the label image of the original image.
circularbeans = ismember(L,idx);  
figure, imshow(circularbeans)

%We can also use the function visboundaries to draw the boundaries for each
%of the object on the original image, 
figure, imshow(img), title('Circular coffee beans')
hold on
%but only for the circular beans objects
visboundaries(circularbeans, 'Color','r')
%% Morphological operations: Erosion, Dilation, Opening, Closing
% Morphological operations are useful for shrinking (erosion) or
% enlarging (dilation) components in  black and white images. Erosion and dilation
% are the primitive morphological operations and all the other morphological operations
% are a sequence of these two operations.

% Structuring elements 
clear global %clean all the variables in the workspace
close all %close all the windows except the main window
bw = imread('text.png');            		%Read in binary image
se=[0 1 0; 1 1 1; 0 1 0];                   %Define structuring element as an array
bw_out=imdilate(bw,se);              		%Dilate image
figure,subplot(1,2,1), imshow(bw), title('Original');         		%Display original
%Display dilated image performed with a structuring element which is similar to a filter mask. 
subplot(1,2,2), imshow(bw_out), title('Dilated');     		

se=ones(6,1);                       		%Define structuring element
bw_out=imerode(bw,se);              		%Erode image
figure, subplot(1,2,1), imshow(bw), title('Original');         		%Display original
subplot(1,2,2), imshow(bw_out), title('Eroded');     		%Display eroded image

% You can use *strel* function to generate more customized structuring
% elements.

se1 = strel('square',4)                     %4-by-4 square
se2 = strel('line',5,45)     				%line, length 5, angle 45 degrees
bw_1=imdilate(bw,se1);              		%Dilate image
bw_2=imerode(bw,se2);              			%Erode image
figure, subplot(1,2,1), imshow(bw_1), title('Dilated w square');         		%Display dilated image
subplot(1,2,2), imshow(bw_2), title('Erode w oblique line');               %Display eroded image


% Opening is the name given to the morphological operation of erosion followed 
% by dilation with the same structuring element. Closing is the name given to 
% the morphological operation of dilation followed by erosion with the same 
% structuring element.

A=imread('openclose_shapes.png'); A=~logical(A);            %Read in image and convert to binary
figure, imshow(A)
figure
se=strel('disk',3); bw1=imopen(A,se); bw2=imclose(A,se);	%Define SEs then open and close
subplot(3,2,1), imshow(bw1); subplot(3,2,2), imshow(bw2);	%Display results
se=strel('disk',6); bw1=imopen(A,se); bw2=imclose(A,se);	%Define SEs then open and close
subplot(3,2,3), imshow(bw1); subplot(3,2,4), imshow(bw2);	%Display results
se=strel('disk',15); bw1=imopen(A,se); bw2=imclose(A,se);	%Define SEs then open and close
subplot(3,2,5), imshow(bw1); subplot(3,2,6), imshow(bw2);	%Display results


clearvars -except bw1
%% Hit-or-miss transform
% Is a basic tool for shape detection
% The hit-or-miss transform indicates the positions where a certain pattern 
%(characterized by a structuring element B) occurs in the input image.
A = imread('text.png');                        				%Read in text 
B=imcrop(A,[153.5 5.5 14 20] );                              				%Crop letter "h"

se1=B; se2=~B;                              			%Define hit and miss structure elements
bw=bwhitmiss(A,se1,se2);                    			%Perform hit-miss transformation
[i,j]=find(bw==1);                         		 		%Get explicit coordinates of locations
figure,
subplot(1,3,1), imshow(A), title('Original');                  			%Display image
subplot(1,3,2), imagesc(B); axis image; axis off, title('Target shape');   	%Display target shape
subplot(1,3,3), imshow(A); hold on; plot(j,i,'r*'), title('Hit-or-miss'); 	%Superimpose locations on image	

%% Region filling
% When thresholding or segmentation procedures are not perfect, then we
% might have some holes (black pixels) in the foreground pixels. Therefore
% we should "fill" these holes.  The filling is done by iterative dilation
% operations. After each dilation we take the intersection of the dilated
% image and the complement of the original image to make sure we don't
% fill the whole image. 
coins = imread('coins.png');
coins_bw = imbinarize(coins);
figure, imshowpair(coins, coins_bw, 'montage'), title('Imperfect segmentation with holes');
coins_filled = imfill(coins_bw, 'holes');
figure, imshow(coins_filled), title('Filled holes');

%% Image segmentation: Region growing

% Region growing method start with a seed pixel and builds a connected
% component in an 8=pixel neighborhood, as long as the neighboring pixels
% are similar in intensity. 
% grayconnected function applies region growing in Matlab and return a black
% and white image corresponding to the segmented region.
figure, imshow(coins)
impixelinfo
seedrow = 105;
seedcolumn = 170;
rg1 = grayconnected(coins, seedrow, seedcolumn);
% We can display the label of the segmented coin superimposed on the image.
figure, imshow(labeloverlay(coins,rg1))

% We can change the "tolerance" argument to the grayconnected function - the maximum
% difference in intensity the 8connected pixels can have with respect to the seed pixel
% in order to be added to the connected component. By default, tolerance is 32 for
% integer images.



%% Exercise 1
% Change the example given for hit-or-miss transformation to find the
% letter "m" in the "text" image. You should find the vertical standing "m"
% as well as the rotated "m". 
%% Exercise 2: Watershed segmentation
% How many coffee beans are there in an image with spilled coffee beans, close to each other?

% Read the image
spilled_beans = rgb2gray(imread('fabienne-hubener-unsplash.jpg'));
w = watershed(spilled_beans)
figure, imshow(label2rgb(w))

% By applying watershed transformation directly on the image, we confront
% with oversegmentation. However, this can be improved with the aid of
% morphological operations. Follow the steps in Matlab example "Marker-Controlled
% Watershed Segmentation" to extract the blobs of coffee beans. Once you
% get the blobs image, count the number of beans. 
% Crop the input image if needed.
% Through this exercise, you'll learn another morphological operation:
% opening by reconstruction.