%% Week 6: Color Image Segmentation. Intro to feature extraction. 


%% Clustering (k-means)
% K-means is both a method of segmentation as well as an example of
% unsupervised classification, where the classification feature is  based on usually the
% color distribution.
img = imread('Christian_Krohg-Portrait_of_the_Painter_Oda_Krohg.jpg');
figure, imshow(img), title('Oda Krohg painted by Christian Krohg');

% Convert the image to L*a*b* color space using rgb2lab.
img_lab = rgb2lab(img);
ab = img_lab(:,:,2:3);
ab = im2single(ab); % we change the data type from float double precision to float single precision
% imsegkmeans function doesn't accept *double* as input
nColors = 4;
% repeat the clustering 3 times to avoid local minima
pixel_labels = imsegkmeans(ab,nColors,'NumAttempts',3);
% Display the label image. It is a grayscale image, where each cluster has a
% different label
figure, imshow(pixel_labels,[]), title('Image Labeled by Cluster Index')
% Use imtool to find out which number corresponds to each cluster/color.
imtool(pixel_labels,[])

% What happens if we had tried kmeans on the RGB image
pixel_labels_rgb = imsegkmeans(img,nColors,'NumAttempts',3);
imtool(pixel_labels_rgb, []) 
% As you can see from the label image, the result is more noisy if we use
% RGB

% Let's segment out the red parts in the image. From imtool window, we find out
% that the red has label number 2.
mask_red = pixel_labels==2; % we use  a comparison expression to get a binary image that keeps only the cluster2
cluster_red = img .* uint8(mask_red); % then we multiply  the binary image with the original image,
% masking out the red pixels 
% This way we get the color overlaid on the label
figure, imshow(cluster_red), title('Red objects using a, b channels');
% We can see that the right side of the shirt was not attached to the same
% clusters.

% How could we improve this?
% Perhaps trying more attempts and different nr of iterations?
pixel_labels_plus = imsegkmeans(ab,nColors,'MaxIterations', 150, 'NumAttempts',5);
imtool(pixel_labels_plus, []) % the result doesn't change much which means that the Matlab implementation is reproducible


% What if we use the L channel as well?
lab = im2single(img_lab);
pixel_labels_lab= imsegkmeans(lab,nColors);
imtool(pixel_labels_lab, []) % the result is better now, we get the full shirt
% We segment again the red part, label is 2 for red also with this
% clustering.
mask_red = pixel_labels_lab==2; 
cluster_red = img .* uint8(mask_red);
figure, imshow(cluster_red), title('Red objects using L, a, b channels');

% What if we take pixel coordinates into account as well?
nrows = size(img,1);
ncols = size(img,2);
[X,Y] = meshgrid(1:ncols,1:nrows);
featureSet = cat(3,lab,X,Y); % we create a set of features by concatenating the lab info and spatial info into one image
pixel_labels_spatial = imsegkmeans(featureSet,nColors);
imtool(pixel_labels_spatial, []) % the result is actually not so good
% Other features could be added to further improve the results, for example
% texture features.

%% Color Palette Extraction from KMeans
% Quantize/summarize the RGB image into 10 colors. 
% The first output of the imsegkmeans functions returns the label of each cluster
% The second output returns the centroid values (the mean color in each
% cluster).
[L,C] = imsegkmeans(img,10);
J = label2rgb(L,im2double(C));
figure, imshow(J), title('Color Quantized Image')

% Call the colormap and colormapeditor to visualize the dominant colors as a
% a palette.
cm = colormap(C);
colormapeditor
% If you want to play more with the color space and color palette
% visualization, this is a nice tool developed by the community, that you
% can download and use: https://se.mathworks.com/matlabcentral/fileexchange/69538-image2palette-simple-k-means-color-clustering

%% Superpixels (SLIC)
% Use the superpixels functions. The second argument is the desired number
% of superpixels you want to cluster the image into.
% The function returns the label matrix and the actual number of labels
% computed.
[lbl,nrl] = superpixels(img,50);
% We see that the actual number of labels (nrl) returned is 48. 
BW = boundarymask(lbl); % computes a mask (binary image) that represents the boundaries for the input label matrix L
% Display original image with superimposed superpixel boundaries
figure, imshow(imoverlay(img,BW,'cyan'),'InitialMagnification',67)

% Compute the average colors inside the superpixels
% It calls the function getAverageColors that you have received as an
% auxiliary file (it is not an in-built function).
avgim = getAverageColors(img, lbl, nrl);
figure, imshow(avgim,'InitialMagnification',67), title('Average Colors')

% An important parameter of the *superpixels* function is the
% 'Compactness'. The default value is 10. If we lower this value, we might
% get less square superpixels, but more loyal to the contour shapes.
%% Hough Transform for Lines
% Hough Transform, given a set of edge points, finds the lines that best 
% explain the data. The *hough* function in Matlab computes the Hough transform 
% in search for all lines in the image. The input must be a binary edge image. 
close all

pisa = rgb2gray(imread('marco-ceschi-unsplash.jpg'));
% Get the binary edge image with Canny operator
e = edge(pisa, 'canny', [], 5);
figure, imshow(e), title('Edge Image');

% Apply Hough transform
[H,T,R] = hough(BW); 
% The Hough transform returns the hough parametric space H, the distance 
% vector R from the line to the origin of the coordinate system and the
% angle  theta T between this vector and the x-axis.
figure, imshow(H,[],'XData',T,'YData',R,'InitialMagnification','fit');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;

% Find peaks in the Hough transform of the image 
P  = houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));
x = T(P(:,2)); y = R(P(:,1));
plot(x,y,'s','color','white');

%Find lines
lines = houghlines(BW,T,R,P,'FillGap',10,'MinLength',30);
%And plot them
figure, imshow(pisa), hold on
% Have a look at the function for plotting the lines (auxiliary file) - it basically passes
% through all the lines and draws them given the x-y coordinates. It is not
% an in-built function.
plotHoughLines(lines, 'green')

% Highlight the line overlapping the architectural structure from the tower
plotHoughLines(lines(4), 'cyan')

% Find the angle to x-axis, in degrees
lines(4).theta
% It seems that the angle of slant is 12 degrees.
% However, this is not an accurate measurement since our photo of the Pisa
% Tower is not complete (from the ground) and we are not sure of the exact
% perspective.
%% Hough Transform for Circles
% Hough Transform can be applied to all parametric shapes, not only lines:
% circles, ellipses, parabolas. The *imfindcircles* in Matlab wraps Hough
% Transform. Even though Hough Transform can find all circles in the image,
% of unknown radius, in order to restrict the detected circles, the
% function asks for an interval of possible radiuses. 
% The function *imfindcircles* receives as input a grayimage (it incorporates
% the edge detection).
close all
img = imread('ball.png');
figure, imshow(img)
[centers, radii, metric] = imfindcircles(rgb2gray(img),[140 150], 'ObjectPolarity','bright', 'Sensitivity', 1);
% Select the strongest signal
centersStrong5 = centers(1,:); 
radiiStrong5 = radii(1);
metricStrong5 = metric(1);
% And display it
viscircles(centersStrong5, radiiStrong5,'EdgeColor','green');

%% Image Segmenter App Demo
% Image Segmenter is an app and provides a Graphical User Interface,
% including most of the segmentation algorithms. It is useful for trying out
% graph cut and active contour segmentations, since you can interactively
% select the scribbles ROI that corresponds to the foreground. If you don't
% have the Image Segmenter app, you can install it from Apps->Image
% Segmenter App. Then, you can run it from the menu or from the command
% line as below:

lys = imread('Oda_krohg_japansk_lykt_1886.jpg');
imageSegmenter(lys)

% Once you open the app, separate the lady in white from the background, with both the
% active contours approach and the graph cut approach. Save masks of the
% foreground resulted with either of the two approaches, since you'll need it for one of the exercise.
%% Exercise 1: Hough Circles
% Try to find the outer and inner circles for the image with donuts using Hough
% transform for circles. You may be tolerant with the precise contours segmented.
% Draw the boundaries of the outer circles and the
% boundaries of the inner circles in different colors.
% You could at first the Image Segmenter app to achieve this task and then
% maybe tune in the parameters without the app.
don = imread('donuts.jpg');
don_gray = rgb2gray(don);
imageSegmenter(don_gray)

% After achieving this with Hough Transform you can try Active Contours and
% Graph Cut as well.
%% Exercise 2: Photo Montage
% With the segmentation exported from the Image Segmenter App for Oda
% Krohg's painting, paste the lady-in-white on top of Trolltunga, as if she
% was looking towards the fjord. 
scene = imread('trolltunga.jpg');
figure, imshow(scene), title ('Lady-in-white at Trolltunga')

