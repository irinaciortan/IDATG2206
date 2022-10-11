%% Week 7: Feature Detection and Extraction
% Author: Irina Ciortan
%% Intro
% For some of the functions presented today you need to have Computer Vision
% toolbox installed.
% Features are a vectorial description of pattern/object/subarea in the
% image. These features can describe color, shape, texture, semantics in an
% image. Based on a set of features, useful tasks in computer vision can be
% solved: template matching, image registration and stitching,
% classification, recognition, etc. 
% The functions for feature detection in Computer Vision toolbox return
% *points* objects that store information regarding a feature, including
% the spatial coordinates of these points in an image. 
% A good feature is the one that is invariant to changes in illumination,
% background, scale and rotation. 

%% Corner Feature Detection
% A corner can be defined as a crossing between a horizontal and a vertical
% edge. Briefly, the steps of the algorithm are:
%%
% # Compute image gradients in x and y direction over small areas in the
% image.
% # Compute the covariance matrix/ also called structure tensor or Hessian
% matrix.
% # Use a corner response formula:
%%
% * Smallest eigenvalue of the covariance matrix.
% * Approximations  of smallest eigenvalue based on the determinant and trace of the
%       covariance matrix.
%%
% A checkerboard image is a classical picture for demonstrating the accuracy
% of a corner detection algorithm.
checkerboard1 = rgb2gray(imread('images/img.jpg'));

% Use *detectMinEigenFeatures* function that computes a corner response by
% finding the smallest eigenvalue of the covariance matrix.
% We have encircled the function with *tic-toc* commands in order to see
% the time of execution.

tic
corners_eigen = detectMinEigenFeatures(checkerboard1);
toc 

% As you can see by double clicking *corners_eigen* in the workspace
% window, it is *cornerPoints* object, that has three attributes (Location,
% Metric, Count).


% Display the strongest corner points superimposed on the image.
% As it was the case with Hough feature detection, we might get many noisy
% points. This is why we should select only the strongest response.
% The corner *points* object returned by the functions implemented in
% Computer Vision toolbox has a method *selectStrongest* for this.

figure, imshow(checkerboard1); hold on;
plot(corners_eigen.selectStrongest(50)), title('Smallest eigenvalue response');

% The function detectHarrisFeatures uses the approximation of the
% smallest eigenvalue.

tic
corners_harris = detectHarrisFeatures(checkerboard1);
toc
figure, imshow(checkerboard1); hold on;
plot(corners_harris.selectStrongest(50)), title('Harris-Stephens response');

% For the checkerboard image (a simple image), visually, the result is the same  with
% both algorithms.  However, if you check the elapsed time in the command
% window, the first function took cca. 0.027 seconds to execute, while the
% second function took cca. 0.023 seconds. So the implementation using
% eigenvalues is more computationally intensive, even though it might be
% more accurate.

% Let's try with more complex images and check whether corners are a
% feature robust to translation and extreme shifts in illumination.

city1 = imread('images/img2.jpg');
city2 = imread('images/img3.jpg');

corners_city1 = detectMinEigenFeatures(rgb2gray(city1));
corners_city2 = detectMinEigenFeatures(rgb2gray(city2));
figure
subplot(121), imshow(city1), hold on, plot(corners_city1.selectStrongest(70)), title(' 70 Corner features');
subplot(122), imshow(city2),hold on, plot(corners_city2.selectStrongest(70)), title('70 Corner features');

% We can see that the corners for the top of the Empire State building are
% not detected in the "bluer" image. 
%% SIFT Feature detection
% Scale Invariant Feature Transform is a patented algorithm proposed in 2004 for overcoming the scale invariance 
% of Harris corner detection. Harris corner detection is a method that is 
% rotation-invariant and limitedly invariant to photometric changes. However, 
% when scale is increased, the corners spread out and get a flattened effect. 
% SIFT solves this by searching for local minimas and maximas in a
% scale-space created with Gaussian filters.
% Over the years, alternatives built on SIFT have been proposed:
%%
% * Speeded-up robust features (SURF) algorithm
% * Binary Robust Invariant Scalable Keypoints
% * Oriented FAST and Rotated BRIEF (ORB) method

% Let's detect SURF features for the city images.
surf_city1 = detectSURFFeatures(rgb2gray(city1));
surf_city2 = detectSURFFeatures(rgb2gray(city2));
figure, 
subplot(121), imshow(city1), hold on, plot(surf_city1.selectStrongest(100)), title('100 SURF features');
subplot(122), imshow(city2),hold on, plot(surf_city2.selectStrongest(100)), title('100 SURF features');

% Let's detect BRISK features for the city images.
brisk_city1 = detectBRISKFeatures(rgb2gray(city1));
brisk_city2 = detectBRISKFeatures(rgb2gray(city2));
figure, 
subplot(121), imshow(city1), hold on, plot(brisk_city1.selectStrongest(100)), title('100 BRISK features');
subplot(122), imshow(city2),hold on, plot(brisk_city2.selectStrongest(100)), title('100 BRISK features');

% Let's detect ORB features for the city images.
orb_city1 = detectORBFeatures(rgb2gray(city1));
orb_city2 = detectORBFeatures(rgb2gray(city2));
figure, 
subplot(121), imshow(city1), hold on, plot(orb_city1.selectStrongest(100)), title('100 ORB features');
subplot(122), imshow(city2), hold on, plot(orb_city2.selectStrongest(100)), title('100 ORB features');

%% Feature Extraction
% Corner detection and SIFT-based corner detections give us *interest
% points*.  However, we need to quantify these keypoints and the areas 
% surrounding them into a meaningful descriptor - basically, numerical distribution (such as pixel 
% intensity values, gradient magnitude, orientation, etc). 
% You can think of a descriptor as fingerprint for each keypoint.
% This is called feature extraction. 


% The feature extraction method depends on the type of *points* object passed
% to the extractFeatures function. SURF has a differnt way of descriptor
% computation than ORB and so on. 
% The function automatically detect the type of *points*, but this can be
% as well explicitly defined. 
[feats_orb_city1, val_points_orb1] = extractFeatures(rgb2gray(city1),orb_city1);
feats_orb_city1.Features;
% We can see that the result is an array, that has for each ORB keypoint a feature a vector.
%% Feature Extraction with Histogram of Oriented Gradients
% The histogram of oriented gradients (HOG) is a feature descriptor that was
% initially proposed for the application of pedestrian detection. HOG
% counts cooccurences of gradient orientations in blocks and cells within an
% image. 


% HOG can be computed directly on the intensity/color values of an image.
% The function to extract HOG features is *extractHOGfeatures*
% It returns the feature vector. It can also return a visualization
% variable, that we can then use for plotting the
[hog_city1, visualization] = extractHOGFeatures(city1);

% Display the image the HOG feature visualized.
figure
subplot(1,2,1); imshow(city1)
subplot(1,2,2); plot(visualization); title('Default Cell Size 8');
% We can notice the visualization is full of small scale details.
% In order to get more large scale size, we can increase the cell size that
% the image is divided in.
[hog_city1_32, visualization32] = extractHOGFeatures(city1,  'CellSize', [32 32]);
figure
subplot(1,2,1); imshow(city1)
subplot(1,2,2); plot(visualization32); title('Cell Size 32');
% Now, we get a feel of the skyline of the city by following the
% orientations given by HOG.

% We can also use HOG around SIFT-based keypoints and corner points. 
% Compute the HOG features around the 10 strongest ORB keypoints.
% In this case, The function also returns an additional output *vali points*, which is the 
% input point locations whose surrounding region is fully contained within
% the image.
[hog_feats, val_points_hog, viz] = extractHOGFeatures(city1,orb_city1.selectStrongest(10));

%Display the original image with an overlay of HOG features around the strongest corners.
figure; imshow(city1);
hold on;
plot(viz,'Color','green'); title('');
%% Feature Matching 
% Once we have the feature array obtained with feature extraction, we can
% perform feature matching, classification, etc.

% Let's find correspondence between city1 with city2 using ORB feature extraction
% around ORB keypoints.
clear feats_orb_city1; clear feats_orb_city1; 
clear val_points_orb1; clear val_points_orb2;

[feats_orb_city1, val_points_orb1] = extractFeatures(rgb2gray(city1),orb_city1);
[feats_orb_city2, val_points_orb2] = extractFeatures(rgb2gray(city2),orb_city2);

% Use the *matchFeatures* function to find correspondences
indexPairs = matchFeatures(feats_orb_city1, feats_orb_city2);

% Locate of the corresponding points for each image.
matchedPoints1 = val_points_orb1(indexPairs(:,1),:);
matchedPoints2 = val_points_orb2(indexPairs(:,2),:);
%There are 0 matched points.
figure; 
showMatchedFeatures(city1,city2,matchedPoints1,matchedPoints2), title('ORB features');

% Let's find correspondence between city1 with city2 using HOG around ORB
% keypoints.
clear feats_orb_city1; clear feats_orb_city1; 
clear val_points_orb1; clear val_points_orb2;
[feats_orb_city1, val_points_orb1] = extractHOGFeatures(city1,orb_city1);
[feats_orb_city2, val_points_orb2] = extractHOGFeatures(city2,orb_city2);

% Use the *matchFeatures* function to find correspondences
indexPairs = matchFeatures(feats_orb_city1, feats_orb_city2);

% Locate of the corresponding points for each image.
matchedPoints1 = val_points_orb1(indexPairs(:,1),:);
matchedPoints2 = val_points_orb2(indexPairs(:,2),:);
% Now we have found some correspondences
figure; 
showMatchedFeatures(city1,city2,matchedPoints1,matchedPoints2);
title('HOG features around ORB keypoints');
% But they are not all quite correct.

%% Find an object in a cluttered scene with Harris corner points.

scene1 = imread('images/chandelier.jpg'); % the object to search for
scene2 = imread('images/scene4.jpg'); % the cluttered scene

corners1 = detectHarrisFeatures(rgb2gray(scene1));
corners2 = detectHarrisFeatures(rgb2gray(scene2));

[feats1, val_points1] = extractFeatures(rgb2gray(scene1),corners1);
[feats2, val_points2] = extractFeatures(rgb2gray(scene2),corners2);

% Use the *matchFeatures* function to find correspondences
indexPairs = matchFeatures(feats1, feats2);

% Locate of the corresponding points for each image.
matchedPoints1 = val_points1(indexPairs(:,1),:);
matchedPoints2 = val_points2(indexPairs(:,2),:);
% There is 1 mathcing points ant not correct actually.
figure; 
showMatchedFeatures(scene1,scene2,matchedPoints1,matchedPoints2, 'montage'), title('Corner features');
% Corner features are not good enough given the change in scale of the
% objects, the illumination and the background.

%% Find the object in a cluttered scene using SURF features
pts1 = detectSURFFeatures(rgb2gray(scene1));
pts2 = detectSURFFeatures(rgb2gray(scene2));

[feats1, val_points1] = extractHOGFeatures(rgb2gray(scene1), pts1);
[feats2, val_points2] = extractHOGFeatures(rgb2gray(scene2), pts2);

% Use the *matchFeatures* function to find correspondences
indexPairs = matchFeatures(feats1, feats2);

% Locate of the corresponding points for each image.
matchedPoints1 = val_points1(indexPairs(:,1),:);
matchedPoints2 = val_points2(indexPairs(:,2),:);
% There are many correspondeces between the object and the scene with SURF
figure; 
showMatchedFeatures(scene1,scene2,matchedPoints1,matchedPoints2, 'montage'), title('SURF features');

%% Locate the object in the scene based on the matches of SURF features
% Function *estimateGeometricTransform2D* estimates the geometric transformation 
% of the object between the one image and the other based on the matched points.
% This transformation allows us to better localize the object in the scene, by eliminating outliers.

[tform, inlierIdx] = estimateGeometricTransform2D(matchedPoints1, matchedPoints2, 'affine');
inlier_pts1   = matchedPoints1(inlierIdx, :);
inlier_pts2 = matchedPoints2(inlierIdx, :);

figure;
showMatchedFeatures(scene1, scene2, inlier_pts1, inlier_pts2, 'montage');
title('Geometrically Matched Points (Inliers Only)');


% As  conclusion, all the keypoint detectors (SURF, ORB, BRISK) work better 
% for finding correspondence for more controlled geometric shifts and enough overlaps
% between two scenes.
%% Exercise 1
% Try to find the correspondences between two images of the city/the two
% domestic scenes using  corner points, SURF, BRISK and ORB keypoints. 
% Try first with normal featureExtraction function and then try extracting 
% HOG features for each of the cases. Compare the results. 
%% (Optional) Exercise 2: Panorama Stitching
% One of the applications of feature and keypoint detection is panorama
% stitching. In addition to what we covered today, the method for panorama
% stitching uses geometric transformations to align/stitch between the
% correspondences found in the image with one of the feature extractors.
% Matlab gives a good example on how to create a panoramic image here: 
% https://se.mathworks.com/help/vision/ug/feature-based-panoramic-image-stitching.html

% You can try creating a panoramic image of Gjovik given the images and
% following the tutorial. 
% Tip: you might want to resize the image to a smaller size to speed up the
% computations. 

%% (Optional) Exercise 3: Image Registration
% Another application is image registration, where we want to overlap
% two images of the same scene that are geometrically misaligned due to the
% geometric displacements. 
% Try this Matlab example: https://se.mathworks.com/help/vision/ug/find-image-rotation-and-scale-using-automated-feature-matching.html