%% Week 8: Supervised Classification
% Author: Irina Ciortan

% When we perform supervised classification, we need to have a labeled
% dataset, where for each data instance (1D numeric data, 2D images, etc.),
% we know the ground-truth class where it belongs. 
% We need as well a set of features that are the discriminatory variables
% between classes.

%% Supervised classification of Numeric Data 
% For this example, we will be using the *Fisher iris* dataset. Fisher's 
% iris data consists of measurements on the sepal length, sepal width, petal
% length, and petal width for 150 iris specimens, belonging to 3 species: setosa,
% versicolor, verginica. There are 50 specimens per species. 

%%
% The iris dataset is included by Matlab.
% Load it:
load fisheriris
%%
% As you can see from the loaded data, the first 4 columns (*meas* array) are nuemric
% data, while the last one represents the label of the corresponding species. Every
% specimend belongs to a species. 

%%
% Try to visualize the distributions of all the specimens in a 2D scatter plot,
% based on only two variables (sepal length and sepal width):
f = figure;
%%
% Create a scatter plot with red, green, blue colormap and circle, square,
% diamon as markers.
gscatter(meas(:,1), meas(:,2), species,'rgb','osd'); 
% label the x,y axes
xlabel('Sepal length');
ylabel('Sepal width');

%%
% *Q*: What if you were to measure the sepal length and width of a new iris
% and based on the already known dataset, you'd want to fit the new
% measurement in one of the 3 species?
%%
% *A*: You train a model. This model is called a classifier. The known
% dataset is called train dataset.

%% Classifier: Decision Tree
% A Decision Tree infers a set of simple rules from the training data
% creating binary answers (yes/no) to each rule. The feature/variable at
% the root of the tree is chosen based on  Gini impurity index (basically,
% a metric that says how well a variable separates the classes).
%%
% The function *fitctree* builds a decision tree based on the training
% data. receives as input the features, the label and aliases abbreviations
% fo the feature names. 
t = fitctree(meas(:,1:2), species,'PredictorNames',{'SL' 'SW' });

%%
% A decision tree can be visualized as a tree data structure. 
view(t,'Mode','graph');
%%
% It is also interesting to visualize how the decision tree partitions the
% 2D space formed by the data distribution.
%%
% Create a grid of (x,y) values based on the limits of each axis
[x,y] = meshgrid(4:.1:8,2:.1:4.5);
x = x(:);
y = y(:);
%%
% And apply the classification function *predict* to that grid.
% Basically, this assigns a class to each point in the 2D plane defined by the
% chosen variables.
[grpname,node] = predict(t,[x y]);
%%
% Then create a scatter plot
figure, gscatter(x,y,grpname,'grb','sod'), title('Data space partitioned by a Decision Tree')
%%
% The division of the 2D plane is made in a non-linear way with a
% decision tree. However, the data is not very well clustered.

%%
% Once you have a model, you need to assess its performance with
% some metrics. 
% *resubPredict* returns a vector of predicted class labels (label) for the 
% trained classification model Mdl using the predictor data stored in Mdl.X.
treeClass = resubPredict(t);

%%
% *resubLoss* returns the resubstitution error, which is the misclassification 
% error (the proportion of misclassified observations) on the training set.
% It returns the in-sample classification loss, for the trained classification 
% model Mdl using the training data stored in Mdl.X and the corresponding 
% class labels stored in Mdl.Y.
% The lower the error, the more accurate the classification. 

treeResubErr = resubLoss(t);

%%
% *confusionchart* computes the confusion matrix. The confusion matrix
% shows how many of the specimens are correctly classified based on the
% chosen model. The diagonal of the confusion matrix represents the
% correctly classified data. The off-diagonal are the incorrectly
% classified data. 
figure
treeResubCM = confusionchart(species,treeClass);
title('CM for Decision Tree');

%%
% We have computed all these indicators of model performance  on the
% training data. Ideally, we would have another labeled dataset, different
% from the train set, called test set (that we never use when we build the
% model, we just use to check the performance of the model on unseen data).


%% Linear Discriminant Analysis (LDA) 
% Finds a linear combination of features that characterizes or separates two 
% or more classes.
% Function in matlab is *fitcdiscr*: receives as input the features and the labels.
% Let's try it on the Iris dataset.

lda = fitcdiscr(meas(:,1:2),species);
ldaClass = resubPredict(lda);
ldaResubErr = resubLoss(lda);
%%
% The misclassification error is higher than it is in the case of the
% decision tree. 
%%
% We can visualize the space partition as before.
grpname = predict (lda,[x y]);
%j = classify([x y],meas(:,1:2),species);
figure, gscatter(x,y,grpname,'grb','sod'), title ('Space partition with LDA');
%%
% The plots show that LDA cuts the space as if it was separate by lines.
% However, we can see the the boundaries between the classes is not a
% perfect line.

%% Quadratic Discriminant Analysis 
% When the separation between our data can't be done in a linear way, then
% we can choose to have quadratic boundaries. 
% We can use the same function, but we need to change the default state
% from linear to quadratic.
qda = fitcdiscr(meas(:,1:2),species, 'DiscrimType','quadratic')

% Compute the resubstitution prediction and error.
qdaClass = resubPredict(qda);
qdaResubErr = resubLoss(qda);
% The misclassification error is the same as in the linear case.

% We can visualize the space partition as before.
grpname = predict(qda,[x y]);
figure, gscatter(x,y,grpname,'grb','sod'), title ('Space partition with QDA');
% However, the quadratic boundaries better explain the data, as seen in the
% plot.
%%
% As mentioned above, a more maningful performance evaluation is to have a
% test set, separate from the train set. What to do?
%%
% # We could hold out a part of the train set for test.
% # If our overall dataset is small, we can perform cross-validation.

%% Cross-validation
% A stratified k-fold cross-validation is a popular choice for estimating 
% the test error on classification algorithms. It randomly divides the training 
% set into k disjoint subsets. Each subset has roughly equal size and roughly 
% the same class proportions as in the training set. 
% Every time, one subset is removed and the model is trained on the other nine subsets.
% Then, the trained model is used to classify the removed subset. 
% This is repeated for each of the ten subsets one at a time.
% A common number for k (the number of folds) is 5 or 10.
%%
% Cross-validation splits the data in a *random* way. Hence, the outcome depends on the initial random seed. 
% A good practice in order to be able to reproduce the results is to make sure to control the initial random seed. 

rng(0,'twister'); % initializes the Mersenne Twister generator with a seed of 0
% Split the available data into 10 folds.
cp = cvpartition(species,'KFold',10)

% Estimate the test error for a classifier using cross-validation on data
% folds. 
cvlda = crossval(lda,'CVPartition',cp);
ldaCVErr = kfoldLoss(cvlda)

% Let's compute the test error for the tree classifier as well.
cvt = crossval(t,'CVPartition',cp);
tCVErr = kfoldLoss(cvt)

%%
% The decision tree has a larger cross-validation error than LDA, even 
% even though it had a lower error for the original split. 
%% Demo Classification Learner App
% Classification Learner App provides a GUI, where you don't need to write
% code since many of the classification functions are integrated. You just
% need to import the data into an array/table format. One of the columns
% needs to be the labels. One of the biggest advantages of using the
% app is that you can perform batch processing: train all existing
% classifiers with just a click. 

% Let's import the fisher iris dataset into a table:
% We could have also concatenated the existing data, but we miss the
% headers, i.e. name of the variables.
fishertable = readtable('fisheriris.csv');

% Start the app with a command or from Apps -> Classification Learner App
classificationLearner;

%% Exercise: Supervised Image Classification based on Histogram of Oriented (HOG) features.
% Now let's try a classification example that deals with image data.
% The Flowers data set contains 3670 images of flowers belonging to five classes 
% (daisy, dandelion, roses, sunflowers, and tulips).
% Download and extract the Flowers data set from http://download.tensorflow.org/example_images/flower_photos.tgz. 

%%
clear global
url = 'http://download.tensorflow.org/example_images/flower_photos.tgz';
downloadFolder = pwd; % it will save the data in the current directory
filename = fullfile(downloadFolder,'flower_dataset.tgz');
dataFolder = fullfile(downloadFolder,'flower_photos');
%%
% You need to download the data only ONCE. Uncomment the following lines
% for the first time, then comment them again to avoid re-download.
% if ~exist(dataFolder,'dir')
%     fprintf("Downloading Flowers data set (218 MB)... ")
%     websave(filename,url);
%     untar(filename,downloadFolder)
%     fprintf("Done.\n")
% end

% If you navigate to the path given by *dataFolder* you will see the images 
% copied locally.
%% 
% Now that we copied the data in our pc, we can make use of Matlab data
% format, which is very useful for classification of images: imageDatastore.

imds = imageDatastore(dataFolder,'IncludeSubfolders',true, 'LabelSource','foldernames');
trainingSet = imds;

%%
% As we saw last time, extractHOGfeature function returns a visualization output 
% that can guides the user into assessing whether the shape information is encoded in a menaingful way
% and at a proper scale.
% By varying the HOG cell size parameter and visualizing the result, you can see 
% the effect the cell size parameter has on the amount of shape information
% encoded in the feature vector.

%Read a random image representative of the daisy class
img = readimage(trainingSet, 400);
img = imresize(img, [256 256]);
figure, imshow(img), title('Random daisy');
% Extract HOG features and HOG visualization
[hog_4x4, vis4x4] = extractHOGFeatures(img,'CellSize',[4 4]);
[hog_8x8, vis8x8] = extractHOGFeatures(img,'CellSize',[8 8]);
[hog_16x16, vis16x16] = extractHOGFeatures(img,'CellSize',[16 16]);
[hog_32x32, vis32x32] = extractHOGFeatures(img,'CellSize',[32 32]);

% Show the original image
figure; 
subplot(2,4,1:4); imshow(img); title('HOG feature extraction at different scale');

% Visualize the HOG features
subplot(2,4,5);  
plot(vis4x4); 
title({'CellSize = [4 4]'; ['Length = ' num2str(length(hog_4x4))]});

subplot(2,4,6);
plot(vis8x8); 
title({'CellSize = [8 8]'; ['Length = ' num2str(length(hog_8x8))]});

subplot(2,4,7);
plot(vis16x16); 
title({'CellSize = [16 16]'; ['Length = ' num2str(length(hog_16x16))]});

subplot(2,4,8);
plot(vis32x32); 
title({'CellSize = [32 32]'; ['Length = ' num2str(length(hog_32x32))]});

% Decide upon the best feature size
cellSize = [16 16];
hogFeatureSize = length(hog_16x16);
%%
% Loop over the trainingSet and extract HOG features from each image. A
% similar procedure will be used to extract features from the testSet.

numImages = numel(trainingSet.Files);
trainingFeatures = zeros(numImages,hogFeatureSize,'single');

for i = 1:numImages
    img = readimage(trainingSet,i);
    % Resize image to 256x256 to make sure we get the same size for the
    % feature vector
    img = imresize(img, [256 256]);
    trainingFeatures(i, :) = extractHOGFeatures(img,'CellSize',cellSize);  
end

% Get labels for each image.
trainingLabels = trainingSet.Labels;

%%
% Now we have two options: we can use the Classification Learner app or
% manually split the data into train and test. 

% For the app, we need to create a table out of the training features and
% training labels
flowers_table = array2table(trainingFeatures);
flowers_table(:, end+1) = num2cell(trainingLabels);
%%
% Start the app and then choose 10 folds for cross-validation.
% Since we have a higher number of features, we can use *feature selection*
% by enabling PCA dimensionality reduction technique (look for the PCA option
% in the menu).
classificationLearner
%% 
% Manually split the image datastore into 10 folds to perform
% cross-validation.
% Use the splitEachLabel function to divide the image datastore into 10
% sets. 

rng(0,'twister'); % initializes the Mersenne Twister generator with a seed of 0
[imd1 imd2 imd3 imd4 imd5 imd6 imd7 imd8 imd9 imd10] = splitEachLabel(imds,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,'randomize');
partStores{1} = imd1.Files ;
partStores{2} = imd2.Files ;
partStores{3} = imd3.Files ;
partStores{4} = imd4.Files ;
partStores{5} = imd5.Files ;
partStores{6} = imd6.Files ;
partStores{7} = imd7.Files ;
partStores{8} = imd8.Files ;
partStores{9} = imd9.Files ;
partStores{10} = imd10.Files; 
for i = 1 :10
    
    test_idx = i;
    train_idx = ~test_idx;
    imdsTest = imageDatastore(partStores{test_idx}, 'IncludeSubfolders', true,'FileExtensions','.jpg', 'LabelSource', 'foldernames');
    imdsTrain = imageDatastore(cat(1, partStores{train_idx}), 'IncludeSubfolders', true,'FileExtensions','.jpg', 'LabelSource', 'foldernames');
        %%Choose desired classification function for training
        %%Then use model to predict on test set
        %%Compute the misclassification error for the test set
end
% In the end average all the errors. 