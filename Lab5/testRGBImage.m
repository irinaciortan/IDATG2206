function testRGBImage()
clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
workspace;  % Make sure the workspace panel is showing.
format long g;
format compact;
fontSize = 20;

%===============================================================================
% Get the name of the first image the user wants to use.
baseFileName = '1n.jpg';
folder = fileparts(which(baseFileName)); % Determine where demo folder is (works with all versions).
fullFileName = fullfile(folder, baseFileName);

% Check if file exists.
if ~exist(fullFileName, 'file')
	% The file doesn't exist -- didn't find it there in that folder.
	% Check the entire search path (other folders) for the file by stripping off the folder.
	fullFileNameOnSearchPath = baseFileName; % No path this time.
	if ~exist(fullFileNameOnSearchPath, 'file')
		% Still didn't find it.  Alert user.
		errorMessage = sprintf('Error: %s does not exist in the search path folders.', fullFileName);
		uiwait(warndlg(errorMessage));
		return;
	end
end

%=======================================================================================
% Read in demo image.
rgbImage = imread(fullFileName);
% Get the dimensions of the image.
[rows, columns, numberOfColorChannels] = size(rgbImage);
% Display the original image.
subplot(2, 3, 1);
imshow(rgbImage, []);
axis on;
caption = sprintf('Original Color Image, %s', baseFileName);
title(caption, 'FontSize', fontSize, 'Interpreter', 'None');

% Set up figure properties:
% Enlarge figure to full screen.
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0.05 1 0.95]);
% Get rid of tool bar and pulldown menus that are along top of figure.
% set(gcf, 'Toolbar', 'none', 'Menu', 'none');
% Give a name to the title bar.
set(gcf, 'Name', 'Demo by ImageAnalyst', 'NumberTitle', 'Off')

drawnow;
hp = impixelinfo(); % Set up status line to see values when you mouse over the image.

% Get mask
[mask,maskedRGBImage] = createMask(rgbImage);
% Display the mask image.
subplot(2, 3, 2);
imshow(mask, []);
axis on;
caption = sprintf('Mask Image');
title(caption, 'FontSize', fontSize, 'Interpreter', 'None');

% Dilate slightly to merge the checkerboards together.
mask = imdilate(mask, ones(15));
% Get rid of blobs smaller than 10,000 pixels
mask = bwareafilt(mask, [10000, inf]);
% Fill holes.
mask = imfill(mask, 'holes');
% Get the convex hull of the blobs.
mask = bwconvhull(mask, 'objects');
% Label each blob with 8-connectivity, so we can make measurements of it
[labeledImage, numberOfBlobs] = bwlabel(mask, 8);
% Apply a variety of pseudo-colors to the regions.
coloredLabelsImage = label2rgb (labeledImage, 'hsv', 'k', 'shuffle'); 
% Display the pseudo-colored image.
subplot(2, 3, 3);
imshow(coloredLabelsImage);
axis on;
caption = sprintf('Mask Image');
title(caption, 'FontSize', fontSize, 'Interpreter', 'None');

% Get areas, and pixel values of the green channel.
props = regionprops(mask, rgbImage(:,:,2), 'Area', 'PixelValues');
allAreas = sort([props.Area], 'descend')
% Display the histogram of areas image.
subplot(2, 3, 4);
histogram(allAreas);
grid on;
caption = sprintf('Histogram of areas');
title(caption, 'FontSize', fontSize, 'Interpreter', 'None');

% Compute the standard deviation of the pixels
for k = 1 : length(props)
	theseValues = double(props(k).PixelValues); % Need to cast to double in order to pass it to std.
	stdDevs(k) = std(theseValues)
end

% Find the card.  It will have the highest standard deviation
[highestSD, index] = max(stdDevs)

% Extract that blob only
cardMask = ismember(labeledImage, index);
% Display the card mask image.
subplot(2, 3, 5);
imshow(cardMask);
axis on;
caption = sprintf('Card Mask Image');
title(caption, 'FontSize', fontSize, 'Interpreter', 'None');

% Mask the image using bsxfun() function to multiply the mask by each channel individually.
maskedRgbImage = bsxfun(@times, rgbImage, cast(cardMask, 'like', rgbImage));
% Display the masked RGB image.
subplot(2, 3, 6);
imshow(maskedRgbImage);
axis on;
caption = sprintf('Masked Image');
title(caption, 'FontSize', fontSize, 'Interpreter', 'None');

end

function [BW,maskedRGBImage] = createMask(RGB)
%createMask  Threshold RGB image using auto-generated code from colorThresholder app.
%  [BW,MASKEDRGBIMAGE] = createMask(RGB) thresholds image RGB using
%  auto-generated code from the colorThresholder app. The colorspace and
%  range for each channel of the colorspace were set within the app. The
%  segmentation mask is returned in BW, and a composite of the mask and
%  original RGB images is returned in maskedRGBImage.

% Auto-generated by colorThresholder app on 11-Dec-2017
%------------------------------------------------------


% Convert RGB image to chosen color space
I = rgb2hsv(RGB);

% Define thresholds for channel 1 based on histogram settings
channel1Min = 0.174;
channel1Max = 0.951;

% Define thresholds for channel 2 based on histogram settings
channel2Min = 0.000;
channel2Max = 0.118;

% Define thresholds for channel 3 based on histogram settings
channel3Min = 0.492;
channel3Max = 1.000;

% Create mask based on chosen histogram thresholds
sliderBW = (I(:,:,1) >= channel1Min ) & (I(:,:,1) <= channel1Max) & ...
	(I(:,:,2) >= channel2Min ) & (I(:,:,2) <= channel2Max) & ...
	(I(:,:,3) >= channel3Min ) & (I(:,:,3) <= channel3Max);
BW = sliderBW;

% Initialize output masked image based on input image.
maskedRGBImage = RGB;

% Set background pixels where BW is false to zero.
maskedRGBImage(repmat(~BW,[1 1 3])) = 0;

end
