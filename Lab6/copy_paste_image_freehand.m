% Lets user freehand draw a region on an image, then paste it onto an image of the same size.
clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
imtool close all;  % Close all imtool figures.
clear;  % Erase all existing variables.
workspace;  % Make sure the workspace panel is showing.
fontSize = 20;
format compact;

% Check that user has the Image Processing Toolbox installed.
hasIPT = license('test', 'image_toolbox');
if ~hasIPT
	% User does not have the toolbox installed.
	message = sprintf('Sorry, but you do not seem to have the Image Processing Toolbox.\nDo you want to try to continue anyway?');
	reply = questdlg(message, 'Toolbox missing', 'Yes', 'No', 'Yes');
	if strcmpi(reply, 'No')
		% User said No, so exit.
		return;
	end
end

% Ask user if tey want to use a color or gray scale image.
folder = fileparts(which('cameraman.tif')); % Determine where demo folder is (works with all versions).
button = menu('Use which demo image?', 'Color office scene', 'Gray scale cells');
if button == 1
	baseFileName1 = 'office_2.jpg';
	baseFileName2 = 'office_6.jpg';
else
	baseFileName1 = 'AT3_1m4_01.tif';
	baseFileName2 = 'AT3_1m4_09.tif';
end

% Read in a standard MATLAB demo image.
% Get the full filename, with path prepended.
fullFileName = fullfile(folder, baseFileName1);
% Check if file exists.
if ~exist(fullFileName, 'file')
	% File doesn't exist -- didn't find it there.  Check the search path for it.
	fullFileName = baseFileName1; % No path this time.
	if ~exist(fullFileName, 'file')
		% Still didn't find it.  Alert user.
		errorMessage = sprintf('Error: %s does not exist in the search path folders.', fullFileName);
		uiwait(warndlg(errorMessage));
		return;
	end
end
sourceImage = imread(fullFileName);
% Get the dimensions of the image.
[rows1, columns1, numberOfColorBands1] = size(sourceImage);
% Display the original image on the left hand side.
subplot(1, 2, 1);
imshow(sourceImage);
axis on;
caption = sprintf('Source image, %s', baseFileName1);
title(caption, 'FontSize', fontSize, 'Interpreter', 'none');
% Enlarge figure to full screen.
set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
% Give a name to the title bar.
set(gcf,'name','Demo by ImageAnalyst','numbertitle','off')

% Display a second image of the same size beside it on the right.
% Get the full filename, with path prepended.
fullFileName = fullfile(folder, baseFileName2);
targetImage = imread(fullFileName);
% Get the dimensions of the image.
[rows2, columns2, numberOfColorBands2] = size(sourceImage);
% Get the full filename, with path prepended.
fullFileName = fullfile(folder, baseFileName2);
subplot(1, 2, 2);
imshow(targetImage);
axis on; % Show distances with tick marks.
caption = sprintf('Target image, %s, original', baseFileName2);
title(caption, 'FontSize', fontSize, 'Interpreter', 'none');

% Bail out if the images are not the same size laterally.
if rows1 ~= rows2 || columns1 ~= columns2
	warningMessage = sprintf('Error: images must be the same size.\nYour image 1 is %d rows by %d columns,\nwhile your image 2 is %d rows by %d columns.',...
		rows1, columns1, rows2, columns2);
	uiwait(warndlg(warningMessage));
	return;
end

% Ask user to draw freehand mask.
message = sprintf('In the LEFT IMAGE...\nLeft click and hold to begin drawing.\nSimply lift the mouse button to finish');
subplot(1, 2, 1);
uiwait(msgbox(message));
hFH = imfreehand(); % Actual line of code to do the drawing.
% Create a binary image ("mask") from the ROI object.
mask = hFH.createMask();
xy = hFH.getPosition;

% Paste it onto the target image.
% Method depends on whether the source and target images are gray scale or color.
if numberOfColorBands1 == 1 && numberOfColorBands2 == 1
	% Both gray scale.
	targetImage(mask) = sourceImage(mask);
elseif numberOfColorBands1 == 3 && numberOfColorBands2 == 1
	% Source image is color and target image is gray scale.
	grayImage = rgb2gray(sourceImage);
	targetImage(mask) = grayImage(mask);
elseif numberOfColorBands1 == 1 && numberOfColorBands2 == 3
	% Source image is gray scale and target image is color.
	% Extract the individual red, green, and blue color channels.
	redChannel2 = targetImage(:, :, 1);
	greenChannel2 = targetImage(:, :, 2);
	blueChannel2 = targetImage(:, :, 3);
	% Do the replacements on each color channel.
	redChannel2(mask) = sourceImage(mask);
	greenChannel2(mask) = sourceImage(mask);
	blueChannel2(mask) = sourceImage(mask);
	targetImage = cat(3, redChannel2, greenChannel2, blueChannel2);
elseif numberOfColorBands1 == 3 && numberOfColorBands2 == 3
	% Source image is color and target image is color.
	% Extract the individual red, green, and blue color channels.
	redChannel1 = sourceImage(:, :, 1);
	greenChannel1 = sourceImage(:, :, 2);
	blueChannel1 = sourceImage(:, :, 3);
	% Extract the individual red, green, and blue color channels.
	redChannel2 = targetImage(:, :, 1);
	greenChannel2 = targetImage(:, :, 2);
	blueChannel2 = targetImage(:, :, 3);
	% Do the replacements on each color channel.
	redChannel2(mask) = redChannel1(mask);
	greenChannel2(mask) = greenChannel1(mask);
	blueChannel2(mask) = blueChannel1(mask);
	targetImage = cat(3, redChannel2, greenChannel2, blueChannel2);
end

% Display new image.
subplot(1, 2, 2); % Switch active axes to right hand axes.
imshow(targetImage);
axis on;
caption = sprintf('Target image, %s, after paste', baseFileName2);
title(caption, 'FontSize', fontSize, 'Interpreter', 'none');
