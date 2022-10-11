function [BW,maskedImage] = segmentImage(X)
%segmentImage Segment image using auto-generated code from imageSegmenter app
%  [BW,MASKEDIMAGE] = segmentImage(X) segments image X using auto-generated
%  code from the imageSegmenter app. The final segmentation is returned in
%  BW, and a masked image is returned in MASKEDIMAGE.

% Auto-generated by imageSegmenter app on 10-Mar-2021
%----------------------------------------------------


% Adjust data to span data range.
X = imadjust(X);

% Find circles
[centers,radii,~] = imfindcircles(X,[45 68],'ObjectPolarity','dark','Sensitivity',0.98);
BW = false(size(X,1),size(X,2));
[Xgrid,Ygrid] = meshgrid(1:size(BW,2),1:size(BW,1));
for n = 1:5
    BW = BW | (hypot(Xgrid-centers(n,1),Ygrid-centers(n,2)) <= radii(n));
end

% Create masked image.
maskedImage = X;
maskedImage(~BW) = 0;
end

