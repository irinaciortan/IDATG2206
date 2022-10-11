function outputImg = mycorr(inputImg,filt)
%MYCORR This function computes 2D correlation with padding.

filtSize = size(filt,1);
%Since filter values are double, change the image from uint8 to double
inputImg = double(inputImg);

%Approximate the nr of padded columns/rows to integer.
padSize = floor(filtSize/2);

%Create padded image with correct size.
paddedImg = zeros([size(inputImg,1)+2*padSize, size(inputImg,2)+2*padSize]);
paddedImg(padSize+1:end-padSize, padSize+1:end-padSize) = inputImg;

%We can loop now through all original image coordinates.
for i =1:size(inputImg,1)
    for j=1:size(inputImg,2)
        outputImg(i,j) = sum(paddedImg(i:i+filtSize-1, j:j+filtSize-1).*filt, 'all');
    end
end

%Transform outputImg from double to uint8.
outputImg = uint8(outputImg);
end

