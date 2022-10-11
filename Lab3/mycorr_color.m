function outputImg = mycorr_color(inputImg,filt)
%MYCORR_COLOR This function computes 2D correlation with padding, for color images.

filtSize = size(filt,1);
%Since filter values are double, change the image from uint8 to double
inputImg = double(inputImg);

%Approximate the nr of padded columns/rows to integer.
padSize = floor(filtSize/2);

%Create padded image with correct size. This time the padded image has 3
%dimensios, as the color input image.
paddedImg = zeros([size(inputImg,1)+2*padSize, size(inputImg,2)+2*padSize, size(inputImg,3)]);
paddedImg(padSize+1:end-padSize, padSize+1:end-padSize, :) = inputImg;

%We add one more level for looping through the color channels
for k = 1:size(inputImg,3)
    for i =1:size(inputImg,1)
        for j=1:size(inputImg,2)
            outputImg(i,j,k) = sum(paddedImg(i:i+filtSize-1, j:j+filtSize-1, k).*filt, 'all');
        end
    end
end

%Transform outputImg from double to uint8.
outputImg = uint8(outputImg);
end



