function outputImg = mycorr(inputImg,filt)
%MYCORR This function computes 2D correlation without padding.
filtSize = size(filt,1);
%since filter values are double, change the image from uint8 to double
inputImg = double(inputImg);

%First option: no padding
for i =1:size(inputImg,1)-filtSize+1
    for j=1:size(inputImg,2)-filtSize+1
        outputImg(i,j) = sum(inputImg(i:i+filtSize-1, j:j+filtSize-1).*filt, 'all');
        
    end
end

%Transform outputImg from double to uint8.
outputImg = uint8(outputImg);
end



function outputImg = mycorr(inputImg,filt)
%MYCORR This function computes 2D correlation without padding.
filtSize = size(filt,1);
%since filter values are double, change the image from uint8 to double
inputImg = double(inputImg);

%First option: no padding
for i =1:size(inputImg,1)-filtSize+1
    for j=1:size(inputImg,2)-filtSize+1
        outputImg(i,j) = %FILL IN HERE
        
    end
end

%Transform outputImg from double to uint8.
outputImg = uint8(outputImg);
end