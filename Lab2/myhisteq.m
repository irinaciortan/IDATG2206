function [im_new,cdfnorm, im_new_reshaped] = myhisteq(im)
%MYHISTEQ Your own computation of histogram equalization
%   im_new - is the output image with equalized histogram
%   cdfnorm - is the normalized cumulative distribution function that is the
%   mapping from input to output image
%   im_new_reshaped = is the same as im_new, with difference in computation
%   (saves one loop).

%first compute the histogram of the image
h = imhist(im);

for i=%%FILL in HERE the range
    cdf(i) = %%FILL IN HERE with formula for cumulative sum of histogram
end
%normalize cdf
cdfnorm = round(((cdf -min(cdf))/(size(im, 1)*size(im, 2) - min(cdf)))*255);


im_new = im;
for i=1:size(im,1)
    for j=1:size(im,2)
        im_new(i,j) = %FILL IN HERE: apply mapping to input image based on cdfnorm
    end
end

%Or if you want to avoid two loops:
im_onedim = im(:);
for i= %FILL in HERE
    im_onedim_new(i) = %FILL in HERE: apply mapping to input image based on cdfnorm
end
%reshape the 1D response back to two dimensions
im_new_reshaped = uint8(reshape(im_onedim_new, [size(im,1) size(im,2)]));
end

