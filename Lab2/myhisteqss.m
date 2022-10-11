function [im_new,cdfnorm, im_new_reshaped] = myhisteqss(im)
%MYHISTEQ FULL FUNCTION
h = imhist(im);

for i=1:256
    cdf(i) = sum(h(1:i));
end
%normalize cdf
if min(cdf) == 0
    dummy = sort(cdf)
    nonzmin = dummy(2)
else
    nonzmin = min(cdf)
cdfnorm = round((cdf -nonzmin)/(size(im, 1)*size(im, 2) - nonzmin)*255);
%figure, plot(cdfnorm),hold on, plot(m)
%apply the mapping to image
im_new = im;
for i=1:size(im,1)
    for j=1:size(im,2)
        im_new(i,j) = cdfnorm(im(i,j)+1);
    end
  end

%Or if you want to avoid two loops:
im_onedim = im(:);
for i=1:size(im_onedim)
    im_onedim_new(i) = cdfnorm(im_onedim(i)+1);
end
%reshape the 1D response back to two dimensions
im_new_reshaped = uint8(reshape(im_onedim_new, [size(im,1) size(im,2)]));
end

