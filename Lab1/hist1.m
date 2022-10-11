function h = hist1(img)
%%Function HIST1 returns the histogram of an image.
%%h is the output(the histogram)
%%img is the 8-bit grayscale input image
%%From another m-file, or from the command line call hist1(IMG)
%%where IMG is a greyscale image.

%Preallocate h, since it's of fixed size, equal to the number of intensity levels.
%Since img is 8-bit grayscale image, there are 2^8 gray levels, so 256 levels
%(with values between 0 and 255).
h=zeros(256, 1);
% The indices in the h array will correspond to the grayscale levels/bins
% (values between 0-255) + 1 (to match the 1-indexing), while
% the value of h at each index represents the counts of these grayscale value encoded in the index.

%% Method 1: Solving the problem with two loops, indexing each pixel in the image
for i=1:size(img,1)
    for j=1:size(img,2)
        %Get the pixel at row i and colum j
        px = img(i,j);
        % Increment the bin in the histogram corresponding to the grayscale
        % value of px.
        % Because px is type uint8 (unsigned integer 8 bits), the sum between (px+1) is
        % automatically compiled as uint8. Therefore the maximum values
        % (px+1) can take is (2^8 -1)=255. However, h has 256 elements. Thus,
        % before incrementing, we need to cast px to a greater type,
        % uint16 (unsigned integer 16 bits).
        px = uint16(px);
        h(px+1) =h(px+1)+1;
        % If you want to save up on memory, you can directly write the
        % following lines by defining less variables
        % h(uint16(img(i,j))+1) =h(uint16(img(i,j))+1)+1;
    end
end

%% Method 2. create a 1 dimensional vector out of the image so that we minimize the loops
% img_vec=img(:); 
% the line above is equivalent to img_vec = reshape(img,[size(img,1)*size(img,2),1])
% for i =1:size(img_vec)
%         %As explained above, changing the type to uint16 is important, otherwise
%         %we will not be able to index h at position 256.
%         h(uint16(img_vec(i))+1)=h(uint16(img_vec(i))+1)+1;
%
% end
end