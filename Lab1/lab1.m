%% Week 1 : Introduction to MATLAB % This week's assignment is a quick introduction to MATLAB for image analysis.% Adapted by Irina Ciortan from the original file created by Fritz Albertsen.%% A 'double comment' like this creates what MATLAB refers to as a block.% (Which, when I publish it online, becomes just simple text)%%% Make a folder named 'IDATG2206' and set this as the 'current directory'% using the user interface or using the MATLAB-function 'cd' % (that stands for change directory).% When using the CD function and to use MATLAB more efficiently,% the autocompletion is often handy. Activate it by hitting 'Tab'.% Save a copy of this file in that folder, and open it in MATLAB.% To run a part of the code, select the code and hit F9.% To run the selected block (the block with the caret), hit 'Ctrl+Enter'.% To run an entire file, you can:%  - type the name of the file, e.g. 'lab1', at the command prompt, or%  - select the file and hit F9, or%  - open the file and hit F5.%% 1. Matrices% The matrix | 1 0 0 |%            | 0 1 0 |%            | 0 0 1 |% may be created like this:someMatrix = [1 0 0 ; 0 1 0 ; 0 0 1]% e.g. [row1 ; row2 ; ... ; rowM], where a row can be defined as% [number1 number2 ... numberN] or [number1, number2, ..., numberN]% MATLAB comes built-in with function for many common operations,% for the case above we could have typed:someMatrix = diag([1 1 1])% If you wonder how a function/operator works, type 'help <function name>'% or for (a sometimes more thorough) description in a separate window,% type 'doc <function name>'.% It is also possible to simply press F1 while the cursor is at a certain% function to access MATLAB's help for that particular funtion.% MATLAB matrices are dynamic objects, meaning one need not know the exact% size of a matrix in advance. Initially a simple matrix can be made, and% and later rows or colums may be added or removed. If, however, the size% is known in advance and the matrix is large, it is wise to preallocate% memory for the matrix before you start using it. This may greatly improve% performance.% E.g. allocate memory for 100*100 matrix and fill it with 1's:big_1 = ones(100); %<-- Terminating a line with ';' will suppress output.% Or, allocate a 50*40*3 matrix and fill it with 0's:big_0 = zeros([50 40 3]);% Another function to automatically allocate and fill a matrix is magic(n).% Let's build a 5x5 magic square matrix with values between 1 and 5^2.magic_5 = magic(5)%Or if we want  a matrix of n random numbers with values between 0 and 1, then we use%rand(n).rand_01 = rand(4)%If you want to change the interval from (0, 1) to (a, b), the formula is r = a + (b-a).*rand(n).rand_14 = 1+ (4-1)*rand(4)% A vector can be created using the '<from>:<step>:<to>' syntax:steg1 = 1:2:50;% The <step> is optional. If omitted the step defaults to 1:steg2 = 1:50;% The standard plotting function is PLOT, e.g.:plot(steg1);% An alternative is STEM. Look up how it works using 'help stem'.% You can add a title to the plot by using the title() function:title('Demonstration of plot()');% Extracting a value, or a series of values, from a matrix is easily% achieved like this:mat = [1 7 4 ; 3 4 5]mat(2,1) % retrives the '3'. NOTE: The first index is 1 in MATLAB!% MATLAB lays out a column at a time in memory, hence the value '7' can% either be retrieved using linear index:mat(3)% or using (row,column)-index:mat(1,2)% A range of elements may retrieved using the <from>:<to> syntax:mat(1:2,1)     % First columnmat(1:end,1)   % The same, since n = 2mat(:,1)       % The same, : is here 'all rows'% We can use the same syntax to set a range of elements:mat(1:2,1:2) = 0% Note that a matrix can be stored in various formats, e.g. UINT8, INT8, SINGLE or% DOUBLE. They all have their conversion functions, see e.g. 'help uint8'.% If your result looks fishy, and you have a hard time figuring out why,% a type conversion (of lack of one) may be the reason!%% 2. Matrix operations% Here we'll demonstrate how some simple matrix operations is done in% MATLAB% We'll startmat1 = [2 3; 5 7];mat2 = [4 5; 1 9];% Elementwise addition and subtraction.mat1 + mat2mat1 - mat2% Transpose.mat1.'% Complex conjugate transpose (same as transpose when the matrix is real).mat1'% But notice the difference for a complex matrixcomplexMatrix = [0 1i-2; 1i+2 0]% Transpose for complex, see the function transposecomplexMatrix.'% Complex conjugate transpose, see the function ctransposecomplexMatrix'% Multiplication, division and power.mat1 * mat2mat1 / mat2mat1^2% Elementwise multiplication, division and power% (add '.' in front of the operator).mat1 .* mat2mat1 ./ mat2mat1 .^ mat2% Row-wise or column-wise summation, average and max/min of elements using in-built functions.magic_5sum(magic_5) %the default dimension, dimension 1 is along the columns.sum(magic_5,2) %dimension 2 is the rows.sum(magic_5, 'all'), % sums all the elements in the array.mean(magic_5) %the average value for every column.mean2(magic_5) %the average value for all elements in the matrix.max(magic_5) %the max value for every column.min(magic_5) %the min value for every column.max(magic_5,[],'all') % the max value among all elements in the matrix.%% 3. Control structures% While and for loops.% Can often be used instead of matrix operations, but tend to lead to a% slightly inferior perfomance, especially in older versions of MATLAB.%% For some tips on how to write fast MATLAB-code:% http://www.mathworks.com/matlabcentral/fileexchange/loadFile.do?objectId=5685%% The reason for introducing performance-tips already is because we will be% storing images as matrices, and these tend to be large. If we also% consider that the algorithms we use are computationally intensive,% writing somewhat optimised code is often worth the extra effort.% E.g. can this simple loop:A = zeros(1,100);for i = 1:100    A(i) = 2;end% be replaced by a single and more efficient command:A = ones(1,100) * 2;% And why write:for i = 1:100    for j = 1:100        stor_1(i,j) = i^2; % MATLAB warning: consider preallocating for speed    endend% when you can do this:big_1_temp = repmat((1:100)'.^2, 1,100);% or alternatively:big_1_temp2 = (1:100)'.^2 * ones(1,100);% Note:% If you are not used to 'doing everything with vectors', you'll likely% want to use loops far more often than you need to. When you get the hang% of it, typing vector-code is faster and often less error-prone than the% loop-equivalents, but loops offer better flexibility. % As always, add lots of COMMENTS, especially for complex one-liners!% Logical structures: if-elseif-else. Which number is displayed?if -1    1elseif -2    2else     3end% Logical structures: switch-case.var = 't';switch var    case 't'        1    case 2        2end%% 4. The FIND function - see 'help find'% A very handy method, but may initially seem a bit tricky.% Example: Find elements which are equal to 4.M = [1 2 3 ; 4 4 5 ; 0 0 2]M == 4                 % a logical mapI = find(M == 4)       % the linear indices[II, J] = find(M == 4) % the (row,column)-indicesM(M == 4) = 3          % change them to 3M(I) = 4               % and back to 4%% 5. Images% Now, we'll finally look at some images. % Open a couple of images.img1 = imread('football.jpg');img2 = imread('coins.png');% Display the first image.imshow(img1)title('The first image'); % A title is nice% Display the second image in a new figure.figure, imshow(img2)title('The second image');%Display both images as subplots of the same figure.figure, subplot(121), imshow(img1), title('The first image');subplot(122), imshow(img2), title('The second image');% Another function for displaying the images is:figure, imagesc(img1), colorbartitle('Image displayed by imagesc');% Histogram the image intensity values.H = imhist(img2);plot(H)title('The histogram for the second image');% Converting a colour image with 3 colour channels to a greyscale image% (the Y in the YIQ colour model).img1 = imread('football.jpg');img3 = rgb2gray(img1);figure, imshow(img3)title('Grayscale image of football');%% EXERCISE 1%%%%%%%%%%%%%%%%% Make a function that returns the same as IMHIST when the parameter is a% 8-bits greyscale image.%% Although it is allowed to use loops, try to avoid using them where% it is possible. One loop should suffice.%% Hint: How to create and use the function% A function is stored in .m files with the same name as the function.% 1. Create an m-file named 'hist1.m', e.g. using 'edit hist1.m'% 2. The first line should be 'function h = hist1(img)'% 3. Write the code to produce a histogram of IMG below the function%    declaration. The histogram should be stored in a variable named h.% 4. Save the file.% 5. From another m-file, or from the command line call hist1(IMG)%    where IMG is a greyscale image.%%%%%%%%%%%%%%%%%% EXERCISE 2 %%%%%%%%%%%%%%%%% Above, we loaded the image 'coins.png' using the command:img2 = imread('coins.png');%% a)% Use the operators >, <, >=, <= to threshold IMG2 using an arbitrary% threshold.%% b)% Image Processing Toolbox (IPT) in MATLAB have a function for computing% the 'optimal' threshold based on Otsu's algorithm.%  - Find this function using MATLAB's help system.%  - Use it to with the 'optimal' threshold.%  - Use the threshold and the function IM2BW to threshold IMG2.%% c)% Compare the binary image resulting from part a with the one from part b% by displaying the images. Do you notice any differences?% Display also the difference between the images.%%%%%%%%%%%%%%%%