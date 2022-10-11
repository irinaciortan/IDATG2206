plane = imread('liftingbody.png');
[a,b,c] = myhisteqss(plane);
[planeq,m] = histeq(plane);
figure, plot(b),hold on, plot(m*255), legend('My CDF', 'Matlab CDF');
figure, imshow(a)
figure, imshow(c)

%plot the equalized histogram
figure
subplot(121), imhist(planeq), title('Matlab Hist Eq');
subplot(122), imhist(c), title('My Hist Eq');