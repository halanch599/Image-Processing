clear
clc
img = imread('C:\D\img\thermal.jpg');
size(img)
img1=rgb2gray(img);
figure
subplot(1,2,1), imshow(img)
subplot(1,2,2), imshow(img1)
max(img1(:))
min(img1(:))
img2 = img1+50;
figure, imshow(img2)
figure, imshow(img1)
max(img2(:))
img3=img1-52;
figure, imshow(img3)
close all
clc
clear
img1= imread('c:\d\img\lena.jpg');
img1= imread('c:\d\img\lion.jpg');
img2= imread('c:\d\img\lena.jpg');
figure,
subplot(2,2,1), imshow(img1)
subplot(2,2,2), imshow(img2)
img = img1+img2;
subplot(2,2,3), imshow(img)
alpha=0.5;
img = (img1*alpha)+(img2(1-alpha));
img = (img1*alpha)+(img2*(1-alpha));
subplot(2,2,3), imshow(img)
1-alpha
alpha=0.8;
img = (img1*alpha)+(img2*(1-alpha));
subplot(2,2,3), imshow(img)
alpha=0.3;
img = (img1*alpha)+(img2*(1-alpha));
subplot(2,2,3), imshow(img)
img = img1+100;
img = imadd(img1,100);
img = imdivide(img1,2);
figure, imshow(img)
figure, imshow(img1)
img = abs(img1-img2);
out = imabsdiff(img1,img2);
clc
img = rgb2gray(img1);
close all
imshow(img)
min(img(:))
max(img(:))
out = img>=100;
figure, imshow(out)
out = img>=130;
figure, imshow(out)
out1= rgb2gray(img2)>100;
figure, imshow(out)
figure, imshow(out1)
out1b= imcomplement(out1);
figure, imshow(out1b)
out1aj = 1-out1;
figure, imshow(out1aj)
close all
figure, imshow(img)
imgInvert =  255-img;
figure, imshow(imgInvert)
close all
clc
figure, imshow(out1)
figure, imshow(out)
img = out1 & out;
figure, imshow(img)
img = out1 | out;
figure, imshow(img)
figure, imshow(img1)
gray = rgb2rgay(img1);
gray = rgb2gray(img1);
figure, imshow(gray)
figure, imshow(gray>100)
figure, imshow(gray>50)
figure, imshow(gray>=50 & gray<=100 )
r = gray>=50 & gray<=100;
output =  img1.*r;
output =  img1.*uint8(r);
figure, imshow(output)
output =  gray.*uint8(r);
figure, imshow(output)
lc
clc
clear
close all
I = imread('c:\d\img\thermal.jpg');
figure, imshow(I)
gray = rgb2gray(I);
clear I
figure, imshow(gray)
255
out = imadjust(gray,[100 255], [0 255],1);
out = imadjust(gray,[0.4 1], [0 1],1);
figure, imshow(out)
out01 = im2double(gray);
max(out01(:))
min(out01(:))
figure, imshow(out01)
max(out(:))
min(out(:))
out0255 = im2uint8(out01);
max(out0255(:))
min(out0255(:))
stretchlim(gray)
out = imadjust(gray,stretchlim(gray), [0 1],1);
figure, imshow(out)
out = imadjust(gray,stretchlim(gray), [0 1],0.2);
figure, imshow(out)
out = imadjust(gray,stretchlim(gray), [0 1],10);
figure, imshow(out)
clc
imhist(gray)
prob = imhist(gray)/numel(gray);
figure, plot(prob)
