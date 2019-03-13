%-- 13.03.2019 09:57 --%
kernel = [ 1 1 1; 1 1 1; 1 1 1]./9
img = imread('c:\d\img\coins.jpg');
figure, subplot(2,2,1),imshow(img,[]), title('Original Image')
help filter2
help conv2
clc
im = imfilter(img,kernel);
subplot(2,2,2), imshow(im),title('Filter 3x3')
kernel5x5 = ones(5,5)/25
im = imfilter(img,kernel5x5);
subplot(2,2,3), imshow(im),title('Filter 5x5')
kernel5x5 = ones(15,15)/(15*15);
im = imfilter(img,kernel5x5);
subplot(2,2,4), imshow(im),title('Filter 15x15')
imtool(im)
h = fspecial('average',[5,5]);
h
close all
clc
figure, imshow(img)
noise = randn(3,3)
noise = randn(3,3)*0.5
noise = randn(size(img))*0.5
noise = randn(size(img))*0.5;
imgn = img+noise;
imgn = double(img)+noise;
figure,imshow(imgn)
figure,imshow(imgn,[])
noise = randn(size(img));
imgn = double(img)+noise;
figure,imshow(imgn,[])
close all
help imnoise
I = imread('eight.tif');
J = imnoise(I,'salt & pepper', 0.02);
figure, imshow(I), figure, imshow(J)
J = imnoise(I,'salt & pepper', 0.5);
figure, imshow(I), figure, imshow(J)
J = imnoise(I,'salt & pepper', 0.05);
figure, imshow(I), figure, imshow(J)
I = gpuArray(imread('eight.tif'));
J = imnoise(I,'salt & pepper',0.02);
K = medfilt2(J);
figure, montage({J,K})
help imnoise
I = imread('eight.tif');
J = imnoise(img,'salt & pepper', 0.05);
figure, imshow(I), figure, imshow(J)
figure, imshow(medfilt2(img))
close all
clc
figure, imshow(img)
img2=imresize(img,0.5);
figure, imshow(img2)
img2=imresize(img,0.2);
figure, imshow(img2)
clc
rand(3,3)
randi(3,3)
rand(3,3)
clc
help ordfilt2
A = imread('snowflakes.png');
B = ordfilt2(A,25,true(5));
figure, imshow(A), figure, imshow(B)
B = ordfilt2(A,1,true(5));
figure, imshow(A), figure, imshow(B)
B = ordfilt2(A,10,true(3));
B = ordfilt2(A,9,true(3));
figure, imshow(A), figure, imshow(B)
B = ordfilt2(img,9,true(3));
figure, imshow(A), figure, imshow(B)
figure, imshow(img), figure, imshow(B)
B = ordfilt2(img,1,true(3));
figure, imshow(img), figure, imshow(B)
B = ordfilt2(img,1,true(25));
figure, imshow(img), figure, imshow(B)
close all
clc
fspecial('gaussian',[3,3],1)
plot(ans)
k = fspecial('gaussian',[3,3],1)
plot(k(:))
test4
close all
test4
clc
clear
img = imread('c:\d\img\coins.jpg');
kernel = fspecial('laplacian',1)
kernel = fspecial('laplacian',-1)
help fspecial
kernel = fspecial('laplacian',.5)
kernel = fspecial('laplacian',0)
out = imfilter(img,kernel);
figure,
imshow(img)
figure, imshow(out_
figure, imshow(out)
kernel
kernel = -kernel
figure, imshow(imfilter(img,kernel));
kernel = fspecial('sobel')
figure, imshow(imfilter(img,kernel));
figure, imshow(imfilter(img,kernel'));
img = imread('c:\d\img\object.jpg');
img = imread('c:\d\img\objects.jpg');
gray = rgb2gray(img);
figure, imshow(imfilter(gray,kernel'));
figure, imshow(imfilter(gray,kernel));
figure, imshow(gray);
figure, imshow(abs(imfilter(gray,kernel)));
close all
kernel'
fspecial('canny')
help edge
I = imread('circuit.tif');
BW1 = edge(I,'prewitt');
BW2 = edge(I,'canny');
figure, imshow(BW1)
figure, imshow(BW2)
BW2 = edge(I,'sobel');
figure, imshow(BW2)
clc
img = imread('rice.png');
imshow(img), title('Original Image');
figure, imshow(img), title('Original Image');
out =imsharpen(img);
figure, imshow(out)
close all
figure, imshow(out)
figure, imshow(img)
help imsharpen
b = imsharpen(a,'Radius',5,'Amount',5);
b = imsharpen(img,'Radius',5,'Amount',5);
figure, imshow(b)
close all
clc
clear
img = imread('rice.png');
figure, imshow(img), title('Original Image');
out = imsharpen(img,'Radius',2,'Amount',1);
figure, figure, out), title('Output Image');
clc
img = imread('rice.png');
figure, imshow(img), title('Original Image');
out = imsharpen(img,'Radius',2,'Amount',1);
figure, imshow(out), title('Output Image');
blur = imfilter(img,fspecial('gaussian'));
figure, imshow(img)
figure, imshow(blur)
blur = imfilter(img,fspecial('gaussian',[5,5]));
figure, imshow(blur)
close all
figure, imshow(blur), title('blur')
figure, imshow(img), title('orignal')
blur = imfilter(img,fspecial('gaussian',[9,9]));
figure, imshow(blur), title('blur')
fspecial('gaussian',[9,9])
fspecial('gaussian')
fspecial('gaussian',[5,5],2)
kernel = fspecial('gaussian',[5,5],2)
figure, imshow(imfilter(img,kernel)), title('blur')
blur =imfilter(img,kernel);
img2=img-blur;
figure, imshow(img2), title('img-blur')
max(img2(:))
figure, imshow(img2,[]), title('img-blur')
mask=img-blur;
k=5;
k=2;
g = img + (k*mask);
figure, imshow(g)
close all
figure, imshow(img),title('original')
figure, imshow(g),title('enhanced')
close all
test4
close all
clear all
clc
clear
clc
close all
clear lang
clear I
clc
%-- 13.03.2019 02:38 --%
img = imread('c:\d\img\coins.jpg');
figure, imshow(img)
mask = ones(3,3)/9
ones(3,3)
mask
out = imfilter(img,mask);
figure, imshow(out,[])
mask = ones(7,7)/49;
out = imfilter(img,mask);
figure, imshow(out,[])
mask = ones(35,35)/(35*35);
out = imfilter(img,mask);
figure, imshow(out,[])
clc
mask = fspecial('average',[3,3])
mask = fspecial('average',[5,5])
mask = fspecial('gaussian',[5,5])
out = imfilter(img,mask);
figure, imshow(out,[])
mask = fspecial('gaussian',[15,15]);
out = imfilter(img,mask);
figure, imshow(out,[])
mask = fspecial('gaussian',[105,105]);
out = imfilter(img,mask);
figure, imshow(out,[])
fspecial('gaussian',[105,105])
fspecial('average',[105,105])
clc
help \fspecial
help fspecial
H = fspecial('motion')
H = fspecial('motion',5,1)
mask = [1 4 1; 1 8 1; 1 4 1]
mask = [1 4 1; 1 8 1; 1 4 1]/sum(mask(:))
mask = [1 4 1; 4 8 4; 1 4 1]
mask = [1 4 1; 1 8 1; 1 4 1]/sum(mask(:))
mask = rand(5,5)
close all
clc
figure, imshow(img)
J = imnoise(I,'salt & pepper', 0.02);
out = imnoise(img,'salt & pepper', 0.02);
figure, imshow(out)
mask
mask = ones(5,5)/25;
avg= imfilter(img,mask);
med = medfilt2(img,[5,5]);
figure,
subplot(2,2,1),imshow(img),title('original')
subplot(2,2,2),imshow(avg),title('Average Filter')
subplot(2,2,3),imshow(med),title('Median Filter')
avg= imfilter(out,mask);
med = medfilt2(out,[5,5]);
subplot(2,2,1),imshow(out),title('original')
subplot(2,2,2),imshow(avg),title('Average Filter')
subplot(2,2,3),imshow(med),title('Median Filter')
clc
help ordfilt2
clc
close all
figure, imshow(img)
figure, imshow(out)
B=ordfilt2(out,1,ones(3,3));
imgMin=ordfilt2(out,1,ones(3,3));
imgMax=ordfilt2(out,9,ones(3,3));
imgMed=ordfilt2(out,5,ones(3,3));
figure,
subplot(2,2,1),imshow(out),title('original')
subplot(2,2,2),imshow(imgMin),title('min')
subplot(2,2,3),imshow(imgMax),title('max')
subplot(2,2,4),imshow(imgMed),title('med')
clc
mask = fspecial('laplacian',0);
mask
mask = fspecial('laplacian',0.5);
mask
mask = fspecial('laplacian',0);
mask
figure, imshow(img)
out =  imfilter(img,mask);
figure, imshow(out)
min(out(:))
max(out(:))
figure, imshow(out>100)
figure, imshow(out>0)
figure, imshow(out>50)
close all
clc
mask = [-1 -2 -1; 0 0 0; 1 2 1]
out = imfilter(img,mask);
figure, imshow(img)
figure, imshow(out)
mask = mask';
mask
out1 = imfilter(img,mask);
figure, imshow(out1)
img1 =  imread('c:\d\img\objects.jpg');
obj = rgb2gray(img1);
figure, imshow(obj)
mask
out2 = imfilter(obj,mask);
out3 = imfilter(obj,mask');
figure, imshow(out2)
figure, imshow(out3)
mask'
mask =-mask'
out4 = imfilter(obj,mask);
figure, imshow(out4)
max(out4(:))
min(out4(:))
out4 = imfilter(double(obj),mask);
figure, imshow(out4)
figure, imshow(out4,[])
figure, imshow(abs(out4),[])
out5 = imfilter(double(obj),mask');
figure, imshow(abs(out5),[])
out6 = out5||out4;
out6 = out5|out4;
figure, imshow(abs(out6),[])
figure, imshow(abs(out4+out5),[])
figure, imshow(abs(out4+out5)>50,[])
figure, imshow(abs(out4+out5)>40,[])
close all
figure, imshow(abs(out4+out5)>0,[])
figure, imshow(abs(out4+out5)>10,[])
figure, imshow(abs(out4+out5)>5,[])
figure, imshow(abs(out4+out5),[])
figure, imshow(abs(out4+out5)>0,[])
figure, imshow(abs(out4+out5)>50,[])
clc
I = imread('circuit.tif');
BW1 = edge(I,'prewitt');
BW2 = edge(I,'canny');
figure, imshow(BW1)
figure, imshow(I)
fspecial('prewitt')
figure, imshow(BW2)
figure, imshow(edge('sobel'))
figure, imshow(edge(I,'sobel'))
close all
clear
clc
img = imread('c:\d\img\coins.jpg');
figure, imshow(img)
kernel = ones(5,5)/25;
blur = imfilter(img,kernel);
mask = img - blur;
figure, imshow(mask)
out = img + mask;
figure, imshow(out,[])
kernel = ones(9,9)/81;
blur = imfilter(img,kernel);
mask = img - blur;
out = img + mask;
close all
figure, imshow(img,[])
figure, imshow(out,[])
out = img + 7*mask;
figure, imshow(out,[])
out = img + 2*mask;
figure, imshow(out,[])
out = img + mask;
figure, imshow(out,[])
figure, imshow(mask,[])