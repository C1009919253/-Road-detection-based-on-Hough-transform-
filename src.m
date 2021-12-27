clear all;
clc;
close all;
RGB = imread('13.jpeg');
%% 转为灰度图
ori_img = 0.5*RGB(:,:,1)+0.5*RGB(:,:,2);
%% 高斯滤波
FFT_img = fft2(ori_img);
FFT_img = fftshift(FFT_img);
[M, N] = size(ori_img);
m = floor(M/2);
n = floor(N/2);
d0 = 50;
for i = 1:M
    for j = 1:N
        D(i,j) = ((i-m).^2+(j-n).^2);
        H(i,j) = exp(-D(i,j)./(2*d0^2));
    end
end
IFFT_img = (H.*FFT_img);
IFFT_img = ifftshift(IFFT_img);
IFFT_img = ifft2(IFFT_img);
cha_img = real(IFFT_img);
cha_img = uint8(cha_img);
figure;
subplot(231);
imshow(ori_img);
title('原图像');
subplot(232);
imshow(cha_img);
title('高斯滤波');
%% OTSU算法二分
p = [];
for i=1:256
    p(i) = 0;
end
OTSU = cha_img;
for i=1:M
    for j=1:N
        k = cha_img(i, j)+1;
        p(k) = p(k) + 1;
    end
end
p = p / (M * N);
deta = [];
mG = 0;
for i=0:255
    mG = mG + i*p(i+1);
end
for k=0:255
    m = 0;
    p1 = 0;
    for i=0:k
        m = m + i * p(i+1);
        p1 = p1 + p(i+1);
    end
    deta(k+1) = (mG * p1 - m)^2 / p1 / (1 - p1);
end
[Max, k] = max(deta);
k = k-1;
for i=1:M
    for j=1:N
        if(cha_img(i, j) < 1.15*k)
            OTSU(i, j) = 0;
        else
            OTSU(i, j) = 255;
        end
    end
end
subplot(233);
imshow(OTSU);
title('OTSU二分');
%% 去除无关区域
L = bwlabel(OTSU,8);
area = regionprops(L, 'Area');
perimeter = regionprops(L, 'Perimeter');
boundingbox = regionprops(L, 'BoundingBox');
n = size(area);
kill = [];
ki = 0;
boundarea = [];
for i=1:n
    if area(i).Area < 30
        ki = ki + 1;
        kill(ki) = i;
    else
        if perimeter(i).Perimeter < 10
            ki = ki + 1;
            kill(ki) = i;
        else
            xy = boundingbox(i).BoundingBox;
            if xy(2) < N * 0.15
                ki = ki + 1;
                kill(ki) = i;
            end
        end
    end
    xy = boundingbox(i).BoundingBox;
    boundarea(i) = xy(3) * xy(4);
end
[Max, label] = max(boundarea);
taba = ismember(L, kill);
OTSU = OTSU .* uint8(1 - taba);
subplot(234);
imshow(OTSU);
title('去除冗杂');
%% 边缘检测
Canny(2:M+1,2:N+1) = OTSU;
Canny(1, :) = 0;
Canny(:, 1) = 0;
Canny(M+2, :) = 0;
Canny(:, N+2) = 0;
canny = OTSU;
can = [0 1 0; 1 -4 1; 0 1 0];
Canny = double(Canny);
for i = 1:M
    for j = 1:N
        canny(i, j) = sum(sum(can .* Canny(i:i+2, j:j+2)));
    end
end
subplot(235);
imshow(canny);
title('边缘检测');
%% 
[H,theta,rho] = hough(canny);
P  = houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));
lines = houghlines(canny,theta,rho,P);
subplot(236);
imshow(canny);
hold on
max_len = 0;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

   % Plot beginnings and ends of lines
   plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
   plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

   % Determine the endpoints of the longest line segment
   len = norm(lines(k).point1 - lines(k).point2);
   if ( len > max_len)
      max_len = len;
      xy_long = xy;
   end
end
title('霍夫变换');