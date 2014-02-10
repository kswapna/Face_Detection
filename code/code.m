
clc;
clear all;
close all;
I=imread('grp.jpg'); 
rgbInputImage = I;
II=I;
%{
close all;
clear all;
clc;
rgbInputImage = imread('s1');
%rgbInputImage=getsnapshot(rgbInputImage);
labInputImage = applycform(rgbInputImage,makecform('srgb2lab'));
Lbpdfhe = fcnBPDFHE(labInputImage(:,:,1));
labOutputImage = cat(3,Lbpdfhe,labInputImage(:,:,2),labInputImage(:,:,3));
rgbOutputImage = applycform(labOutputImage,makecform('lab2srgb'));
figure, imshow(rgbInputImage);
figure, imshow(rgbOutputImage);
img=rgbOutputImage;
final_image = zeros(size(img,1), size(img,2));
%}




%{
labInputImage = applycform(rgbInputImage,makecform('srgb2lab'));
Lbpdfhe = fcnBPDFHE(labInputImage(:,:,1));
labOutputImage = cat(3,Lbpdfhe,labInputImage(:,:,2),labInputImage(:,:,3));
rgbOutputImage = applycform(labOutputImage,makecform('lab2srgb'));
figure, imshow(rgbInputImage);
figure, imshow(rgbOutputImage);%%

I=rgbOutputImage;
%}




segment = zeros(size(I,1), size(I,2));
figure(1)
imshow(I)
I=double(I); 
R=I(:,:,1); 
G=I(:,:,2);
B=I(:,:,3);
cb = 0.148* I(:,:,1) - 0.291* I(:,:,2) + 0.439 * I(:,:,3) + 128; %calculating cb value using formula
cr = 0.439 * I(:,:,1) - 0.368 * I(:,:,2) -0.071 * I(:,:,3) + 128; 
[w h]= size(I(:,:,1));
for i=1:w 
    for j=1:h
        if 140<=cr(i,j) && cr(i,j)<=165 && 140<=cb(i,j) && cb(i,j)<=195
            segment(i,j)=1;
        else
            segment(i,j)=0;
        end
    end
end
figure(2)
imshow(segment)


%{
if(size(img, 3) > 1)
for i = 1:size(img,1)
for j = 1:size(img,2)
R = img(i,j,1);
G = img(i,j,2);
B = img(i,j,3);
if(R > 92 && G > 40 && B > 20)
v = [R,G,B];
if((max(v) - min(v)) > 15)
if(abs(R-G) > 15 && R > G && R > B)
final_image(i,j) = 1;
end
end
end
end
end
end
figure(3);
subplot(1,2,1);
imshow(final_image);  
%}



%Grayscale To Binary.

binaryImage=im2bw(segment,0.6);
figure, imshow(binaryImage);

%binary=im2bw(II,0.6);

filledBW = imfill(binaryImage,'holes');


figure, imshow(filledBW);

U=binaryImage;
se2 = strel('disk',1); %create structuring element
erodedBW = imerode(filledBW,se2);
figure, imshow(erodedBW);

se1 = strel('disk',1);
dilateBW=imdilate(erodedBW,se1);
figure, imshow(dilateBW);

dilateBW = immultiply(dilateBW,binaryImage);
figure, imshow(dilateBW);

%{
se = strel('disk',1);        
erodedBW = imerode(binaryImage,se);
 
figure, imshow(erodedBW);

%se1 = strel('ball',5,5);

%Dilate a grayscale image with a rolling ball structuring element.

I2 = imdilate(erodedBW,se);
figure, imshow(I2);


I3=imfill(I2, 'holes');
figure, imshow(I3);
%}
binaryImage=dilateBW;
new = ~bwareaopen(~binaryImage,5);
binaryImage=new;
%{
J = edge(binary,'sobel');
%J=imcomplement(J);
segment=imcomplement(segment);
figure(5);
imshow(J);
K=J+segment;
K=imcomplement(K);
figure(6);
imshow(K);
binaryImage=K;

%}
%Filling The Holes.
%binaryImage = imfill(binaryImage, 'holes');
%figure, imshow(binaryImage);



binaryImage = bwareaopen(binaryImage,2300);   
figure,imshow(binaryImage);

labeledImage = bwlabel(binaryImage, 8);

%{
e = regionprops(labeledImage,'EulerNumber');
eulers=cat(1,e.EulerNumber);
figure, imshow(eulers);
major = regionprops(binaryImage,'MajorAxisLength');

major_length=cat(1,major.MajorAxisLength);

minor = regionprops(binaryImage,'MinorAxisLength');

minor_length=cat(1,minor.MinorAxisLength);

ratiolist=major_length./minor_length; %compute aspect ratio

%And, any region with aspect ratio unlikely to be a face is rejected. Here, we need to set acceptable aspect rat
%io range by considering the fact that face and neck may be connected and region aspect ratio will be quite high. I set this range by experiment.

region_index = find(aspect_ratio<=3.5 & aspect_ratio>=1);

figure, imshow(region_index);
%}
blobMeasurements = regionprops(labeledImage, segment, 'all');
numberOfPeople = size(blobMeasurements, 1)
imagesc(II); title('Outlines, from bwboundaries()');
labeledImage11 = bwlabel(U, 8);
blobMeasurements11 = regionprops(labeledImage11, segment, 'all');
%axis square;
hold on;
%{
for k = 1:numberOfPeople
  w = blobMeasurements(k).BoundingBox(3)
  y=blobMeasurements(k).BoundingBox(4)
%}





%{
boundaries = bwboundaries(binaryImage);
for k = 1 : numberOfPeople
thisBoundary = boundaries{k};
plot(thisBoundary(:,2), thisBoundary(:,1), 'g', 'LineWidth', 2);
end
% hold off;
%}

imagesc(II);
hold on;
title('Original with bounding boxes');
%fprintf(1,'Blob # x1 x2 y1 y2\n');

%widths=blobMeasurements.BoundingBox(3);
%heights=blobMeasurements.BoundingBox(4)
%hByW=heights./widths;

major_length=cat(1,blobMeasurements.MajorAxisLength);
minor_length=cat(1,blobMeasurements.MinorAxisLength);
ratiolist=major_length./minor_length; %compute aspect ratio

for k = 1 : numberOfPeople % Loop through all blobs.
w = blobMeasurements(k).BoundingBox(3)
h =blobMeasurements(k).BoundingBox(4)
e = blobMeasurements(k).EulerNumber
area = blobMeasurements(k).Area


thisBlobsBox = blobMeasurements(k).BoundingBox; % Get list of pixels in current blob.
%{
hByW=h/w;
if (hByW>1.75 || hByW<0.75)
        % this cannot be a mouth region. discard
        continue;
    end
    
    % implemented by me: Impose a min face dimension constraint
  %  if (heights(k)<20 && widths(k)<20)
        if (h<20 && w<20)
        continue;
    end
%}
 if(1-e>=2)
     if(ratiolist(k)<=3.5 && ratiolist(k)>=1)
         blobMeasurements(k).Eccentricity;
     if(blobMeasurements(k).Eccentricity <.39905)
        
         areab=thisBlobsBox(3)*thisBlobsBox(4)
         if(area>(areab/3))
         %.89905

            x1 = thisBlobsBox(1);
            y1 = thisBlobsBox(2);
            x2 = x1 + thisBlobsBox(3);
            y2 = y1 + thisBlobsBox(4);


            % fprintf(1,'#%d %.1f %.1f %.1f %.1f\n', k, x1, x2, y1, y2);
            x = [x1 x2 x2 x1 x1];
            y = [y1 y1 y2 y2 y1];
            %subplot(3,4,2);
            plot(x, y, 'LineWidth', 2);
         end
     end
   end
 end
 
end