I1 = imread('Bar4.jpg');
I = rgb2gray(I1);
BW = imbinarize(I);
BW = imcomplement(BW);
a = bwareaopen(BW, 100);
ed = edge(a, 'canny', 0.2);
% imshow(ed);
stats = regionprops(ed, 'all');
fh = figure();
imshow(I);
hold on;

numbars = 0;
for n = 1 : length(stats)    
     thisBB = stats(n).BoundingBox; 
     
     area = stats(n).Area;
     if area > 1
         numbars = numbars + 1;
        rectangle('Position', [thisBB(1),thisBB(2),thisBB(3),thisBB(4)],...    
        'EdgeColor','r','LineWidth',1, 'Facecolor', 'b')    
     end
end 

for n = 1 : length(stats)/2    
     thisBB = stats(n).BoundingBox; 
     
     bar1BB = stats(2*n - 1).BoundingBox;
     bar2BB = stats(2*n).BoundingBox;
     
     x = bar1BB(1);
     width = bar2BB(3) + bar1BB(3);
     y = bar1BB(2) + bar1BB(4);
     height = size(I,1) - (bar1BB(2)+ bar1BB(4)) - (.2 * size(I,1));
%      plot(x, y,'p','Color','green');
%      plot(x, y + height ,'p','Color','blue');

     rectangle('Position', [x y width height],...    
     'EdgeColor','b','LineWidth',1, 'Facecolor', 'r')   
 
     x_axis_label = ocr(I, [x y width height]);  
     display(x_axis_label.Text);
end 

for n = 1 : length(stats)  
     thisBB = stats(n).BoundingBox;
     roi2 = [25, thisBB(2),25,25]; 
     rectangle('Position', roi2,...    
     'EdgeColor','g','LineWidth',1, 'Facecolor', 'y') 
     y_axis_label = ocr(I, roi2);
     display(y_axis_label.Text);
end     
disp(['Number of bars : ', num2str(numbars)]);

