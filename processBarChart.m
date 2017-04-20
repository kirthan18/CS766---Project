I = imread('bar2.png');
txt = ocr(I);
BW = im2bw(I, graythresh(I));
ed = edge(BW, 'canny');
imshow(ed);
stats = regionprops(ed, 'all');
fh = figure();
imshow(I);
hold on;

numbars = 0;
for n = 1 : length(stats)    
     thisBB = stats(n).BoundingBox; 
     
     area = stats(n).Area;
     if area > 150
         numbars = numbars + 1;
%         plot(thisBB(1), thisBB(2),'p','Color','green');
%         plot(thisBB(1) + thisBB(3), thisBB(2),'p','Color','red');
%         plot(thisBB(1), thisBB(2) + thisBB(4),'p','Color','cyan');
%         plot(thisBB(1) + thisBB(3), thisBB(2) + thisBB(4),'p','Color','yellow');
%      
%         plot(thisBB(1), size(I,1),'p','Color','magenta');
%         
%         roi = [thisBB(1), thisBB(2) + thisBB(4), thisBB(3), (size(I,1) - thisBB(2) + thisBB(4))]; 
        roi = [thisBB(1), thisBB(2) + thisBB(4), thisBB(3), size(I,1) - (thisBB(2)+ thisBB(4))]; 
        rectangle('Position', [thisBB(1),thisBB(2),thisBB(3),thisBB(4)],...    
        'EdgeColor','r','LineWidth',1, 'Facecolor', 'b')    
        
        roi2 = [5, thisBB(2) - 20,60,60]; 
        rectangle('Position', roi,...    
        'EdgeColor','y','LineWidth',1, 'Facecolor', 'r')
        rectangle('Position', roi2,...    
        'EdgeColor','y','LineWidth',1, 'Facecolor', 'r')    
     end
end    
disp(['Number of bars : ', num2str(numbars)]);
figure; imshow(I);
for n = 1 : length(stats)    
     thisBB = stats(n).BoundingBox; 
     area = stats(n).Area;
     if area > 150
        roi = [thisBB(1), thisBB(2) + thisBB(4), thisBB(3), size(I,1) - (thisBB(2)+ thisBB(4))]; 
        roi2 = [5, thisBB(2) - 20,60,60];
        %roi = round(getPosition(imrect));
        x_axis_label = ocr(I, roi);
        y_axis_label = ocr(I, roi2);
        disp(['(', x_axis_label.Text, ',' , y_axis_label.Text, ')'])
     end
end   


