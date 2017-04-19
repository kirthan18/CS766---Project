I = imread('bar1.jpg');
txt = ocr(I);
BW = im2bw(I, graythresh(I));
ed = edge(BW, 'canny');
imshow(ed);
stats = regionprops(ed, 'all');
fh = figure();
imshow(I);
hold on;
for n = 1 : length(stats)    
     thisBB = stats(n).BoundingBox; 
     
     area = stats(n).Area;
     if area > 100
        plot(thisBB(1), thisBB(2),'p','Color','green');
        plot(thisBB(1) + thisBB(3), thisBB(2),'p','Color','red');
        plot(thisBB(1), thisBB(2) + thisBB(4),'p','Color','cyan');
        plot(thisBB(1) + thisBB(3), thisBB(2) + thisBB(4),'p','Color','yellow');
        
        plot(thisBB(1) - 10, thisBB(2) + thisBB(4),'p','Color','cyan');
        plot(thisBB(1) + thisBB(3), thisBB(2) + thisBB(4),'p','Color','yellow');
        
        plot(thisBB(1) - 10, thisBB(2) + thisBB(4) + 10,'p','Color','blue');
        plot(thisBB(1) + thisBB(3) + 10, thisBB(2) + thisBB(4) + 10 ,'p','Color','magenta');
        
        roi = [thisBB(1) - 5 ,thisBB(2) + thisBB(4) ,size(I,2) - thisBB(1),thisBB(4) + 5];
%         rectangle('Position', [thisBB(1) - 5 ,thisBB(2) + thisBB(4) ,size(I,2) - thisBB(1),thisBB(4) + 5],...  
%             'EdgeColor','r','LineWidth',1, 'Facecolor', 'b')    
        
%         'EdgeColor','r','LineWidth',1, 'Facecolor', 'b')    
%         plot(thisBB(1)+thisBB(3), thisBB(2),'p','Color','green')
        rectangle('Position', [thisBB(1),thisBB(2),thisBB(3),thisBB(4)],...    
        'EdgeColor','r','LineWidth',1, 'Facecolor', 'b')    
     end
end    

figure; imshow(I);
for n = 1 : length(stats)    
     thisBB = stats(n).BoundingBox; 
     area = stats(n).Area;
     if area > 100
%         roi = [thisBB(1) - 5 ,thisBB(2) + thisBB(4) ,size(I,2) - thisBB(1),thisBB(4) + 5];
         roi = round(getPosition(imrect));
        txt = ocr(I, roi);
        txt.Text
     end
end   
% fh = figure();
% imshow(I);
% [x y] = ginput(4);
% x
% y
% [B,L,N] = bwboundaries(BW);
% imshow(BW); hold on;
% colors=['b' 'g' 'r' 'c' 'm' 'y'];
% for k=1:length(B),
%   boundary = B{k};
%   cidx = mod(k,length(colors))+1;
%   plot(boundary(:,2), boundary(:,1),...
%        colors(cidx),'LineWidth',2);
% 
%   %randomize text position for better visibility
%   rndRow = ceil(length(boundary)/(mod(rand*k,7)+1));
%   col = boundary(rndRow,2); row = boundary(rndRow,1);
%   h = text(col+1, row-1, num2str(L(row,col)));
%   set(h,'Color',colors(cidx),'FontSize',14,'FontWeight','bold');
% end

% figure; imshow(BW); hold on;
% for k=1:length(B),
%     boundary = B{k};
%     if(k > N)
%         plot(boundary(:,2),...
%             boundary(:,1),'g','LineWidth',2);
%     else
%         plot(boundary(:,2),...
%             boundary(:,1),'r','LineWidth',2);
%     end
% end
% cc=bwconncomp(BW);    
% stats=regionprops(cc,'BoundingBox');    
% for n = 1 : length(stats)    
%      thisBB = stats(n).BoundingBox;    
%      rectangle('Position', [thisBB(1),thisBB(2),thisBB(3),thisBB(4)],...    
%      'EdgeColor','r','LineWidth',1 )    
% end    
%  
% [L,num] = bwlabel(BW);
% num
% M = im2uint8(L/num);
% imwrite(M,jet,'label.jpg');
