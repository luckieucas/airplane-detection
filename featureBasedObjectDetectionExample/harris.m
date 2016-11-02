boxImage = imread('saveImage2.jpg');
boxImage = imread('589A.jpg');
boxImage= im2single(boxImage) ;
if size(boxImage,3) > 1, boxImage = rgb2gray(boxImage) ; else boxImage = boxImage; end
  corners = detectSURFFeatures(boxImage);
  [features, valid_corners] = extractFeatures(boxImage, corners);
  airplanepoints=selectStrongest(valid_corners, 300);
  A=airplanepoints.Location;
      opts = statset('Display','final');
[idx,C] =  kmedoids(A,35,'Distance','cityblock',...
    'Replicates',5,'Options',opts);
    figure;
       imshow(boxImage);
 hold on
    plot(A(:,1),A(:,2),'.');
    plot(C(:,1),C(:,2),'kx',...
     'MarkerSize',15,'LineWidth',3,'MarkerEdgeColor','b')
%     opts = statset('Display','final');
% [idx,C] = kmeans(A,30,'Distance','cityblock',...
%     'Replicates',5,'Options',opts);
% figure;
% plot(A(idx==1,1),A(idx==1,2),'r.','MarkerSize',12)
hold on
% plot(A(idx==2,1),A(idx==2,2),'b.','MarkerSize',12)
% plot(C(:,1),C(:,2),'kx',...
%      'MarkerSize',15,'LineWidth',3)
% legend('Cluster 1','Cluster 2','Centroids',...
%        'Location','NW')
title 'Cluster Assignments and Centroids'
hold off
%     imshow(boxImage);
%     hold on
%     
%      plot(selectStrongest(valid_corners, 200));