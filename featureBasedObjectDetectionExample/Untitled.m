run('Q:/图像分割/用于目标检测/vlfeat-0.9.20-bin/vlfeat-0.9.20/toolbox/vl_setup.m');
I1=imread('airplane/airplane45.tif');
image(I1);
I2=imread('airplane/airplane01.tif');
I1= im2single(I1) ;
I2= im2single(I2) ;
if size(I1,3) > 1, im1g = rgb2gray(I1) ; else im1g = I1; end
if size(I2,3) > 1, im2g = rgb2gray(I2) ; else im2g = I2 ; end

    [f1,d1] = vl_sift(im1g) ;
    [f2,d2] = vl_sift(im2g) ;
    [matches, scores] = vl_ubcmatch(d1,d2) ;
perm = randperm(size(f1,2)) ;  
sel = perm(1:50) ;  
h1 = vl_plotframe(f1(:,sel)) ;  
h2 = vl_plotframe(f1(:,sel)) ;  
set(h1,'color','k','linewidth',3) ;  
set(h2,'color','y','linewidth',2) ; 
