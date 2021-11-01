# lineSLAM
Source code of line-based SLAM project using image processing with optimization for indoor environment
Running on Raspberry Pi 3, on top of simple 2-wheel robot

Main idea:
1. Using lines to model floor border
2. Perform floor scoring segmentation based on Yinxiao Li algorithm [1]
3. Calculate Extented Kalman filter for SLAM

Demo: https://www.youtube.com/watch?v=FkEdK9wP5Ro

![SLAM](https://user-images.githubusercontent.com/36019046/132967966-24214d9d-1c3a-4c1d-8706-b4106e0c1786.png)
![IMG_2191](https://user-images.githubusercontent.com/36019046/132968039-76124330-e8d6-4904-bc46-6f2232e54c9e.JPG)

[1] Y. Li và S. T. Birchfield, “Image-Based Segmentation of Indoor Corridor Floors for a Mobile Robot,” 2011. 
