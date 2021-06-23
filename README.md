image_stitch


基于视频流的图片拼接，用的光流法，根据两帧间的差距不大关系，用光流法找到y方向的偏移，从下一帧抠出偏移量的像素值，在往全图上拼接就可以了；

注意：1.目前只能y方向，不支持多方向，可以修改成x或y方向，但不能支持两个方向；
2.视频的移动速度不能太快；

效果图：


![jigui_1_optical_flow_result](https://user-images.githubusercontent.com/49335389/123021987-a3a08e00-d407-11eb-9063-87a3af80c3e2.png)
