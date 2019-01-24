# Mask-YOLO: A Multi-task Learning Architecture for Object Detection and Instance Segmentation

- This work combined the one-stage detection pipeline, YOLOv2 with the idea of two-branch architecture from Mask R-CNN. Due to the hardware limitation, I only implemented it on MobileNet backbone with depthwise separable blocks, though it has the potential to be implemented with deeper network, e.g. ResNet-50 or ResNet-101 with FPN (Feature Pyramid Networks).
- The overall architecture can be visualized like this: 

<img src="https://github.com/jianing-sun/Mask-YOLO/blob/master/asset/mask_yolo.png" />

- Some training results:

<!-- <div align="center">
  <img src="https://github.com/jianing-sun/Mask-YOLO/blob/master/asset/InferMaskYOLO-Shapes-Dec-28-16-29.png">                                                                                                             <img src="https://github.com/jianing-sun/Mask-YOLO/blob/master/asset/InferMaskYOLO-Shapes-Dec-28-16-31.png">

<img src="https://github.com/jianing-sun/Mask-YOLO/blob/master/asset/InferMaskYOLO-Shapes-Dec-28-16-32.png" >

<img src="https://github.com/jianing-sun/Mask-YOLO/blob/master/asset/InferMaskYOLO-Shapes-Dec-28-16-33.png" >

</div> -->