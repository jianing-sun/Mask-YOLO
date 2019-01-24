# Mask-YOLO: A Multi-task Learning Architecture for Object Detection and Instance Segmentation

- This work combined the one-stage detection pipeline, YOLOv2 with the idea of two-branch architecture from Mask R-CNN. Due to the hardware limitation, I only implemented it on MobileNet backbone with depthwise separable blocks, though it has the potential to be implemented with deeper network, e.g. ResNet-50 or ResNet-101 with FPN (Feature Pyramid Networks).
- The overall architecture can be visualized like this: 

<img src="https://github.com/jianing-sun/Mask-YOLO/blob/master/asset/mask_yolo.png" />

- Training results on Shapes dataset:

<table sytle="border: 0px;">
<tr>

<td><img src="https://github.com/jianing-sun/Mask-YOLO/blob/master/asset/InferMaskYOLO-Shapes-Dec-28-16-29.png" ></td>

<td><img src="https://github.com/jianing-sun/Mask-YOLO/blob/master/asset/InferMaskYOLO-Shapes-Dec-28-16-31.png" ></td>

<td><img src="https://github.com/jianing-sun/Mask-YOLO/blob/master/asset/InferMaskYOLO-Shapes-Dec-28-16-32.png" ></td>
<td><img src="https://github.com/jianing-sun/Mask-YOLO/blob/master/asset/InferMaskYOLO-Shapes-Dec-28-16-33.png" ></td>
</tr>
</table>

- Training results on Rice:

<img src="https://github.com/jianing-sun/Mask-YOLO/blob/master/asset/InferMaskYOLO-Rice-Jan-02-15-32" widt="500px"/>

- Training results on Generic Food:

<img src="https://github.com/jianing-sun/Mask-YOLO/blob/master/asset/InferMaskYOLO-Food-Jan-06-19-45" widt="500px"/>

