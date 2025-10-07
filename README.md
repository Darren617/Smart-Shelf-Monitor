# Smart-Shelf-Monitor
基于yolov8的摄像头实时检测物体的项目

## 环境配置&运行说明

1.利用`conda`创造虚拟环境（`ultralytics`）

```python
conda create -n ultralytics python=3.9
```

2.激活`ultralytics`环境

```python
conda activate ultralytics
```

3.安装(pytorch)[https://pytorch.org/]

```python
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

4.安装`ultralytics`

```python
https://github.com/ultralytics/ultralytics
pip install -e .
```



## 实现步骤

1.使用cv2调取摄像头，设置判断条件避免程序崩溃

2.取出yolov8的`names`、`boxes`

3.使用for循环将xyxy、conf、class_id实时迭代，实现实时定位预测



## 思考题

**1.如果画面中同时出现多个水杯，你的程序会如何处理？**（身边没有水杯，故只能用瓶子（bottle）替代）

>  对瓶子（bottle）进行逐个编号与计数并打印中心点坐标。

​	解决方案：

​	加一个`model.track()`跟踪，然后标记ID，这样就可以区分不同的瓶子



**2.如果需要获取物体在三维空间中的方向，仅靠当前信息足够吗？为什么？**

​	不够。

​	YOLOv8 的检测结果只有：

- ​	二维图像坐标（x1, y1, x2, y2）
  - 类别标签（例如 cup, laptop）
  - 置信度（conf）

​	想要确定物体的方向/朝向，必须知道摄像头的参数，物体在3维世界中的几何特征点等等，可	通过PnP解算来确定。



