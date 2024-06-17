#!/usr/bin/env python
# -*- coding = utf-8 -*-
# @Time       : 2023/9/3 21:44
# @Author     : fany
# @Project    : PyCharm
# @File       : retinaNet.py
# @description: retinaNet
import torch
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.transforms import functional as F

# 加载预训练的 RetinaNet 模型
model = retinanet_resnet50_fpn(weights='coco')
model.eval()

import cv2

# 打开默认摄像头（通常是摄像头0）
cap = cv2.VideoCapture(0)

# 设置摄像头参数，例如分辨率
cap.set(3, 1280)  # 设置宽度
cap.set(4, 720)   # 设置高度

while True:
    ret, frame = cap.read()  # 读取一帧
    if not ret:
        break  # 如果未成功读取帧，退出循环

    # 预处理图像
    image = F.to_tensor(frame).unsqueeze(0)  # 转换为 PyTorch 张量并添加批次维度
    image /= 255.0  # 归一化像素值到 [0, 1]

    # 目标检测
    with torch.no_grad():
        predictions = model(image)

    # 处理检测结果并绘制框
    # 此处需要解析 predictions 并在图像上绘制检测框和类别标签

    # 显示带有检测结果的图像
    cv2.imshow('RetinaNet Object Detection', frame)

    # 通过按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


