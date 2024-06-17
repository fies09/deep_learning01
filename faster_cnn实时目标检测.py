#!/usr/bin/env python
# -*- coding = utf-8 -*-
# @Time       : 2023/8/19 21:41
# @Author     : fany
# @Project    : PyCharm
# @File       : faster_cnn实时目标检测.py
# @description:
import torch
import torchvision
import cv2
from configs import LABELS
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# # 加载预训练的 Faster R-CNN 模型q
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

model.eval()

# 打开摄像头并实时检测
cap = cv2.VideoCapture(0)  # 打开摄像头，0表示默认摄像头，如果有多个摄像头可以尝试不同的索引
saved = False  # 是否已保存新目标图像
counter = 0  # 保存的图像计数器

while True:
    # 读取摄像头帧
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为 PyTorch 的 Tensor 格式
    img = torch.from_numpy(frame / 255.0).permute(2, 0, 1).float().unsqueeze(0)

    # 使用Faster R-CNN模型进行推理
    with torch.no_grad():
        predictions = model(img)

    # 解析预测结果
    boxes = predictions[0]['boxes'].detach().numpy()
    labels = predictions[0]['labels'].detach().numpy()

    # 显示检测结果
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, LABELS[label], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        if not saved:
            # 保存带有目标框选的图像，并重新命名新的文件
            counter += 1
            filename = f'detected_image_{counter}.jpg'
            cv2.imwrite(filename, frame)
            saved = True

    # 显示帧
    cv2.imshow('Object Detection', frame)

    # 按下'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 当有新目标出现时，重置保存状态
    if len(boxes) > 0:
        saved = False

cap.release()  # 释放摄像头资源
cv2.destroyAllWindows()  # 关闭窗口
