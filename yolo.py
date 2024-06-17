#!/usr/bin/env python
# -*- coding = utf-8 -*-
# @Time       : 2023/8/23 00:35
# @Author     : fany
# @Project    : PyCharm
# @File       : yolo.py
# @description:
from PIL import Image
from ultralytics import YOLO

# 加载预训练的YOLOv8n模型
model = YOLO('yolov8n.pt')

# 在'bus.jpg'上运行推理
results = model('bus.jpeg')  # 结果列表

# 展示结果
for r in results:
    im_array = r.plot()  # 绘制包含预测结果的BGR numpy数组
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL图像
    im.show()  # 显示图像
    im.save('results.jpg')  # 保存图像
