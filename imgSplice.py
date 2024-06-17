#!/usr/bin/env python
# -*- coding = utf-8 -*-
# @Time       : 2023/10/13 15:22
# @Author     : fany
# @Project    : PyCharm
# @File       : imgSplice.py
# @description: 水平拼接
import cv2
import numpy as np

# 读取多个图像
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')

# 确保图像尺寸相同
image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

# 水平拼接图像
merged_image = np.hstack((image1, image2_resized))

# 保存拼接后的图像
cv2.imwrite('merged_image.jpg', merged_image)
