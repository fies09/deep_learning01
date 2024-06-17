#!/usr/bin/env python
# -*- coding = utf-8 -*-
# @Time       : 2023/10/13 15:23
# @Author     : fany
# @Project    : PyCharm
# @File       : imgSplice2.py
# @description: 行/列拼接
import cv2
import numpy as np

# 读取多个图像
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')

# 确保图像尺寸相同
image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

# 按行拼接图像
merged_image_row = np.vstack((image1, image2_resized))

# 按列拼接图像
merged_image_col = np.hstack((image1, image2_resized))

# 保存拼接后的图像
cv2.imwrite('merged_image_row.jpg', merged_image_row)
cv2.imwrite('merged_image_col.jpg', merged_image_col)

