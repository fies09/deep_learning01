#!/usr/bin/env python
# -*- coding = utf-8 -*-
# @Time       : 2023/10/13 15:21
# @Author     : fany
# @Project    : PyCharm
# @File       : imgSegmentation2.py
# @description: 使用scikit-image进行图像分割
from skimage import io, color, segmentation
import matplotlib.pyplot as plt

# 读取图像
image = io.imread('input_image.jpg')

# 将图像转换为灰度图
gray_image = color.rgb2gray(image)

# 使用Felzenszwalb分割算法进行图像分割
segments = segmentation.felzenszwalb(image, scale=100, sigma=0.5, min_size=50)

# 显示分割后的图像
plt.imshow(segments)
plt.title('Segmented Image')
plt.show()
