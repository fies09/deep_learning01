#!/usr/bin/env python
# -*- coding = utf-8 -*-
# @Time       : 2023/10/13 15:25
# @Author     : fany
# @Project    : PyCharm
# @File       : faceLocation.py
# @description: faceLocation
'''
在这个示例中，我们首先加载dlib的预训练模型，然后使用dlib的人脸检测器检测人脸，接着获取关键点并绘制到图像上。
确保你有shape_predictor_68_face_landmarks.dat模型文件。你可以在dlib的官方网站下载这个模型文件。
根据你的需求，你可能需要调整代码以适应不同的模型或输入图像。
'''
import dlib
import cv2

# 加载dlib的预训练模型
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 初始化dlib的人脸检测器
detector = dlib.get_frontal_face_detector()

# 读取图像
image = cv2.imread("face_image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用人脸检测器检测人脸
faces = detector(gray)

# 遍历每张脸，获取关键点并绘制
for face in faces:
    landmarks = predictor(image, face)

    # 绘制关键点
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

# 显示图像
cv2.imshow("Facial Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
