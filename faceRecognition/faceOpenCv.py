#!/usr/bin/env python
# -*- coding = utf-8 -*-
# @Time       : 2023/10/8 14:37
# @Author     : fany
# @Project    : PyCharm
# @File       : faceOpenCv.py
# @description: faceOpenCv
import cv2

# 加载Haar级联分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 加载图像
img = cv2.imread('../img/Martha_Bowen_0002.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 检测人脸
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 在图像上绘制人脸框
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# 保存处理后的图像
cv2.imwrite('../result/faceOpenCv.jpg', img)

# 显示结果图像
cv2.imshow('Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
