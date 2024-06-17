#!/usr/bin/env python
# -*- coding = utf-8 -*-
# @Time       : 2023/10/8 14:43
# @Author     : fany
# @Project    : PyCharm
# @File       : faceDlib.py
# @description: faceDlib
import dlib
import cv2

# 加载dlib的人脸检测器
detector = dlib.get_frontal_face_detector()

# 加载图像
img = cv2.imread('../img/Talisa_Soto_0001.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 检测人脸
faces = detector(gray)

# 在图像上绘制人脸框
for face in faces:
    x, y, w, h = (face.left(), face.top(), face.width(), face.height())
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)


cv2.imwrite('../result/faceDlib.jpg', img)  # 保存为JPEG图像


# 显示结果图像
cv2.imshow('Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
