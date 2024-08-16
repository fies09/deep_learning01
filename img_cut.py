import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Cityscapes
from torchvision.models import segmentation

# 定义数据预处理和增强的转换
transform = transforms.Com[pos](https://geek.csdn.net/educolumn/0399089ce1ac05d7729a569fd611cf73?spm=1055.2569.3001.10083)e([
    transforms.To[tensor](https://geek.csdn.net/educolumn/0ebc891269ff76b86c4b41f64bffd5db?spm=1055.2569.3001.10083)(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载Cityscapes数据集
train_set = Cityscapes(root='path_to_dataset', split='train', transform=transform)
val_set = Cityscapes(root='path_to_dataset', split='val', transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
val_loader = DataLoader(val_set, batch_size=4, shuffle=False)

# 定义模型
model = segmentation.deeplabv3_resnet50(num_classes=19)  # 19是Cityscapes数据集中的类别数

# 定义损失[函数](https://geek.csdn.net/educolumn/ba94496e6cfa8630df5d047358ad9719?dp_token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpZCI6NDQ0MDg2MiwiZXhwIjoxNzA3MzcxOTM4LCJpYXQiOjE3MDY3NjcxMzgsInVzZXJuYW1lIjoid2VpeGluXzY4NjQ1NjQ1In0.RrTYEnMNYPC7AQdoij4SBb0kKEgHoyvF-bZOG2eGQvc&amp;spm=1055.2569.3001.10083)和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练模型
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        total_iou = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            predicted_labels = torch.argmax(outputs, dim=1)
            
            iou = (predicted_labels == labels).sum().item() / (labels.size(0) * labels.size(1) * labels.size(2))
            total_iou += iou
        
        average_iou = total_iou / len(val_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, IoU: {average_iou}')

# 使用训练好的模型进行预测
test_image = torch.randn(1, 3, 512, 512).to(device)
model.eval()
with torch.no_grad():
    test_image = test_image.to(device)
    output = model(test_image)
    predicted_label = torch.argmax(output, dim=1)

# 输出预测结果
print(predicted_label)