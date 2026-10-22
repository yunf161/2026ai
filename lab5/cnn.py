import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path

EPOCH = 20 #训练轮数 #跑801/101 的数据集改成15
BATCH_SIZE = 8 # 每波次训练样本数
LR = 0.0001 # learning rate  #跑801/101 的数据集改成0.0002




data_dir = Path(__file__).resolve().parent / 'cnn图片'
train_dir = data_dir / 'train'
test_dir = data_dir / 'test'


class CenterMask:
    def __init__(self, p=0.3, scale=(0.12, 0.22), fill=0.5, jitter=4):
        self.p = p
        self.scale = scale
        self.fill = fill
        self.jitter = jitter

    def __call__(self, image):
        if torch.rand(1).item() >= self.p:
            return image

        _, height, width = image.shape
        mask_height = max(1, int(height * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()))
        mask_width = max(1, int(width * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()))

        center_y = height // 2 + int(torch.randint(-self.jitter, self.jitter + 1, (1,)).item())
        center_x = width // 2 + int(torch.randint(-self.jitter, self.jitter + 1, (1,)).item())

        top = max(0, min(height - mask_height, center_y - mask_height // 2))
        left = max(0, min(width - mask_width, center_x - mask_width // 2))

        image = image.clone()
        image[:, top:top + mask_height, left:left + mask_width] = self.fill
        return image



transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((80, 80)),
        torchvision.transforms.CenterCrop((64, 64)),
        torchvision.transforms.ToTensor(),
        CenterMask(p=0.1, scale=(0.12, 0.22), fill=0.5, jitter=4),
        
    ])

train_data = torchvision.datasets.ImageFolder(root=str(train_dir), transform=transform)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
print(train_data.class_to_idx)
print(len(train_data))

transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize((80, 80)),
        torchvision.transforms.CenterCrop((64, 64)),
        torchvision.transforms.ToTensor(),
    ])

test_data = torchvision.datasets.ImageFolder(root=str(test_dir), transform=transform_test)
test_loader = Data.DataLoader(dataset=test_data, batch_size=10, shuffle=False)
print(test_data.class_to_idx)
print(len(test_data))

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( # 3, 64, 64
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5,stride=1, padding=2), # 输入通道数为3，输出通道数为32，卷积核大小为5x5，padding为2
            nn.ReLU(), # 激活函数  # 32, 64, 64
            nn.MaxPool2d(kernel_size=2) # 池化层，池化核大小为2x2  # -> 32, 32, 32
        )
        self.conv2 = nn.Sequential( # -> 32, 32, 32
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2), # 输入通道数为32，输出通道数为64，卷积核大小为5x5，padding为2 # -> 64, 16, 16
            nn.ReLU(), # 激活函数  # 64, 32, 32
            nn.MaxPool2d(kernel_size=2) # 池化层，池化核大小为2x2  # -> 64, 16, 16
        )
        self.fc1 = nn.Linear(64 * 16 * 16, 128) # 输入 64x16 经过两次 2x2 池化后变为 16x16，全连接层，输入特征数为64*16*16，输出特征数为128
        self.fc2 = nn.Linear(128, 5) # 全连接层，输入特征数为128，输出特征数为5（类别数）

    def forward(self, x):
        x = self.conv1(x) # 卷积层1
        x = self.conv2(x) # 卷积层2  #batch_size, 64, 16, 16
        x = x.view(x.size(0), -1) # 展平操作，将多维特征图展平成一维向量  #batch_size, 64*16*16
        x = self.fc1(x) # 全连接层1  #batch_size, 128
        x = self.fc2(x) # 全连接层2  #batch_size, 5
        return x
    
cnn = CNN()
#print(cnn)  
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR) # 优化器
loss_func = nn.CrossEntropyLoss() # 损失函数
best_acc = 0.0
# 训练模型

record_loss = []
record_train_acc = []
record_test_acc = []


for epoch in range(EPOCH):
    if epoch ==15:
       LR = 0.00001
       optimizer.param_groups[0]['lr'] = LR
    train_correct = 0
    train_total = 0   
    epoch_loss=[]
    for step, (b_x, b_y) in enumerate(train_loader): # 训练数据加载器
        output = cnn(b_x) # 前向传播
        
        loss = loss_func(output, b_y) # 计算损失
        optimizer.zero_grad() # 梯度清零
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
        train_correct += (torch.argmax(output, dim=1) == b_y).sum().item()
        train_total += b_y.size(0)
        epoch_loss.append(loss.item())

        if step == 112:
            train_acc = train_correct / train_total # 计算训练准确率
            loss_avg = sum(epoch_loss) / len(epoch_loss) # 计算平均损失
            record_loss.append(loss_avg)
            record_train_acc.append(train_acc)

            cnn.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for test_x, test_y in test_loader:
                    test_output = cnn(test_x)
                    pred_y = torch.argmax(test_output, dim=1)
                    correct += (pred_y == test_y).sum().item()
                    total += test_y.size(0)
            test_acc = correct / total
            record_test_acc.append(test_acc)
            
            print('Epoch:', epoch, '| train loss: %.4f' % loss_avg, '| train acc: %.4f' % train_acc, '| test acc: %.4f' % test_acc)
            
            cnn.train()

#测试模型

cnn.eval()
correct = 0
total = 0
with torch.no_grad():
    for test_x, test_y in test_loader:
            test_output = cnn(test_x)
            pred_y = torch.argmax(test_output, dim=1)
            correct += (pred_y == test_y).sum().item()
            total += test_y.size(0)

    test_acc = correct / total
print('Final Test Accuracy: %.4f' % test_acc)

epoches=range(EPOCH)
fig,axes=plt.subplots(3,1,figsize=(12,8))

axes[0].plot(epoches, record_loss, label='Train Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(epoches, record_train_acc, label='Train Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

axes[2].plot(epoches, record_test_acc, label='Test Accuracy')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Accuracy')
axes[2].legend()

plt.show()
