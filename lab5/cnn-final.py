import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import copy
import random
from pathlib import Path

EPOCH = 60 #训练轮数 #跑801/101 的数据集改成15
BATCH_SIZE = 8 # 每波次训练样本数
LR = 0.0005 # learning rate
WEIGHT_DECAY = 1e-4

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)




data_dir = Path(__file__).resolve().parent / 'cnn图片'
train_dir = data_dir / 'train_01'
test_dir = data_dir / 'test'
val_dir = data_dir / 'val'

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



mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((80, 80)),
        torchvision.transforms.CenterCrop((64, 64)),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
        torchvision.transforms.ToTensor(),
        CenterMask(p=0.03, scale=(0.10, 0.18), fill=0.5, jitter=3),
        torchvision.transforms.Normalize(mean=mean, std=std),
    ])

train_data = torchvision.datasets.ImageFolder(root=str(train_dir), transform=transform)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
print(train_data.class_to_idx)
print(len(train_data))



transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize((80, 80)),
        torchvision.transforms.CenterCrop((64, 64)),
        torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=mean, std=std),
    ])

train_eval_data = torchvision.datasets.ImageFolder(root=str(train_dir), transform=transform_test)
train_eval_loader = Data.DataLoader(dataset=train_eval_data, batch_size=32, shuffle=False)

val_data = torchvision.datasets.ImageFolder(root=str(val_dir), transform=transform_test)
val_loader = Data.DataLoader(dataset=val_data, batch_size=10, shuffle=False)
print(val_data.class_to_idx)
print(len(val_data))


test_data = torchvision.datasets.ImageFolder(root=str(test_dir), transform=transform_test)
test_loader = Data.DataLoader(dataset=test_data, batch_size=10, shuffle=False)
print(test_data.class_to_idx)
print(len(test_data))

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( # 3, 64, 64
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5,stride=1, padding=2), # 输入通道数为3，输出通道数为32，卷积核大小为5x5，padding为2
            nn.BatchNorm2d(32),
            nn.ReLU(), # 激活函数  # 32, 64, 64
            nn.MaxPool2d(kernel_size=2) # 池化层，池化核大小为2x2  # -> 32, 32, 32
        )
        self.conv2 = nn.Sequential( # -> 32, 32, 32
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2), # 输入通道数为32，输出通道数为64，卷积核大小为5x5，padding为2 # -> 64, 16, 16
            nn.BatchNorm2d(64),
            nn.ReLU(), # 激活函数  # 64, 32, 32
            nn.MaxPool2d(kernel_size=2) # 池化层，池化核大小为2x2  # -> 64, 16, 16
        )
        self.conv3 = nn.Sequential( 
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2), # 输入通道数为64，输出通道数为128，卷积核大小为5x5，padding为2 # -> 128, 16, 16, 16
            nn.BatchNorm2d(128),
            nn.ReLU(), # 激活函数  # 128, 16, 16, 16
            nn.MaxPool2d(kernel_size=2) # 池化层，池化核大小为2x2  # -> 128, 8, 8, 8
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1)) # 全局平均池化，减少参数量
        self.fc = nn.Linear(128, 5) # 全连接层，输入特征数为128，输出特征数为5（类别数）

    def forward(self, x):
        x = self.conv1(x) # 卷积层1
        x = self.conv2(x) # 卷积层2  #batch_size, 64, 16, 16
        x = self.conv3(x) # 卷积层3  #batch_size, 128, 8, 8, 
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x) # 全连接层  #batch_size, 5
        return x
    
cnn = CNN()
#print(cnn)  
optimizer = torch.optim.AdamW(cnn.parameters(), lr=LR, weight_decay=WEIGHT_DECAY) # 优化器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=4,
    min_lr=1e-6
)
class_counts = torch.bincount(torch.tensor(train_data.targets), minlength=len(train_data.classes)).float()
class_weights = class_counts.sum() / (len(train_data.classes) * class_counts)
loss_func = nn.CrossEntropyLoss(weight=class_weights) # 使用类别权重缓解类别不均衡
best_acc = 0.0
best_state_dict = copy.deepcopy(cnn.state_dict())
# 训练模型

record_loss = []
record_train_aug_acc = []
record_train_clean_acc = []
record_val_acc = []


for epoch in range(EPOCH):
    cnn.train()
    epoch_loss=[]
    for step, (b_x, b_y) in enumerate(train_loader): # 训练数据加载器
        output = cnn(b_x) # 前向传播
        
        loss = loss_func(output, b_y) # 计算损失
        optimizer.zero_grad() # 梯度清零
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
        epoch_loss.append(loss.item())

    cnn.eval()
    train_aug_correct = 0
    train_aug_total = 0
    with torch.no_grad():
        for train_x_aug, train_y_aug in train_loader:
            train_aug_output = cnn(train_x_aug)
            train_aug_pred = torch.argmax(train_aug_output, dim=1)
            train_aug_correct += (train_aug_pred == train_y_aug).sum().item()
            train_aug_total += train_y_aug.size(0)
    train_acc = train_aug_correct / train_aug_total # eval模式下记录train aug准确率

    loss_avg = sum(epoch_loss) / len(epoch_loss) # 计算平均损失
    record_loss.append(loss_avg)
    record_train_aug_acc.append(train_acc)

    clean_correct = 0
    clean_total = 0
    with torch.no_grad():
        for train_x, train_y in train_eval_loader:
            train_output = cnn(train_x)
            train_pred = torch.argmax(train_output, dim=1)
            clean_correct += (train_pred == train_y).sum().item()
            clean_total += train_y.size(0)
    train_clean_acc = clean_correct / clean_total
    record_train_clean_acc.append(train_clean_acc)

    correct = 0
    total = 0

    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_output = cnn(val_x)
            pred_y = torch.argmax(val_output, dim=1)
            correct += (pred_y == val_y).sum().item()
            total += val_y.size(0)
    val_acc = correct / total
    record_val_acc.append(val_acc)
    scheduler.step(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        best_state_dict = copy.deepcopy(cnn.state_dict())

    current_lr = optimizer.param_groups[0]['lr']
    print('Epoch:', epoch + 1, '| train loss: %.4f' % loss_avg, '| train aug acc: %.4f' % train_acc, '| train clean acc: %.4f' % train_clean_acc, '| val acc: %.4f' % val_acc, '| lr: %.6f' % current_lr)

#测试模型

cnn.load_state_dict(best_state_dict)
print('Best Val Accuracy: %.4f' % best_acc)

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

epoches = range(1, len(record_loss) + 1)
if len(record_loss) == 0:
    print('No training records to plot.')
else:
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    axes[0].plot(epoches, record_loss, label='Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(epoches, record_train_aug_acc, label='Train Aug Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    axes[2].plot(epoches, record_train_clean_acc, label='Train Clean Accuracy')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].legend()

    axes[3].plot(epoches, record_val_acc, label='Val Accuracy')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Accuracy')
    axes[3].legend()

    plt.tight_layout()
    plt.show()
