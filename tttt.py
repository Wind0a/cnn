import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms
import copy
import os

# 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('cuda')


# 参数初始化
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)


# 验证模型在验证集上的正确率
def validate(model, dataset, batch_size):
    val_loader = data.DataLoader(dataset, batch_size, pin_memory=True)
    result, num = 0.0, 0
    model.eval()  # 评估模式
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            pred = torch.argmax(pred, dim=1)
            result += (pred == labels).sum().item()
            num += labels.size(0)
    acc = result / num
    return acc


# 早停类
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0


# 自定义数据集
class FaceDataset(data.Dataset):
    def __init__(self, root, transform=None):
        super(FaceDataset, self).__init__()
        self.root = root
        self.transform = transform
        df_path = pd.read_csv(root + '\\image_emotion.csv', header=None, usecols=[0])
        df_label = pd.read_csv(root + '\\image_emotion.csv', header=None, usecols=[1])
        self.path = np.array(df_path)[:, 0]
        self.label = np.array(df_label)[:, 0].astype(np.int64)

    def __getitem__(self, item):
        face = cv2.imread(self.root + '\\' + self.path[item])
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_hist = cv2.equalizeHist(face_gray)
        face_normalized = face_hist.reshape(1, 48, 48) / 255.0
        face_tensor = torch.from_numpy(face_normalized).type(torch.FloatTensor)
        
        if self.transform is not None:
            face_tensor = self.transform(face_tensor)
        
        label = torch.tensor(self.label[item], dtype=torch.long)
        return face_tensor, label

    def __len__(self):
        return self.path.shape[0]


class FaceCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(FaceCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.2)
        )

        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.conv3.apply(gaussian_weights_init)

        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(256 * 6 * 6, 2048),  # 减少参数
            nn.RReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(2048, 512),
            nn.RReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 7),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)


def compute_confusion_matrix(model, dataset, batch_size):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data.DataLoader(dataset, batch_size):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    return cm


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('./confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics(metrics):
    emotions = list(metrics.keys())
    precision_values = [metrics[emotion]['precision'] for emotion in emotions]
    recall_values = [metrics[emotion]['recall'] for emotion in emotions]
    f1_values = [metrics[emotion]['f1-score'] for emotion in emotions] if 'f1-score' in metrics[emotions[0]] else None

    x = np.arange(len(emotions))
    width = 0.25

    plt.figure(figsize=(14, 6))
    plt.bar(x - width, precision_values, width, label='Precision')
    plt.bar(x, recall_values, width, label='Recall')
    if f1_values is not None:
        plt.bar(x + width, f1_values, width, label='F1-score')

    plt.xlabel('Emotion Classes')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1-score for Each Emotion Class')
    plt.xticks(x, emotions, rotation=45)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 在柱状图上添加数值标签
    for i, v in enumerate(precision_values):
        plt.text(i - width, v + 0.01, f'{v:.2f}', ha='center')
    for i, v in enumerate(recall_values):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
    if f1_values is not None:
        for i, v in enumerate(f1_values):
            plt.text(i + width, v + 0.01, f'{v:.2f}', ha='center')

    plt.savefig('./precision_recall_f1.png', dpi=300, bbox_inches='tight')
    plt.close()


def compute_metrics(model, dataset, batch_size):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data.DataLoader(dataset, batch_size):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算分类报告
    report = classification_report(all_labels, all_preds, 
                                 target_names=list(emotion_labels.values()),
                                 output_dict=True)
    
    # 提取每类的精确率、召回率和F1-score
    metrics = {}
    for emotion in emotion_labels.values():
        metrics[emotion] = {
            'precision': report[emotion]['precision'],
            'recall': report[emotion]['recall'],
            'f1-score': report[emotion]['f1-score']
        }
    
    return metrics, all_labels, all_preds


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(model, optimizer, scheduler, filename='checkpoint.pth'):
    if os.path.isfile(filename):
        print(f"Loading checkpoint: {filename}")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_val_loss = checkpoint['best_val_loss']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        train_accuracies = checkpoint['train_accuracies']
        val_accuracies = checkpoint['val_accuracies']
        epochs_list = checkpoint['epochs_list']
        print(f"Loaded checkpoint at epoch {start_epoch}")
        return start_epoch, best_val_loss, train_losses, val_losses, train_accuracies, val_accuracies, epochs_list
    else:
        print(f"No checkpoint found at {filename}")
        return 0, float('inf'), [], [], [], [], []

def train(train_dataset, val_dataset, batch_size, epochs, learning_rate, wt_decay, resume=False):
    train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True)
    model = FaceCNN(dropout_rate=0.5).to(device)
    
    # 计算类别权重
    labels = torch.tensor([sample[1] for sample in train_dataset])
    class_counts = torch.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts.float())
    class_weights = class_weights.to(device)

    # 使用带权重的交叉熵损失
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wt_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=5
    )

    # 初始化或加载检查点
    if resume:
        start_epoch, best_val_loss, train_losses, val_losses, train_accuracies, val_accuracies, epochs_list = load_checkpoint(
            model, optimizer, scheduler)
    else:
        start_epoch = 0
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        epochs_list = []
    
    # 早停
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            total_loss = 0
            num_batches = 0
            
            for images, emotion in train_loader:
                images = images.to(device)
                emotion = emotion.to(device)

                optimizer.zero_grad()
                output = model(images)
                loss = loss_function(output, emotion)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_train_loss = total_loss / num_batches
            train_losses.append(avg_train_loss)
            epochs_list.append(epoch + 1)

            print('After {} epochs , the loss_rate is : '.format(epoch + 1), avg_train_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print('Current learning rate: {:.6f}'.format(current_lr))

            # 验证阶段
            model.eval()
            val_loss = 0
            val_batches = 0
            with torch.no_grad():
                for images, emotion in data.DataLoader(val_dataset, batch_size, pin_memory=True):
                    images = images.to(device)
                    emotion = emotion.to(device)
                    output = model(images)
                    loss = loss_function(output, emotion)
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)
            
            # 更新学习率
            scheduler.step(avg_val_loss)
            
            # 每5个epoch记录一次准确率
            if epoch % 5 == 0:
                acc_train = validate(model, train_dataset, batch_size)
                acc_val = validate(model, val_dataset, batch_size)
                train_accuracies.append(acc_train)
                val_accuracies.append(acc_val)
                print('After {} epochs , the acc_train is : '.format(epoch + 1), acc_train)
                print('After {} epochs , the acc_val is : '.format(epoch + 1), acc_val)

            # 保存检查点
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_accuracies': train_accuracies,
                    'val_accuracies': val_accuracies,
                    'epochs_list': epochs_list
                })

            # 早停检查
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                model.load_state_dict(early_stopping.best_model)
                break

    except Exception as e:
        print(f"Training interrupted: {str(e)}")
        # 保存当前状态
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'epochs_list': epochs_list
        }, 'interrupt_checkpoint.pth')
        raise e

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, train_losses, label='Training Loss')
    plt.plot(epochs_list, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('./loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 绘制准确率曲线
    accuracy_epochs = [i * 5 for i in range(len(train_accuracies))]
    plt.figure(figsize=(10, 6))
    plt.plot(accuracy_epochs, train_accuracies, label='Training Accuracy')
    plt.plot(accuracy_epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('./accuracy_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 绘制混淆矩阵
    cm = compute_confusion_matrix(model, val_dataset, batch_size)
    class_names = list(emotion_labels.values())
    plot_confusion_matrix(cm, class_names)

    # 绘制精确率、召回率与F1-score柱状图
    metrics, val_labels, val_preds = compute_metrics(model, val_dataset, batch_size)
    plot_metrics(metrics)

    return model


def main():
    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.1),
    ])

    train_dataset = FaceDataset(root='face_images/train_set', transform=train_transform)
    val_dataset = FaceDataset(root='face_images/verify_set')
    
    try:
        model = train(train_dataset, val_dataset, 
                     batch_size=64,
                     epochs=100, 
                     learning_rate=0.001,
                     wt_decay=0.01,
                     resume=True)  # 设置resume=True来从检查点恢复
        
        torch.save(model.state_dict(), 'model/model_cnn.pth')
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e


emotion_labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

if __name__ == '__main__':
    main()
