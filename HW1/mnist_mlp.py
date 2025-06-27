import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from matplotlib import pyplot as plt

X_train = np.load('./mnist/X_train.npy')
y_train = np.load('./mnist/y_train.npy')
X_val = np.load('./mnist/X_val.npy')
y_val = np.load('./mnist/y_val.npy')
X_test = np.load('./mnist/X_test.npy')
y_test = np.load('./mnist/y_test.npy')

class MNISTDataset(Dataset):

    def __init__(self, data=X_train, label=y_train):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = self.data[index].astype('float32') 
        label = self.label[index].astype('int64') 
        return data, label

    def __len__(self):
        return len(self.data)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(1024,512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.tanh(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

model = Net()
model.to(device='cuda')

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-5)


train_loader = DataLoader(MNISTDataset(X_train, y_train), \
                            batch_size=64, shuffle=True)
val_loader = DataLoader(MNISTDataset(X_val, y_val), \
                            batch_size=64, shuffle=True)
test_loader = DataLoader(MNISTDataset(X_test, y_test), \
                            batch_size=64, shuffle=True)


EPOCHS = 10

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'test_loss': [], 'test_acc': []}
for epoch in range(EPOCHS):
    model.train()

    loss_train = []
    acc_train = []
    correct_train = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device='cuda'), target.to(device='cuda')
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        loss_train.append(loss.item())
        pred = output.max(1, keepdim=True)[1]
        
        correct = pred.eq(target.view_as(pred)).sum().item()
        correct_train += correct
        acc_train.append(100.* correct / len(data))

    history['train_loss'].append(np.mean(loss_train))
    history['train_acc'].append(np.mean(acc_train))
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        np.mean(loss_train), correct_train, len(train_loader.dataset),
        100. * correct_train / len(train_loader.dataset)))
    

    model.eval()
    val_loss = []
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device='cuda'), target.to(device='cuda')
            
            output = model(data)
            val_loss.append(criterion(output, target).item())
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss = np.mean(val_loss)

    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

    history['val_loss'].append(val_loss)
    history['val_acc'].append(100. *correct / len(val_loader.dataset))

print("\nEvaluating Test set...")
model.eval()
test_loss = []
test_correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device='cuda'), target.to(device='cuda')
        output = model(data)
        test_loss.append(criterion(output, target).item())
        pred = output.max(1, keepdim=True)[1]
        test_correct += pred.eq(target.view_as(pred)).sum().item()
test_loss = np.mean(test_loss)
test_accuracy = 100. * test_correct / len(test_loader.dataset)
print('Test Result: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
    test_loss, test_correct, len(test_loader.dataset), test_accuracy))
history['test_loss'].append(test_loss)
history['test_acc'].append(test_accuracy)

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='train_loss')
plt.plot(history['val_loss'], label='val_loss')
plt.plot([EPOCHS-1], [history['test_loss'][0]], label='test_loss', marker='o')
plt.title('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='train_acc')
plt.plot(history['val_acc'], label='val_acc')
plt.plot([EPOCHS-1], [history['test_acc'][0]], label='test_acc', marker='o')
plt.title('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()