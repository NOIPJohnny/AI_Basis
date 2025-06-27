import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import load_imdb_dataset, Accuracy
import sys
from utils import load_imdb_dataset, Accuracy
from tqdm import tqdm

print("MLU is not available, use GPU/CPU instead.")
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("Using GPU, device: ",torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')

X_train, y_train, X_test, y_test = load_imdb_dataset('data', nb_words=20000, test_split=0.2)

seq_Len = 200
vocab_size = len(X_train) + 1
print("vocab_size: ", vocab_size)

class ImdbDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):

        data = self.X[index]
        data = np.concatenate([data[:seq_Len], [0] * (seq_Len - len(data))]).astype('int32')  # set
        label = self.y[index]
        return data, label

    def __len__(self):

        return len(self.y)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.W_ii= nn.Linear(input_size, hidden_size, bias=True)
        self.W_hi= nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_if= nn.Linear(input_size, hidden_size, bias=True)
        self.W_hf= nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_ig= nn.Linear(input_size, hidden_size, bias=True)
        self.W_hg= nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_io= nn.Linear(input_size, hidden_size, bias=True)
        self.W_ho= nn.Linear(hidden_size, hidden_size, bias=True)
    
    def forward(self, x):
        seq_len, batch_size, input_size = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        outputs = []
        for t in range(seq_len):
            x_t = x[t]
            i_t = torch.sigmoid(self.W_ii(x_t) + self.W_hi(h_t))
            f_t = torch.sigmoid(self.W_if(x_t) + self.W_hf(h_t))
            g_t = torch.tanh(self.W_ig(x_t) + self.W_hg(h_t))
            o_t = torch.sigmoid(self.W_io(x_t) + self.W_ho(h_t))
            #上面的c_t和h_t是上一个时刻的c_t和h_t，即c_{t-1}和h_{t-1}
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            outputs.append(h_t)
        outputs = torch.stack(outputs, dim=0)
        return outputs, (h_t, c_t)

class Net(nn.Module):
    def __init__(self, embedding_size=64, hidden_size=64, num_classes=2):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = LSTM(input_size=embedding_size, hidden_size=hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        outputs, (h_t, c_t) = self.lstm(x)
        x = torch.mean(outputs, dim=0)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


n_epoch = 5
batch_size = 64
print_freq = 2

train_dataset = ImdbDataset(X=X_train, y=y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = ImdbDataset(X=X_test, y=y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

net = Net()
metric = Accuracy()
print(net)


def train(model, device, train_loader, optimizer, epoch):
    model = model.to(device)
    model.train()
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
    train_acc = 0
    train_loss = 0
    n_iter = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.long()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        metric.update(output, target)
        train_acc += metric.result()
        train_loss += loss.item()
        metric.reset()
        n_iter += 1
    print('Train Epoch: {} Loss: {:.6f} \t Acc: {:.6f}'.format(epoch, train_loss / n_iter, train_acc / n_iter))


def test(model, device, test_loader):
    model = model.to(device)
    model.eval()
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
    test_loss = 0
    test_acc = 0
    n_iter = 0
    with torch.no_grad():
        for data, target in test_loader:
            target = target.long()
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += loss_func(output, target).item()
            metric.update(output, target)
            test_acc += metric.result()
            metric.reset()
            n_iter += 1
    test_loss /= n_iter
    test_acc /= n_iter
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_acc))


optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0)
gamma = 0.7
for epoch in range(1, n_epoch + 1):
    train(net, device, train_loader, optimizer, epoch)
    test(net, device, test_loader)
