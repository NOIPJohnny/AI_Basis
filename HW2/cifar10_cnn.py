import os
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

def main():
    batch_size = 128
    learning_rate = 0.0005
    num_epochs = 100

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0), 
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                            download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                                download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8) 

    class CNN(nn.Module):
        def __init__(self, num_classes=10):
            super(CNN, self).__init__()
            
            self.conv_block1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1), 
                
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
            self.conv_block2 = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2),
                
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
            self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
            
            self.classifier = nn.Sequential(
                nn.Linear(256, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(1024, num_classes)
            )
            
            self._initialize_weights()
        
        def forward(self, x):
            x = self.conv_block1(x)
            x = self.conv_block2(x)
            x = self.global_avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
        
        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)


    model = CNN()

    use_mlu = False
    try:
        use_mlu = torch.mlu.is_available()
    except:
        use_mlu = False

    if use_mlu:
        device = torch.device('mlu:0')
    else:
        print("MLU is not available, use GPU/CPU instead.")
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print("Using GPU.")
            print("GPU Device Name:", torch.cuda.get_device_name())
        else:
            device = torch.device('cpu')
            print("Using CPU.")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            if (i + 1) % 50 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch + 1, num_epochs, i + 1, len(train_loader), 
                            loss.item(), 100 * train_correct / train_total))

        train_accuracy = 100 * train_correct / train_total
        print('Epoch [{}/{}], Training Accuracy: {:.2f}%'.format(
            epoch + 1, num_epochs, train_accuracy))

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            test_loss = 0.0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            test_accuracy = 100 * correct / total
            print('Epoch [{}/{}], Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(
                epoch + 1, num_epochs, test_loss / len(test_loader), test_accuracy))
            
            scheduler.step(test_accuracy)
            

    print("Starting final evaluation...")
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        final_test_loss = 0.0
        class_correct = [0] * 10  
        class_total = [0] * 10    
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            final_test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for i in range(labels.size(0)):
                label = labels[i]
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

        final_accuracy = 100 * correct / total
        print('Final Model Performance:')
        print(f'Test Loss: {final_test_loss / len(test_loader):.4f}')
        print(f'Test Accuracy: {final_accuracy:.2f}%')
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        print('\nClass-wise Accuracy:')
        for i in range(10):
            print(f'{classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')

if __name__ == '__main__':
    main()