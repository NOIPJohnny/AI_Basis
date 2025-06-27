import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import torchvision

start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
flags = {
    "batch_size": 128,
    "z_dim": 100,
    "n_epoch": 10,
    "lr": 0.0002,
    "beta1": 0.5,
    "save_every_epoch": 5,
    "sample_size": 64,
    "checkpoint_dir": "checkpoints/{}".format(start_time),
    "sample_dir": "samples/{}".format(start_time),
}
print(start_time)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
os.makedirs(flags["checkpoint_dir"], exist_ok=True)
os.makedirs(flags["sample_dir"], exist_ok=True)

transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])


train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=flags["batch_size"], shuffle=True)

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        
        self.cnn1 = nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=512)
        self.act1 = nn.ReLU(inplace=True)
        
        self.cnn2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=256)
        self.act2 = nn.ReLU(inplace=True)
       
        self.cnn3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=7, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        self.act3 = nn.ReLU(inplace=True)
        
        self.cnn4 = nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False)
        self.act4 = nn.Tanh() 
    def forward(self, x):
        x = x.view(-1, 100, 1, 1)
        x = self.act1(self.bn1(self.cnn1(x)))
        x = self.act2(self.bn2(self.cnn2(x)))
        x = self.act3(self.bn3(self.cnn3(x)))
        x = self.act4(self.cnn4(x))
        return x


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        self.act1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.cnn2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=256)
        self.act2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.cnn3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=512)
        self.act3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.cnn4 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)
        self.act4 = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.cnn1(x))
        x = self.act2(self.bn1(self.cnn2(x)))
        x = self.act3(self.bn2(self.cnn3(x)))
        x = self.act4(self.cnn4(x))
        return x.squeeze().squeeze().squeeze()


G = Generator().to(device)
D = Discriminator().to(device)

d_optimizer = optim.Adam(D.parameters(), lr=flags["lr"], betas=(flags["beta1"], 0.5))
g_optimizer = optim.Adam(G.parameters(), lr=flags["lr"], betas=(flags["beta1"], 0.5))

criterion = nn.BCELoss()

fixed_z = torch.randn(flags["sample_size"], flags["z_dim"], device=device)

flags["log_interval"] = 100
flags["sample_rows"] = 8 

def plot_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.title("Training Losses")
    plt.plot(losses['d'], label="Discriminator")
    plt.plot(losses['g'], label="Generator")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(flags["sample_dir"], "training_loss.png"))
    plt.close()


def save_sample_images(epoch, batches_done, generator, fixed_z):
    with torch.no_grad():
        generated = generator(fixed_z).cpu()

    filename = f"epoch{epoch}_batch{batches_done}.png"

    real_images = next(iter(train_loader))[0][:flags["sample_size"]].cpu()
    comparison = torch.cat([real_images, generated])
    save_image(comparison, os.path.join(flags["sample_dir"], f"compare_{filename}"), nrow=flags["sample_rows"], normalize=True)

losses = {'d': [], 'g': []}

def train():
    fixed_z = torch.randn(flags["sample_size"], flags["z_dim"], device=device)
    total_batches = 0
    for epoch in range(flags["n_epoch"]):
        start_time = time.time() 
        for i, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)

            D.zero_grad() 
            d_real_output = D(real_images)
            d_real_loss = criterion(d_real_output, real_labels)
            
            z = torch.randn(batch_size, flags["z_dim"], device=device)
            fake_images = G(z)
            d_fake_output = D(fake_images.detach())
            d_fake_loss = criterion(d_fake_output, fake_labels)
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            

            G.zero_grad()
            g_output = D(fake_images)
            g_loss = criterion(g_output, real_labels)
            g_loss.backward()
            g_optimizer.step()

            losses['d'].append(d_loss.item())
            losses['g'].append(g_loss.item())
            if total_batches % flags["log_interval"] == 0:
                plot_loss(losses)
                print(f"[Epoch {epoch+1}/{flags['n_epoch']}] "
                      f"[Batch {i}/{len(train_loader)}] "
                      f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")
                
            total_batches += 1

        save_sample_images(epoch+1, "end", G, fixed_z)
        torch.save(G.state_dict(), os.path.join(flags["checkpoint_dir"], f"G_epoch{epoch+1}.pth"))
        torch.save(D.state_dict(), os.path.join(flags["checkpoint_dir"], f"D_epoch{epoch+1}.pth"))
        print(f"Epoch {epoch+1} took {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    train()