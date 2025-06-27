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
# 配置参数
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Using device: ", device)
# 创建目录
os.makedirs(flags["checkpoint_dir"], exist_ok=True)
os.makedirs(flags["sample_dir"], exist_ok=True)

# 数据加载
transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))  # MNIST是单通道
])

# 创建用于训练的MNIST数据集对象
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)

# 初始化训练数据加载器，用于批量加载和打乱训练数据
train_loader = DataLoader(train_dataset, batch_size=flags["batch_size"], shuffle=True)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        # 定义第一个转置卷积层，用于将输入的100通道的特征图转换为512通道的4x4特征图
        self.cnn1 = nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=512)  # 对512通道的特征图进行批归一化处理
        self.act1 = nn.ReLU(inplace=True)  # 应用ReLU激活函数
        # 输入: 512x4x4 → 输出: 256x8x8
        self.cnn2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=256)  # 对256通道的特征图进行批归一化处理
        self.act2 = nn.ReLU(inplace=True)  # 应用ReLU激活函数
        # 输入: 256x8x8 → 输出: 128x14x14
        self.cnn3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=7, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=128)  # 对128通道的特征图进行批归一化处理
        self.act3 = nn.ReLU(inplace=True)  # 应用ReLU激活函数
        # 输入: 128x14x14 → 输出: 1x28x28
        self.cnn4 = nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False)
        self.act4 = nn.Tanh()  # 应用Tanh激活函数，输出范围为[-1, 1]

    def forward(self, x):
        x = x.view(-1, 100, 1, 1)
        # 对输入数据进行批归一化和卷积操作，然后应用激活函数
        x = self.act1(self.bn1(self.cnn1(x)))
        # 对第二次卷积操作的结果进行批归一化和卷积操作，然后应用激活函数
        x = self.act2(self.bn2(self.cnn2(x)))
        # 对第三次卷积操作的结果进行批归一化和卷积操作，然后应用激活函数
        x = self.act3(self.bn3(self.cnn3(x)))
        # 对第四次卷积操作的结果应用激活函数
        x = self.act4(self.cnn4(x))
        return x


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        # 输入:1x28x28 → 输出:128x14x14
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        self.act1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # 输入:128x14x14 → 输出:256x7x7
        self.cnn2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=256)
        self.act2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # 输入:256x7x7 → 输出:512x3x3
        self.cnn3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=512)
        self.act3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # 输入:512x3x3 → 输出:1x1x1
        self.cnn4 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)
        self.act4 = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.cnn1(x))
        x = self.act2(self.bn1(self.cnn2(x)))
        x = self.act3(self.bn2(self.cnn3(x)))
        x = self.act4(self.cnn4(x))
        return x.squeeze().squeeze().squeeze()


# 初始化模型
G = Generator().to(device)
D = Discriminator().to(device)

# 优化器
d_optimizer = optim.Adam(D.parameters(), lr=flags["lr"], betas=(flags["beta1"], 0.999))
g_optimizer = optim.Adam(G.parameters(), lr=flags["lr"], betas=(flags["beta1"], 0.999))

# 损失函数: TODO
pass

# 固定噪声用于生成样本
fixed_z = torch.randn(flags["sample_size"], flags["z_dim"], device=device)

# 增强可视化配置
flags["log_interval"] = 100  # 每100个batch记录一次
flags["sample_rows"] = 8  # 可视化图片的行列数


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
    # 生成并保存图片
    with torch.no_grad():
        generated = generator(fixed_z).cpu()

    # 保存为单独图片文件
    filename = f"epoch{epoch}_batch{batches_done}.png"

    # 创建对比图（真实 vs 生成）
    real_images = next(iter(train_loader))[0][:flags["sample_size"]].cpu()
    comparison = torch.cat([real_images, generated])
    save_image(comparison, os.path.join(flags["sample_dir"], f"compare_{filename}"), nrow=flags["sample_rows"], normalize=True)


# 初始化训练记录
losses = {'d': [], 'g': []}


def train():
    fixed_z = torch.randn(flags["sample_size"], flags["z_dim"], device=device)
    total_batches = 0  # 记录总batch数

    for epoch in range(flags["n_epoch"]):
        start_time = time.time()  # 每个epoch记录开始时间
        for i, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # DCGAN 训练步骤提示：
            # 1. 训练判别器D，使其能够区分真实图片和生成图片：
            #    a. 真实图片通过判别器D，得到真实图片的判别结果，记为D(x)
            #    b. 生成图片通过判别器D，得到生成图片的判别结果，记为D(G(z))
            #    c. 计算判别器D的损失，使其能够区分真实图片和生成图片：loss = -log(D(x)) - log(1-D(G(z)))
            #    d. 计算判别器D的梯度，更新判别器D的参数
            # 2. 训练生成器G，使其能够生成更逼真的图片：
            #    a. 随机生成噪声z，通过生成器G，得到生成图片G(z)
            #    b. 计算生成器G的损失，使其能够生成更逼真的图片：loss = -log(D(G(z)))
            #    c. 计算生成器G的梯度，更新生成器G的参数


            # 判别器的损失: TODO
            pass
            # 判别器梯度更新: TODO
            pass


            # 生成器的损失: TODO
            pass
            # 生成器梯度更新: TODO
            pass    


            # 记录损失
            losses['d'].append(d_loss.item())
            losses['g'].append(g_loss.item())

            # 可视化过程
            if total_batches % flags["log_interval"] == 0:

                # 打印损失曲线
                plot_loss(losses)

                # 打印进度信息
                print(f"[Epoch {epoch}/{flags['n_epoch']}] "
                      f"[Batch {i}/{len(train_loader)}] "
                      f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

            total_batches += 1

        # 每个epoch结束后保存
        save_sample_images(epoch, "end", G, fixed_z)
        torch.save(G.state_dict(), os.path.join(flags["checkpoint_dir"], f"G_epoch{epoch}.pth"))
        torch.save(D.state_dict(), os.path.join(flags["checkpoint_dir"], f"D_epoch{epoch}.pth"))
        print(f"Epoch {epoch} took {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    train()
