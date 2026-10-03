
```py
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, num_blocks, reduce_fm_size=True):
        super(ResidualConvBlock, self).__init__()

        num_features = in_channels * 2 if reduce_fm_size else in_channels
        modules = []

        for i in range(num_blocks):                          ①
            if i == 0 and reduce_fm_size:
                stride = 2
                downsample = nn.Sequential( 
                    nn.Conv2d(                               ②
                        in_channels, num_features,
                        kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(num_features, eps=0.001),
                )
                basic_block = BasicBlock(
                    in_channels=in_channels, num_features=num_features,
                    stride=stride, downsample=downsample)
            else:
                basic_block = BasicBlock(
                    in_channels=num_features, num_features=num_features, stride=1)
            modules.append(basic_block)

        self.conv_block = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv_block(x)
```

① 残差块是基本块的堆叠

② 1 × 1 卷积用于下采样输入特征图

这样，我们就准备好实现 ResNet-34。

列表 11.10 ResNet-34 的 PyTorch 代码

```py
class ResNet34(nn.Module):
    def __init__(self, num_basic_blocks, num_classes):
        super(ResNet, self).__init__()
        conv1 = nn.Sequential(                       ①
            nn.Conv2d(3, 64, kernel_size=7,
                stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1)
        )

        assert len(num_basic_blocks) == 4            ②

        conv2 = ResidualConvBlock(                   ③
            in_channels=64, num_blocks=num_basic_blocks[0], reduce_fm_size=False)
        conv3 = ResidualConvBlock(
            in_channels=64, num_blocks=num_basic_blocks[1], reduce_fm_size=True)
        conv4 = ResidualConvBlock(
            in_channels=128, num_blocks=num_basic_blocks[2], reduce_fm_size=True)
        conv5 = ResidualConvBlock(
            in_channels=256, num_blocks=num_basic_blocks[3], reduce_fm_size=True)

        self.conv_backbone = nn.Sequential(*[conv1, conv2, conv3, conv4, conv5])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        conv_out = self.conv_backbone(x)
        conv_out = self.avg_pool(conv_out)
        logits = self.classifier(                    ④
            conv_out.view(conv_out.shape[0], -1))
        return logits
```

① 实例化第一个卷积层

② 大小为 4 的列表，指定每个 `ResidualConvBlock` 的基本块数量

③ 实例化四个残差块

④ 在传递给分类器之前对卷积特征进行展平

如前所述，我们通常不会实现自己的 ResNet。相反，我们使用来自 `torchvision` 包的现成实现，如下所示：

```py
import torchvision
resnet34 = torchvision.models.resnet34()  ①
```

① 从 `torchvision` 包中实例化 `resnet34`

当我们查看 ResNet-34 时，还有更深层次的 ResNet 架构，如 ResNet-50、ResNet-101 和 ResNet-151，它们使用一个名为 `BottleneckLayer` 的不同版本的 `BasicBlock`。同样，还有许多受 ResNet 启发的其他变体，如 ResNext、Wide ResNet 等。在这本书中，我们不讨论这些个别变体，因为它们背后的核心思想是相同的。我们鼓励您阅读原始论文，以更深入地了解该主题。

### 11.2.4 PyTorch Lightning

让我们回顾一下之前讨论的数字分类问题。我们主要讨论了 LeNet 架构，并在 PyTorch 中实现了它。现在，让我们实现训练 LeNet 模型的端到端代码。我们不是使用普通的 PyTorch 来做这件事，而是使用 Lightning 框架，因为它显著简化了模型开发和训练过程。

虽然 PyTorch 有我们训练模型所需的一切，但深度学习远不止于附加层。当涉及到实际训练时，我们需要编写大量的样板代码，正如我们在之前的例子中所看到的。这包括将数据从 CPU 转移到 GPU、实现训练驱动程序等。此外，如果我们需要在多个设备/机器上扩展训练/推理，通常还需要完成另一套集成。

PyTorch Lightning 是一个提供构建模型、数据集等所需 API 的解决方案。它提供了干净的接口和要实现的钩子。底层 Lightning 框架在训练过程中的适当点调用这些钩子。其理念是 Lightning 将研究逻辑留给我们，而自动处理其余的样板代码。此外，Lightning 还带来了多 GPU 训练、16 位浮点数和无需代码更改即可在 TPU 上训练等特性。有关 PyTorch Lightning 的更多详细信息，请参阅 [`www.pytorchlightning.ai/tutorials`](https://www.pytorchlightning.ai/tutorials)。

使用 PyTorch Lightning 训练模型涉及三个主要组件：`DataModule`、`LightningModule` 和 `Trainer`。让我们看看每个组件的作用。

数据模块

`DataModule` 是一个可共享、可重用的类，它封装了处理数据所需的所有步骤。所有数据模块都必须继承自 `LightningDataModule`，它提供了可覆盖的方法。在这种情况下，我们将实现 MNIST 作为数据模块。现在，这个数据模块可以在跨越各种模型和架构的多个实验中使用。

列表 11.11 PyTorch 中 MNIST 数据模块的代码

```py
class MNISTDataModule(LightningDataModule):
    DATASET_DIR = "datasets"

    def __init__(self, transform=None, batch_size=100):
        super(MNISTDataModule, self).__init__()
        if transform is None:
            transform = transforms.Compose(
                [transforms.Resize((32, 32)),
                transforms.ToTensor()])
        self.transform = transform
        self.batch_size = batch_size

    def prepare_data(self):                                  ①
        datasets.MNIST(root = MNISTDataModule.DATASET_DIR,
            train=True, download=True)
        datasets.MNIST(root=MNISTDataModule.DATASET_DIR,
            train=False, download=True)

    def setup(self, stage=None):
        train_dataset = datasets.MNIST(
            root = MNISTDataModule.DATASET_DIR, train=True,
            download=False, transform=self.transform)
        self.train_dataset, self.val_dataset = random_split( ②
            train_dataset, [55000, 5000])

        self.test_dataset = datasets.MNIST(
            root = MNISTDataModule.DATASET_DIR, train = False,
            download = False, transform=self.transform)

    def train_dataloader(self):                              ③
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=0)

    def val_dataloader(self):                                ④
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=0)

    def test_dataloader(self):                               ⑤
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=0)

    @property
    def num_classes(self):                                   ⑥
        return 10
```

① 下载、标记并准备原始数据

② 将训练数据集分为训练集和验证集

③ 创建训练数据加载器，它提供了一个干净的接口来迭代数据集。它处理批处理、洗牌以及通过多进程获取数据，所有这些都在幕后完成。

④ 创建验证数据加载器

⑤ 创建测试数据加载器

⑥ 数据集中对象类别的数量

LightningModule

`LightningModule`本质上将所有研究代码组合成一个单一模块，使其自包含。注意`DataModule`和`LightningModule`之间的清晰分离——这使得在不同的数据集上训练/评估相同的模型变得容易。同样，不同的模型也可以轻松地在相同的数据集上训练/评估。

Lightning 模块由以下内容组成：

+   在`init`方法中定义的模型或模型系统

+   在`training_step`中定义的训练循环

+   在`validation_step`中定义的验证循环

+   在`testing_step`中定义的测试循环

+   在`configure_optimizers`中定义的优化器和调度器

让我们看看如何将 LeNet 分类器定义为 Lightning 模块。

列出 11.12 作为 Lightning 模块的 LeNet 的 PyTorch 代码
