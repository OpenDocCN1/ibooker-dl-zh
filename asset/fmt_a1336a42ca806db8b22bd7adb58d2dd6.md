
###### 图 3-6\. 运行滤波器时丢失像素

因此，一个形状为 *a* × *b* 像素的图像经过 3 × 3 滤波器处理后，其形状将变为 (*a* – 2) × (*b* – 2) 像素。同样，5 × 5 滤波器会使它变为 (*a* – 4) × (*b* – 4)，依此类推。由于我们使用的是 28 × 28 的图像和 3 × 3 滤波器，我们的输出现在将是 26 × 26。但是，因为我们已经将图像填充到 30 × 30（再次，为了防止信息丢失），所以输出现在是 28 × 28。

之后，池化层将是 2 × 2，因此图像在每个轴上的大小将减半，然后变为 14 × 14。下一个卷积层不使用填充，因此它将进一步减少到 12 × 12，下一个池化将输出 6 × 6。

因此，当图像通过两个卷积层后，结果将是许多 6 × 6 的图像。有多少个？我们可以在“参数数量”列中看到。

每个卷积都是一个 3 × 3 滤波器，加上一个偏差。记得之前，在我们的密集层中，每个层都是 *y* = *wx* + *b*，其中 *w* 是我们的参数（即权重）和 *b* 是我们的偏差？这个情况非常相似，只是因为滤波器是 3 × 3，所以有 9 个参数要学习。鉴于我们定义了 64 个卷积，我们将有 640 个总参数。（每个卷积有 9 个参数加上一个偏差，总共 10 个，共有 64 个。）

“ReLU 和 MaxPooling”层不会学习任何东西；它们只是减少图像的大小，因此那里没有学习的参数——因此，报告为 0。

下一个卷积层有 64 个滤波器，但每个滤波器都是乘以前面的 64 个滤波器，每个滤波器有 9 个参数。我们在每个新的 64 个滤波器上都有一个偏差，所以我们的参数数量应该是 (64 × (64 × 9)) + 64，这给我们带来了网络需要学习的 36,928 个参数。

如果这让你感到困惑，试着将第一层的卷积数量改为其他数值——例如，10。你会发现第二层的参数数量变为 5,824，这是 (64 × (10 × 9)) + 64)。

到我们完成第二次卷积时，我们的图像是 6 × 6，我们有 64 个这样的图像。如果我们乘以这个数字，我们将有 1,600 个值，我们将把这些值输入到一个有 128 个神经元的密集层中。每个神经元都有一个权重和偏差，我们将有 128 个这样的神经元，所以网络将学习的参数数量是((6 × 6 × 64) × 128) + 128，这给我们 295,040 个参数。

然后，我们的最终密集层将有 10 个神经元，它将接收之前 128 个神经元的输出，所以学习的参数数量将是(128 × 10) + 10，即 1,290。

参数总数将是所有这些的总和：333,898。

训练这个网络需要我们学习最佳的 333,898 个参数集合，以匹配输入图像和它们的标签。由于参数更多，这是一个更慢的过程，但正如我们从结果中可以看到的，它也构建了一个更精确的模型！

当然，使用这个数据集，我们仍然存在限制，即图像是 28 × 28，单色，并且居中。所以接下来我们将看看如何使用卷积来探索一个更复杂的数据集，该数据集包含马和人类的彩色图片，并且我们将尝试让模型确定图像中包含的是马还是人。在这种情况下，主题不总是像 Fashion MNIST 那样居中在图像中，因此我们不得不依赖卷积来识别区分特征。

# 构建 CNN 以区分马和人

在本节中，我们将探索比 Fashion MNIST 分类器更复杂的场景。我们将扩展我们对卷积和 CNN 的了解，尝试对图像内容进行分类，其中特征的位置不总是相同。我为此目的创建了“马或人”数据集。

## “马或人”数据集

[本节的数据集](https://oreil.ly/8VXwy)包含超过一千张 300 × 300 像素的图像。大约一半的图像是马，另一半是人——并且它们都以不同的姿势呈现。您可以在图 3-7 中看到一些示例。

![](img/aiml_0307.png)

###### 图 3-7\. 马和人

如您所见，主题有不同的方向和姿势，图像构图也各不相同。以两匹马为例，它们的头部方向不同，其中一张图像被放大（显示整个动物），而另一张图像被缩小（只显示头部和身体的一部分）。同样，人类被不同的光线照亮，有不同的肤色，姿势也不同。男人把手放在臀部，而女人把手伸展开。图像还包含背景，如树木和海滩，因此分类器必须确定图像中哪些部分是重要的特征，这些特征决定了马是马，人是人，而不受背景的影响。

虽然之前的预测*y* = 2*x* – 1 或对小型单色服装图像的分类可能通过传统编码实现，但很明显，这个例子要困难得多，而且你已经跨过了机器学习成为解决问题必需品的界限。

一个有趣的旁注是，这些图像都是计算机生成的。理论是，在 CGI 马图像中发现的特征应该适用于真实图像，你将在本章后面看到这是如何工作的。

## 处理数据

到目前为止你一直在使用的 Fashion MNIST 数据集包含标签，并且每个图像文件都有一个与之关联的标签详细信息文件。许多基于图像的数据集没有这个，"马或人"也不例外。没有标签，图像被分类到每个类型的子目录中，而在 PyTorch 的 `DataLoader` 中，你可以使用这种结构来*自动*为图像分配标签。

首先，你需要确保你的目录结构有一组命名子目录，每个子目录都是一个标签。例如，“马或人”数据集作为一组 ZIP 文件提供，其中一个包含训练数据（1,000+图像），另一个包含验证数据（256 图像）。当你下载并解压缩到本地目录用于训练和验证时，你需要确保它们具有如图图 3-8 所示的文件结构。

这是获取训练数据并将其提取到适当命名的子目录中的代码，如图图 3-8 所示：

```py
import urllib.request
import zipfile

url = "https://storage.googleapis.com/learning-datasets/
                                            horse-or-human.zip"
file_name = "horse-or-human.zip"
training_dir = 'horse-or-human/training/'
urllib.request.urlretrieve(url, file_name)

zip_ref = zipfile.ZipFile(file_name, 'r')
zip_ref.extractall(training_dir)
zip_ref.close()
```

![](img/aiml_0308.png)

###### 图 3-8\. 确保图像在命名子目录中

此代码仅下载训练数据的 ZIP 文件，并将其解压缩到*horse-or-human/training*目录中。（我们很快就会处理验证数据的下载。）这是包含图像类型子目录的父目录。

现在，要使用`DataLoader`，我们只需使用以下代码：

```py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformations
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the datasets
train_dataset = datasets.ImageFolder(root=training_dir, 
                                     transform=transform)
val_dataset = datasets.ImageFolder(root=validation_dir, 
                                   transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

首先，我们创建一个名为`transform`的`transforms`对象实例。这将确定我们修改图像的规则。它将图像调整大小到 150 × 150，然后将其归一化到一个张量中。请注意，原始图像实际上是 300 × 300，但为了学习目的加快训练速度，我已经将它们调整大小到 150 × 150。

然后，我们指定`dataset`对象为`datasets.ImageFolder`类型，并将它们指向所需的目录，这样就会在应用转换的同时从该目录生成用于训练过程的图像。训练目录是`training_dir`，验证目录是`validation_dir`，如前所述。

## “马或人”的 CNN 架构

与 Fashion MNIST 数据集相比，这个数据集有几个主要的不同之处，在设计用于分类图像的架构时，你必须考虑到它们。首先，图像要大得多——150 × 150 像素，所以可能需要更多的层。其次，图像是全彩色的，而不是灰度的，所以每个图像将有三通道而不是一个。第三，只有两种图像类型，因此我们可以实际上只用*一个*输出神经元来对它们进行分类。为此，我们将该神经元的值驱动到其中一个标签的 0，并驱动到另一个标签的 1。`sigmoid`函数非常适合这个过程，可以将值驱动到这两个极端之一。你可以在`forward`函数的底部看到这一点：

```py
class HorsesHumansCNN(nn.Module):
    def __init__(self):
        super(HorsesHumansCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 18 * 18, 512)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, 1)  
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(–1, 64 * 18 * 18)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)  # Use sigmoid to output probabilities
        return x
```

这里有几个需要注意的地方。首先，看看最顶层。我们定义了 16 个过滤器，每个过滤器的`kernel_size`为 3，但输入形状是 3。记住这是因为我们的输入图像是彩色的：有三个通道，而不是我们之前使用的单色 Fashion MNIST 数据集的一个通道。

在另一端，注意输出层中只有一个神经元。这是因为我们使用的是二元分类器，如果我们用 sigmoid 函数激活它，我们就可以只用一个神经元得到二元分类。sigmoid 函数的目的是将一组值驱动到 0，另一组驱动到 1，这对于二元分类来说非常完美。

接下来，注意我们如何堆叠更多的卷积层。我们这样做是因为我们的图像源相当大，我们希望随着时间的推移，拥有许多较小的图像，每个图像都突出显示了一些特征。如果我们查看`summary`的结果，我们会看到这一点：

```py
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [–1, 16, 150, 150]             448
         MaxPool2d-2           [–1, 16, 75, 75]               0
            Conv2d-3           [–1, 32, 75, 75]           4,640
         MaxPool2d-4           [–1, 32, 37, 37]               0
            Conv2d-5           [–1, 64, 37, 37]          18,496
         MaxPool2d-6           [–1, 64, 18, 18]               0
            Linear-7                  [–1, 512]      10,617,344
           Dropout-8                  [–1, 512]               0
            Linear-9                    [–1, 1]             513
================================================================
Total params: 10,641,441
Trainable params: 10,641,441
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.26
Forward/backward pass size (MB): 5.98
Params size (MB): 40.59
Estimated Total Size (MB): 46.83
----------------------------------------------------------------
```

注意，当数据通过所有卷积和池化层后，最终会变成 18 × 18 个项。理论上是这样的，这些将是相对简单的激活特征图，因为它们将只包含 324 个像素。然后我们可以将这些特征图传递给密集神经网络，以匹配适当的标签。

当然，这使得这个网络比之前的网络有更多的参数，所以训练会慢一些。使用这种架构，我们将学习超过 1000 万个参数。

###### 小贴士

这一部分的代码，以及本书中的许多其他地方，可能需要你导入 Python 库。要找到正确的导入，你可以查看[本书的仓库](https://github.com/lmoroney/PyTorch-Book-FIles)。

为了训练网络，我们需要使用损失函数和优化器来编译它。在这种情况下，损失函数可以是`BCELoss`，其中`BCE`代表*二元交叉熵*。正如其名所示，因为在这个场景中只有两个类别，这是一个为此设计的损失函数。对于`optimizer`，我们可以继续使用之前使用的相同的`Adam`。以下是代码：

```py
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

我们然后以通常的方式进行训练：

```py
def train_model(num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()  
            optimizer.zero_grad()
            outputs = model(images).view(–1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
```

需要注意的一点是，由于二元交叉熵，标签被转换为`floats`，最终输出节点的值将是一个浮点值。

在仅仅 15 个 epoch 之后，这个架构在训练集上就给我们带来了非常令人印象深刻的 95%以上的准确率。当然，这仅仅是基于训练数据，这种性能并不能表明网络在之前未见过数据上的潜在性能。

接下来，我们将查看添加验证集并衡量其性能，以给我们一个很好的指示，了解这个模型在现实生活中的表现可能如何。

## 将验证添加到“马或人”数据集中
