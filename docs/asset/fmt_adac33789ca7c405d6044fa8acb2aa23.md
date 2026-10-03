

# 第三章. 超越基础：检测图像中的特征

在第二章中，你学习了如何通过创建一个简单的神经网络来入门计算机视觉，该神经网络将 Fashion MNIST 数据集的输入像素与 10 个标签相匹配，每个标签代表一种（或一类）服装。虽然你创建了一个在检测服装类型方面相当不错的网络，但存在一个明显的缺点。你的神经网络是在小单色图像上训练的，每个图像只包含一件服装，而且每件服装都位于图像的中心。

要将模型提升到下一个层次，你需要它能够检测图像中的*特征*。例如，如果我们能够将图像过滤到构成元素，而不是仅仅查看图像中的原始像素，那会怎么样？用这些元素匹配，而不是原始像素，将有助于模型更有效地检测图像内容。例如，考虑我们在上一章中使用的 Fashion MNIST 数据集。在检测鞋子时，神经网络可能被图像底部的许多暗色像素簇激活，它将看到这是鞋底。但如果鞋子没有居中且填满画面，这种逻辑就不成立了。

检测特征的一种方法来自摄影和图像处理方法，你可能已经熟悉。如果你曾经使用过 Photoshop 或 GIMP 之类的工具来锐化图像，你就已经使用了一种作用于图像像素的数学滤波器。这些滤波器所做的工作的另一个词是*卷积*，通过在神经网络中使用这些滤波器，你将创建一个*卷积神经网络*（CNN）。

在本章中，你将首先学习如何使用卷积来检测图像中的特征。然后，你将深入了解基于图像内部特征进行图像分类。我们还将探索图像增强以获取更多特征，以及迁移学习，即利用他人学习到的现有特征，然后我们将简要探讨通过使用 dropout 来优化你的模型。

## 卷积

*卷积*简单来说是一个权重滤波器，它通过将像素与其邻居相乘来得到像素的新值。例如，考虑 Fashion MNIST 中的踝靴图像及其像素值（见图 3-1）。

![](img/aiml_0301.png)

###### 图 3-1\. 女式踝靴与卷积

如果我们查看选择区域中间的像素，我们可以看到它的值是 192。（回想一下，Fashion MNIST 使用像素值为 0 到 255 的单色图像。）上面的像素值是 0，直接上面的像素值是 64，等等。

如果我们然后在下面的原始值中定义一个 3 × 3 的网格过滤器，就像下面所示，我们可以通过计算它的新值来转换该像素。我们通过将网格中每个像素的当前值乘以过滤器网格中相同位置的值，然后求和来实现这一点。这个总和将是当前像素的新值，然后我们重复这个计算过程，直到图像中的所有像素。

因此，在这种情况下，尽管选择中心像素的当前值为 192，我们通过以下方式计算应用过滤器后的新值：

```py
new_val = (–1 * 0) + (0 * 64) + (–2 * 128) + 
     (.5 * 48) + (4.5 * 192) + (–1.5 * 144) + 
     (1.5 * 142) + (2 * 226) + (–3 * 168)
```

结果等于 577，这将是像素的新值。对图像中的每个像素重复此过程将给我们一个过滤后的图像。

现在，让我们考虑在一个更复杂的图像上应用过滤器的影响：具体来说，是 SciPy 中内置的用于简单测试的[ascent image](https://oreil.ly/wP8TE)。这是一张 512 × 512 的灰度图像，显示了两个人在爬楼梯。

使用左侧带负值、右侧带正值和中间为零的过滤器，最终会从图像中移除大部分信息，除了垂直线（参见图 3-2）。

![](img/aiml_0302.png)

###### 图 3-2\. 使用过滤器提取垂直线

类似地，对过滤器的小幅调整可以强调水平线（参见图 3-3）。

![](img/aiml_0303.png)

###### 图 3-3\. 使用过滤器提取水平线

这些例子还表明，图像中的信息量减少了。因此，我们可能可以*学习*一组过滤器，将图像减少到特征，而这些特征可以像以前一样与标签匹配。以前，我们学习了用于匹配输入到输出的神经元参数，同样，我们可以随着时间的推移学习最佳的过滤器来匹配输入到输出。

当我们将卷积与池化结合使用时，我们可以在保持特征的同时减少图像中的信息量。我们将在下一节中探讨这一点。

## 池化

*池化*是在保持图像内容语义的同时消除图像中像素的过程。它最好通过视觉来解释。图 3-4 展示了*最大池化*的概念。

![](img/aiml_0304.png)

###### 图 3-4\. 最大池化的示例

在这种情况下，将左边的框视为单色图像中的像素。我们将它们分成 2 × 2 的数组，因此在这种情况下，16 个像素被分成四个 2 × 2 的数组。这些数组被称为*池*。

然后，我们从每个组中选择*最大*值，并将它们重新组装成一个新的图像。因此，左边的像素减少了 75%（从 16 变为 4），每个池中的最大值构成了新的图像。图 3-5 显示了从图 3-2 的上升版本，在应用最大池化后，垂直线得到了增强。

![aiml_0305.png](img/aiml_0305.png)

###### 图 3-5\. 应用垂直滤波和最大池化后的上升

注意过滤后的特征不仅被保留，而且得到了进一步的增强。同时，图像大小从 512 × 512 变为 256 × 256——变成了原始大小的四分之一。

###### 注意

存在其他池化方法。这些包括*最小池化*，它从池中取最小的像素值，以及*平均池化*，它取池中的整体平均值。

## 实现卷积神经网络

在第二章中，你创建了一个用于识别时尚图像的神经网络。为了方便，这里提供了定义模型的代码：

```py
# Define the model
class FashionMNISTModel(nn.Module):
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = FashionMNISTModel()

# Define the loss function and optimizer
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())

# Train the model
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  
                    [{current:>5d}/{size:>5d}]")

# Training process
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_function, optimizer)
print("Done!")
```

要将此转换为 CNN，你只需在当前线性层之上使用我们的模型定义中的卷积层。你还将添加池化层。

要实现卷积层，你将使用`nn.Conv2D`类型。它接受作为参数的层中要使用的卷积数量、卷积的大小、激活函数等。

例如，这里有一个使用此类型的卷积层：

```py
nn.Conv2d(1, 64, kernel_size=3, padding=1)
```

在这个例子中，我们希望层学习`64`个卷积。它将随机初始化它们，并且随着时间的推移，它将学习最佳的过滤器值，以匹配输入值和它们的标签。`kernel_size = 3`表示过滤器的大小。之前我们展示了 3 × 3 的过滤器，这就是我们在这里指定的。3 × 3 的过滤器是最常见的过滤器大小。你可以根据需要更改它，但通常你会看到像 5 × 5 或 7 × 7 这样的奇数轴，因为过滤器会从图像的边缘移除像素，就像你稍后将要看到的那样。

这是在神经网络中使用池化层的方法。你通常会在卷积层之后立即这样做：

```py
nn.MaxPool2d(kernel_size=2, stride=2)
```

在之前的图 3-4 示例中，我们将图像分割成 2 × 2 的池，并从每个池中选取最大值。然而，我们可以使用这里看到的参数来定义池的大小。`kernel_size=2`参数表示我们的池是 2 × 2 的，而`stride=2`参数意味着过滤器将跳过两个像素以获取下一个池。

现在，让我们探索定义 Fashion MNIST 模型的完整代码：

```py
# Define the CNN model
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2))  # Output: 64 x 6 x 6

        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), –1)  # Flatten the output
        out = self.fc1(out)
        out = self.fc2(out)
        return out
```

在这里，我们看到该类有两个函数，一个用于初始化，另一个将在每个训练周期的正向传递期间被调用。

`init`简单地定义了我们的神经网络中每一层的样子。第一层（`self.layer1`）将接受一维输入，有`64`个卷积，`kernel_size`为`3`，`padding`为`1`。然后它将在最大池化之前对输出进行 ReLU 操作。

下一层（`self.layer2`）将接受前一层输出的 64 个卷积，然后输出自己的`64`个卷积，在 ReLU 和最大池化之前。由于`MaxPool`将图像大小减半，其输出现在将是`64 × 6 × 6`。

然后，数据被送入下一层（`self.fc1`，其中`fc`代表*全连接*），输入是前一层输出的形状。输出是 128，这与我们在第二章中用于深度神经网络（DNN）的神经元数量相同。

最后，这 128 个神经元被送入最终的层（`self.fc1`），有 10 个输出——代表 10 个类别。

###### 注意

在 DNN 中，我们在将其送入第一个`Dense`层之前，先通过一个`Flatten`层对输入进行处理。在这里，我们失去了输入层——相反，我们只是指定了 1-D 输入形状。注意，在第一个`Linear`层之前，经过卷积和池化后，数据将被展平。

然后，我们在`forward`函数中堆叠这些层。我们可以看到我们得到数据`x`，通过`layer1`得到`out`，然后将其传递给`layer2`以得到新的`out`。在这个阶段，我们有了我们学习到的卷积，但在将它们加载到`Linear`层`fc1`和`fc2`之前，我们需要将它们展平。`out = out.view(out.size(0), -1)`实现了这一点。

如果我们用与我们在第二章中训练网络时相同的数据和相同的 50 个 epoch 来训练这个网络，我们将看到它运行得很好。我们可以在测试集上轻松达到 91%的准确率：

```py
Train Epoch: 44 -- Loss: 0.091689
Train Epoch: 45 -- Loss: 0.066864
Train Epoch: 46 -- Loss: 0.061322
Train Epoch: 47 -- Loss: 0.056557
Train Epoch: 48 -- Loss: 0.039695
Train Epoch: 49 -- Loss: 0.056213
Accuracy of the network on the 10000 test images: 91.31%
```

因此，我们可以看到向神经网络添加卷积确实增加了其分类图像的能力。接下来，让我们看看图像在网络中的旅程，这样我们可以更深入地理解这个过程为什么有效。

###### 注意

如果您正在使用我 GitHub 上的配套代码，您会注意到我大量使用了`model.to(device)`。在 PyTorch 中，如果可用加速器，您可以使用此命令请求模型及其数据使用加速器。

## 探索卷积网络

使用 torchsummary 库，您可以检查您的模型。当您在我们一直在工作的 Fashion MNIST 卷积网络上运行它时，您会看到类似这样的内容：

```py
from torchsummary import summary
model = FashionCNN().to(device) 
summary(model, input_size=(1, 28, 28))  # (Channels, Height, Width)
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [–1, 64, 28, 28]             640
              ReLU-2           [–1, 64, 28, 28]               0
         MaxPool2d-3           [–1, 64, 14, 14]               0
            Conv2d-4           [–1, 64, 12, 12]          36,928
              ReLU-5           [–1, 64, 12, 12]               0
         MaxPool2d-6             [–1, 64, 6, 6]               0
            Linear-7                  [–1, 128]         295,040
            Linear-8                   [–1, 10]           1,290
================================================================
Total params: 333,898
Trainable params: 333,898
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.02
Params size (MB): 1.27
Estimated Total Size (MB): 2.30
```

让我们先看看“输出形状”列，以了解这里发生了什么。我们的第一层将有 28 × 28 的图像，并对其应用 64 个滤波器。但由于我们的滤波器是 3 × 3，图像周围通常会有一个像素的边界丢失，这会减少我们的整体信息到 26 × 26 像素。然而，因为我们使用了 `padding=1` 参数，图像被人工膨胀到 30 × 30，这意味着其输出将是正确的 28 × 28，并且不会丢失任何信息。

如果你没有对图像进行填充，你最终会得到 图 3-6 中的结果。如果我们把每个方框看作图像中的一个像素，第一个可能的滤波器从第二行第二列开始。同样的事情会在图的右侧和底部发生。

![](img/aiml_0306.png)

