
# 9.6 使用预训练 GAN

由于训练 GAN 是昂贵的，在本节中我们使用 PyTorch *Hub* 下载一个预训练的 GAN。Hub 是一个人们可以上传有用预训练模型的存储库，PyTorch 内置了对这些模型的下载和使用集成。我们将下载一个在更高分辨率的面部图像上训练的示例，以便比 MNIST 更有趣地查看。

首先要做的：我们从 Hub 加载所需的 GAN 模型。这是通过 `hub.load` 函数完成的，其中第一个参数是要加载的存储库，后续参数取决于存储库。在这种情况下，我们加载了一个名为 PGAN 的模型，该模型是在名人高分辨率数据集上训练的。我们还加载了 `torchvision` 包，它为 PyTorch 提供了视觉特定的扩展：

```
import torchvision 
model = torch.hub.load(’facebookresearch/pytorch_GAN_zoo:hub’, ’PGAN’, 
➥ model_name=’celebAHQ-512’, pretrained=True, useGPU=False)
```

## PGAN Hub 模型的具体细节

加载到 Hub 的代码定义了一个 `buildNoiseData` 函数，它接受我们想要生成的样本数量，并生成用于生成这么多图像的噪声向量。然后我们可以使用模型的 `test` 方法从噪声生成图像。让我们试试吧！

```
num_images = 2 
noise, _ = model.buildNoiseData(num_images) 
with torch.no_grad(): 
    generated_images = model.test(noise)
```

使用 `torchvision` 辅助函数绘制图像，你应该会看到一个合成的男性和女性：

```
    grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1), 
    ➥ scale_each=True, normalize=True) 
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())

[50]: <matplotlib.image.AxesImage at 0x7f2abd028f90>
```

![](img/CH09_UN16_Raff.png)

就像我们之前的 GAN 一样，这些图像是通过学习将噪声转换为逼真图像创建的！如果我们打印噪声值，我们会看到来自高斯分布的随机采样值（下面没有隐藏任何魔法）：

```
print(noise)

tensor([[-0.0766, 0.3599, -0.7820, {\ldots}, -1.0038, 0.5046, -0.4253],
       [ 0.5643, 0.4867, 0.2295, {\ldots}, 1.1609, -1.2194, 0.8006]])
```

### 9.6.2 插值 GAN 输出

GAN 的一个酷特性是，当你从数学上操作时，学习到的潜在值 `z` 往往表现得很好。所以如果我们取我们的两个噪声样本，我们可以在噪声向量之间进行插值（比如说，一个的 50% 和另一个的 50%），以生成图像的插值。这通常被称为 *在潜在空间中行走*：如果你有一个潜在向量，并走到第二个潜在向量的一定距离，你最终得到的是代表你原始和目标潜在值的混合体。

如果我们将此应用于我们的两个样本，男性图像会逐渐过渡到女性，头发变暗成棕色，笑容变宽。在代码中，我们从第一个图像到第二个图像在潜在空间中移动了八步，改变了每个潜在对结果贡献的比例：

```
    steps = 8 interpolated_z = [] ❶ 
    for x in torch.arange(0,steps)/float(steps)+0.5/steps: 
        z_mix = x*noise[0,:] +
        (1-x)*noise[1,:]          ❷ 
        interpolated_z.append(z_mix)

    with torch.no_grad():         ❸ 
        mixed_g = model.test(torch.stack(interpolated_z)).clamp(min=-1, max=1) 
    grid = torchvision.utils.make_grid( 
    mixed_g.clamp(min=-1, max=1), scale_each=True, 
    ➥ normalize=True)            ❹ 
    plt.figure(figsize=(15,10)) plt.imshow(grid.permute(1, 2, 0).cpu().numpy())

[52]: <matplotlib.image.AxesImage at 0x7f2af48f42d0>
```

❶ 保存插值图像的位置

❷ 取第一个潜在向量的步长/步数和第二个的 (1 - 步长/步数)，也就是行走

❸ 从插值中生成图像

❹ 当可视化生成的输出时，看起来像是一种混合。

![](img/CH09_UN17_Raff.png)

我们甚至可以更进一步。如果我们愿意对我们的数据集中的几个实例进行标记，我们可以提取具有语义意义的向量。图 9.9 展示了这在高层次上可能的工作方式。

![](img/CH09_F09_Raff.png)

图 9.9 生成图像的示例以及你可能能找到的语义向量类型。左侧显示的是 G 的原始潜在变量及其相关输出。通过添加这些语义向量，我们可以改变生成的图像。这些被称为潜在向量，因为我们没有告诉 GAN 它们是什么：它们对 GAN 是隐藏的，GAN 必须从数据模式中自己学习它们。

这基本上意味着我们可以找到一个向量 `z[smile]`，我们可以将其添加到任何其他潜在向量中，使其某人微笑，或者从另一个潜在向量中减去以移除微笑。好事是，我们 *从未告诉 G 关于任何这些潜在属性的信息*。G 学习自己这样做，如果我们能发现语义向量，我们就可以进行这些修改！为此，让我们首先生成一些随机的图像来使用：

```
    set_seed(3)                          ❶ 
    noise, _ = model.buildNoiseData(8*4) ❷ 
    with torch.no_grad(): 
        generated_images = model.test(noise) 
    grid = torchvision.utils.make_grid( 
    generated_images.clamp(min=-1, max=1), 
    ➥ scale_each=True, normalize=True)  ❸ 
    plt.figure(figsize=(13,6)) 
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())

[53]: <matplotlib.image.AxesImage at 0x7f2abc505710>
```

❶ 获取一致的结果

❷ 通过添加性别向量到我们的原始潜在向量来生成新的图像

❸ 可视化它们

![图片](img/CH09_UN18_Raff.png)

### 9.6.3 标记潜在维度

现在我们有 32 个示例面孔。如果我们能识别出我们对每个生成的图像关心的某些属性，我们就可以对它们进行标记，并尝试学习哪些噪声控制了输出的不同方面。³ 让我们识别出男性和女性，以及那些微笑或不微笑的人。我为每个属性创建了一个数组，对应于每个图像，因此我们有 32 个“男性”标签和 32 个“微笑”标签。本质上，我们正在为我们关心的属性创建自己的标签 `y`，但我们将能够用很少的标记示例提取这些语义⁴ 向量。G 已经独立学习了概念：

```
male = [0, 1, 0, 0, 1, 0, 0, 0,                      ❶ 
        1, 1, 1, 1, 0, 1, 0, 0,                      ❶ 
        1, 0, 0, 1, 0, 0, 1, 0,                      ❶ 
        0, 0, 0, 0, 0, 0, 1, 0]                      ❶ 
smile = [1, 1, 0, 0, 1, 0, 1, 1,                     ❶ 
         0, 0, 0, 0, 0, 1, 0, 0,                     ❶ 
         1, 0, 1, 0, 1, 1, 1, 1,                     ❶ 
         0, 0, 1, 1, 0, 0, 0, 1]                     ❶ 
male = np.array(male, dtype=np.bool) 
smile = np.array(smile, dtype=np.bool)

male = torch.tensor(np.expand_dims(male, axis=-1))   ❷ 
smile = torch.tensor(np.expand_dims(smile, axis=-1)) ❷
```

❶ 为哪些图像显然是男性或微笑的进行标记。我手动检查生成的图像以创建这些列表。

❷ 将形状从 (32) 转换为 (32, 1)

#### 计算语义向量

现在我们想计算平均男性向量和平均非男性向量。我们对微笑也做同样的事情，所以让我们定义一个简单的函数，该函数使用二进制标签 `male` 和 `smile` 来提取表示差异的向量。我们生成的标签越多，结果越好：

```
def extractVec(labels, noise): 
    posVec = torch.sum(noise*labels, axis=0)/torch.sum(labels)   ❶ 
    negVec = torch.sum(noise* labels, axis=0)/torch.sum( labels) ❷ 
    return posVec-negVec                                         ❸
```

❶ 获取所有具有类别标签 0 的平均值

❷ 所有具有类别标签 1 的平均值

❸ 通过取平均值之间的差异来近似潜在概念之间的差异

#### 使用语义向量操纵图像

现在我们可以使用 `extractVec` 函数提取一个男性性别向量。如果我们将其添加到任何潜在向量 `z` 中，我们就会得到一个新的潜在向量，使其更具男性特征。生成的图像中的其他所有内容通常应保持不变，例如背景、发色、头部位置等。让我们试试：

```
    gender_vec = extractVec(male, noise)                ❶ 
    with torch.no_grad():
        generated_images = model.test(noise+gender_vec) ❷

    grid = torchvision.utils.make_grid( 
    ➥ generated_images.clamp(min=-1, max=1), 
    ➥ scale_each=True, normalize=True)                 ❸ 
    plt.figure(figsize=(13,6)) 
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())

[56]: <matplotlib.image.AxesImage at 0x7f2abc4f5850>
```

❶ 提取性别向量

❷ 通过将性别向量添加到我们的原始潜在向量来生成新的图像

❸ 绘制结果

![图片](img/CH09_UN19_Raff.png)

总体来说，结果相当不错。虽然不是完美无缺，但我们并没有使用很多例子来发现这些向量。我们还可以减去这个性别向量，从图像中去除男性特征。在这种情况下，我们会让每个图像看起来更女性化。以下代码通过简单地改变`+`为`-`来实现这一点：

```
    with torch.no_grad(): 
        generated_images = model.test(noise-gender_vec) 
    grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1), 
    ➥ scale_each=True, normalize=True) 
    plt.figure(figsize=(13,6)) 
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())

[57]: <matplotlib.image.AxesImage at 0x7f2abd67d590>
```

![](img/CH09_UN20_Raff.png)

也许你希望每个人都快乐！我们可以使用我们提取的`smile`向量来做同样的事情，让每个人笑得更多：

```
    smile_vec = extractVec(smile, noise) 
    with torch.no_grad():
        generated_images = model.test(noise+smile_vec) 
    grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1), 
    ➥ scale_each=True, normalize=True) 
    plt.figure(figsize=(13,6)) 
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())

[58]: <matplotlib.image.AxesImage at 0x7f2ab47d0750>
```

![](img/CH09_UN21_Raff.png)

## 9.7 深度学习中的伦理

在第四章中，我们简要地偏离了主题，讨论了理解我们如何建模世界以及使用这些模型如何影响他人感知世界的重要性。现在我们已经了解了 GANs，这是一个值得再次讨论的话题。

GANs 学习到的语义概念基于数据，并不一定代表世界运作的真实情况。例如，大多数人都会认同并表现出自己是男性或女性，这在我们的数据中有所体现，因此 GAN 学习到男性和女性之间的一种相对线性的关系，作为一个统一的光谱。但这并不符合许多不适合仅限于男性或女性类别的人的现实。因此，在使用 GAN 对图像进行性别操纵之前，你应该三思而后行。

对于你构建的任何系统（机器学习或其他），你应该问自己一些简单的问题：它将如何影响使用该系统的多数人？它将如何影响使用该系统的少数人？是否有受益或受损的人，这些人是否应该得到这种利益或损害？你的部署是否会以积极或消极的方式改变人们的行为，或者甚至被有良好意图的用户滥用？通常，从微观和宏观层面思考*可能出错的地方*。

我并不是试图向你强加任何哲学或道德信念体系。伦理是一个非常复杂的话题，我无法在一个章节的小节中公正地处理它（而且这也不是本书的主题）。但是，随着深度学习的发展，我们能够自动化许多不同的事情。这可能会给社会带来好处，让人们从繁重和耗人的工作中解放出来，但它也可能以新的高效规模放大和复制不希望存在的差异。因此，我希望你有所警觉，并开始训练自己思考这些类型的考虑。以下是一些你可以使用的资源链接，以帮助你加深对这些问题的理解：

+   首先，一篇较老的文章，它从更广泛的角度捕捉了许多这些担忧：B. Friedman 和 H. Nissenbaum，“计算机系统中的偏见”，*ACM Trans. Inf. Syst.* 第 14 卷，第 3 期，第 330-347 页，1996 年，[`doi.org/10.1145/230538.230561`](https://doi.org/10.1145/230538.230561)，[`nissenbaum.tech.cornell.edu/papers/Bias%20in%20Computer%20Systems.pdf`](https://nissenbaum.tech.cornell.edu/papers/Bias%20in%20Computer%20Systems.pdf)。

+   我之前提到了凯特·克劳福德（www.katecrawford.net），我还推荐蒂姆尼特·盖布卢（[`scholar.google.com/citations?user=lemnAcwAAAAJ`](https://scholar.google.com/citations?user=lemnAcwAAAAJ)）和鲁本·宾斯（[`www.reubenbinns.com/`](https://www.reubenbinns.com/)）。

+   尤其是鲁本·宾斯的论文“机器学习中的公平性：政治哲学的启示”，R. Binns，*机器学习研究会议论文集* 第 81 卷，第 1-11 页，2018 年，[`proceedings.mlr.press/v81/binns18a/binns18a.pdf`](http://proceedings.mlr.press/v81/binns18a/binns18a.pdf)。它易于理解，并从不同的角度阐述了棘手的公平性概念意味着什么。

你不需要掌握这些知识就能成为一名优秀的深度学习研究人员或实践者，但你*应该*学会思考和考虑这些问题。如果你知道得足够多，仅仅说，“这可能会是一个问题；我们应该寻求一些指导”，你比许多今天的实践者更有准备。
