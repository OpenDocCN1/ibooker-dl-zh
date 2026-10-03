
现在我们可以定义我们的 Faster R-CNN 模型。我们给它提供骨干网络，并告诉它存在多少个类别以及如何归一化图像（如果你的图像值在 `[0,1]` 范围内，对于大多数图像来说，`0.5` 的平均值和 `0.23` 的偏差是好的默认值）。我们还告诉 Faster R-CNN 最小和最大的图像尺寸。为了使这个运行更快，我们将它们都设置为唯一的图像尺寸 `100`。但对于真实数据，这是用来尝试在多个尺度上检测对象的；这样它就可以处理近距离或远距离的对象。然而，这需要更多的计算来运行，甚至更多的训练来训练。

注意：在现实问题中，应该如何设置 Faster R-CNN 的最小和最大尺寸？这里有一个经验法则：如果对于人类来说太小而无法完成，那么网络可能也无法完成。在你的数据中，尝试以不同的图像分辨率查找你关心的对象。如果你能找到你的对象的最小分辨率为 `256 × 256`，那么这是一个好的最小尺寸。如果你需要将图像放大到 `1024 × 1024` 以发现对象，那么这是一个好的最大尺寸。

截至 PyTorch 1.7.0，当使用自己的骨干网络时，你还必须指定一些关于 RPN 和 RoI 网络的信息。这是通过 `AnchorGenerator` 和 `MultiScaleRoIAlign` 对象分别完成的。`AnchorGenerator` 控制 RPN 生成的提议数量，以不同的长宽比（例如，`1.0` 是一个正方形，`0.5` 是一个长宽比为 `2:1` 的矩形，`2` 是一个长宽比为 `1:2` 的矩形）生成，以及那些预测应该有多高（即你的对象可能有多大或小？对于现实世界问题，你可能使用 `16` 到 `512` 像素）。`MultiScaleRoIAlign` 需要我们告诉 `FasterRCNN` 骨干网络的哪个部分提供特征图（它支持多个特征图作为一项高级功能），RoI 网络的大小，以及如何处理从 RPN 预测的分数像素位置。以下代码将这些内容组合在一起：

```
anchor_generator = AnchorGenerator(sizes=((32),), ❶
➥ aspect_ratios=((1.0),)) 

roi_pooler = torchvision.ops.MultiScaleRoIAlign(  ❷
➥ featmap_names=[’0’], output_size=7, 
➥ sampling_ratio=2) 

model = FasterRCNN(backbone, num_classes=10,      ❸ 
➥ image_mean = [0.5], image_std = [0.229], 
➥ min_size=100, max_size=100, 
➥ rpn_anchor_generator=anchor_generator, 
➥ box_roi_pool=roi_pooler)
```

❶ 应该生成多少个提议 `k`？每个长宽比都会是一个，并且这个过程会针对多个图像尺寸重复进行。为了使这个运行更快，我们告诉 PyTorch 只查找大小为 `32 × 32` 的正方形图像。

❷ 告诉 PyTorch 使用骨干网络的最终输出作为特征图（`[]`）；使用自适应池化到 `7 × 7` 网格（`output_size=7`）。`sampling_ratio` 的命名不佳：它控制当预测分数像素位置时 RoI 如何从特征图中抓取切片的细节（例如，`5.8` 而不是 `6`）。我们不会深入这些低级细节；对于大多数工作，`2` 是一个合理的默认值。

❸ 创建 `FasterRCNN` 对象。我们给它提供骨干网络、类别数量、处理图像的最小和最大尺寸（我们知道所有我们的图像都是 `100` 像素），从图像中减去的平均值和标准差，以及锚生成（RPN）和 RoI 对象。

## 实现一个 Faster R-CNN 训练循环

由于对张量列表和字典列表的不寻常使用，我们无法使用我们的标准 `train_network` 函数来处理这种情况。因此，我们编写了一个最小的训练循环来为我们完成这项训练。主要技巧是将每个列表（输入和标签）中的每个项目移动到我们想要使用的计算设备上。这是因为 `.to(device)` 方法仅存在于 PyTorch `nn.Module` 类中，Python 的标准列表和字典没有这些功能。幸运的是，我们早期定义了一个 `moveTo` 函数，它执行此操作并且适用于列表和字典。

第二个奇怪之处在于，`FasterRCNN` 对象在训练模式和评估模式下的行为不同。在训练模式下，它期望将标签与输入一起传递，以便它可以计算每个预测的损失。它还返回每个单独预测损失的列表，而不是单个标量。因此，我们需要将这些单个损失全部加起来以获得最终的损失总和。以下代码显示了用于训练一个 `FasterRCNN` 周期的简单循环：

```
model = model.train() model.to(device) optimizer =
torch.optim.AdamW(model.parameters())

for epoch in tqdm(range(1), desc="Epoch", disable=False):
    running_loss = 0.0 
    for inputs, labels in tqdm(train_loader, desc="Train Batch", 
    ➥ leave=False, disable=False): 
        inputs = moveTo(inputs, device)      ❶ 
        labels = moveTo(labels, device) 

        optimizer.zero_grad() losses =
        model(inputs, labels)                ❷ 
        loss = 0                             ❸ 
        for partial_loss in losses.values(): 
        loss += partial_loss 
        loss.backward()                      ❹

        optimizer.step() 

        running_loss += loss.item()
```

❶ 将批次移动到我们使用的设备

❶ RCNN 需要 `model(inputs, labels)`，而不仅仅是 `model(inputs)`。

❶ 计算损失。RCNN 给我们一个损失列表来累加。

❶ 按照常规进行。

这就是使用 PyTorch 提供的 Faster R-CNN 实现所需的所有步骤。现在让我们看看它的表现如何。首先，将模型设置为 `eval` 模式，这将改变 Faster R-CNN 实现处理输入和返回输出的方式：

```
model = model.eval() 
model = model.to(device)
```

接下来，让我们快速从测试数据集中获取一个项目并看看它是什么样子。在这种情况下，我们看到有三个对象，`8`、`0` 和 `4`：

```
x, y = test_data[0] 
print(y)                                     ❶

{'boxes': tensor([[31., 65., 59., 93.],
         [10., 36., 38., 64.],
         [64., 24., 92., 52.]]), 'labels': tensor([8, 0, 4])\}
```

❶ 这是我们要获取的地面实况。

让我们进行一个预测。由于我们处于 `eval` 模式，PyTorch 希望有一个 `list` 图像列表来进行预测。它也不再需要将 `labels` 对象作为第二个参数传递，这很好，因为如果我们已经知道所有对象的位置，我们就不会这样做：

```
with torch.no_grad(): 
    pred = model([x.to(device)])
```

## 检查结果

现在我们可以查看我们的结果。PyTorch 的实现返回一个包含三个项目的 `dict` 列表：`boxes` 用于预测项的位置，`labels` 用于预测每个项目的类别，`scores` 用于与每个预测相关的置信度。以下代码显示了此图像的 `pred` 内容：

```
print(pred)

[\{'boxes': tensor([[31.9313, 65.4917, 59.7824, 93.3052],
        [64.1321, 23.8941, 92.0808, 51.8841],
        [70.3358, 26.2407, 96.2834, 53.7900],
        [64.9917, 24.2980, 92.9516, 52.2016],
        [30.9127, 65.1308, 58.6978, 93.3224]], device='cuda:0'), 'labels':
tensor([8, 4, 1, 9, 5], device='cuda:0'), 'scores': tensor([0.9896, 0.9868, 
0.1201, 0.0699, 0.0555], device='cuda:0')\}]
```

每个字典都有一个 `boxes` 张量，其形状为 (*k*′,4)，用于 *k*′ 个预测。`labels` 张量具有 (*k*′) 形状，为每个对象预测了标签。最后，`scores` 张量也具有 (*k*′) 形状，并为每个返回的预测返回一个范围在 `[0,1]` 之间的分数。这些分数是 RPN 子网络的“对象”分数。

在这种情况下，模型对其找到了一个 `8` 和一个 `4`（≥`0.9` 分）的信心很高，但对其他类别的信心较低。通过将这些结果打印成图片，更容易理解这些结果，所以我们快速定义了一个函数来做这件事：

```
import matplotlib.patches as patches

def plotDetection(ax, abs_pos, label=None): 
    """ 
    ax: the matplotlib axis to add this plot to 
    abs_pos: the positions of the bounding box 
    label: the label of the prediction to add 
    """ 
    x1, y1, x2, y2 = abs_pos 
    rect = patches.Rectangle((x1,y1),x2-x1,y2-y1, 
    linewidth=1,edgecolor=’r’,facecolor=’none’)    ❶ 
    ax.add_patch(rect) if label is not None:       ❷ 
        plt.text(x1+0.5, y1, label, color=’black’, 
        ➥ bbox=dict(facecolor=’white’, edgecolor=’white’, pad=1.0))

    return

def showPreds(img, pred): 
    """ 
    img: the image to show the bounding box predictions for 
    pred: the Faster R-CNN predictions to show on top of the image 
    """ 
    fig,ax = plt.subplots(1) 
    ax.imshow(img.cpu().numpy()[0,:])              ❸ 
    boxes = pred[’boxes’].cpu()                    ❹ 
    labels = pred[’labels’].cpu() 
    scores = pred[’scores’].cpu()

    num_preds = labels.shape[0] 
    for i in range(num_preds):                     ❺
        plotDetection(ax, boxes[i].cpu().numpy(), 
        ➥ label=str(labels[i].item()))

    plt.show()
```

❶ 为边界框制作一个矩形。

❷ 如果提供了标签，则添加标签。

❸ 绘制图像。

❹ 获取预测。

❺ 对于每个预测，绘制其是否有足够高的分数。

拥有这段代码，我们可以在该图像上绘制 Faster R-CNN 的结果。我们可以清楚地看到，网络在 `4` 和 `8` 上做得很好，但完全错过了 `0`。我们还从网络中得到了一些虚假预测，识别 `4` 和 `8` 的子部分为其他数字。例如，看看 `4` 的右半部分。如果你没有注意到左半部分，你可能会说它是数字 `1`——或者你可能正在看整个图像，认为它是不完整的 `9`。`8` 也有类似的问题。如果你忽略左上角的环，它看起来像 `6`；如果你忽略 `8` 的右半部分，你可能会被原谅为称之为 `9`：

```
showPreds(x, pred[0])
```

![图片](img/CH08_UN08_Raff.png)

假重叠对象是目标检测器中常见的问题。有时这些重叠对象是同一对象的预测（例如，识别几个 `8`）或者像我们这里看到的误标记预测。

### 8.4.3 抑制重叠盒子

解决这个问题的简单有效的方法是*抑制*重叠的盒子。我们如何知道要抑制哪些盒子？我们想要确保我们选择的是*正确*的盒子来使用，但我们也不希望丢弃正确预测相邻对象的盒子。

一种称为*非极大值抑制*（NMS）的简单方法可以用来做这个。NMS 使用两个盒子之间的*交集与并集*（IoU）来确定它们是否重叠*太多*。IoU 是一个分数：`1` 表示盒子具有完全相同的位置，`0` 表示没有重叠。图 8.13 显示了它是如何计算的。

![图片](img/CH08_F13_Raff.png)

图 8.13 交集与并集分数是通过将两个盒子之间的重叠面积除以两个盒子并集的面积来计算的。

IoU 通过将两个盒子交集的大小除以两个盒子的大小来计算。这样，我们得到一个对两个盒子位置相似性的大小敏感度量。NMS 通过取每对具有大于某个指定阈值的 IoU 的盒子，并只保留 RPN 网络中分数*最高*的一个来工作。

让我们快速看看 NMS 方法在我们数据上的工作方式。我们可以从 PyTorch 导入它作为 `nms` 函数：

```
from torchvision.ops import nms
```

让我们通过从 `model` 返回的 `pred` 中打印出这些字段来提醒自己我们拥有的盒子和它们相关的分数：

```
print(pred[0][’boxes’])

tensor([[31.9313, 65.4917, 59.7824, 93.3052],
        [64.1321, 23.8941, 92.0808, 51.8841],
        [70.3358, 26.2407, 96.2834, 53.7900],
        [64.9917, 24.2980, 92.9516, 52.2016],
        [30.9127, 65.1308, 58.6978, 93.3224]], device='cuda:0')

print(pred[0][’scores’])

tensor([0.9896, 0.9868, 0.1201, 0.0699, 0.0555], device='cuda:0')
```

`nms` 函数愉快地接受盒子张量作为第一个参数，分数张量作为第二个参数。第三个和最后一个参数是调用两个盒子为不同对象的阈值。以下代码表示，如果两个盒子的 IoU 达到 `50%` 或更高，它们是同一项，我们应该保留分数最高的盒子：

```
print(nms(pred[0][’boxes’], pred[0][’scores’], 0.5))

tensor([0, 1], device='cuda:0')
```

我们使用 `50%` 的重叠阈值，`nms` 返回一个大小相等或更小的张量，告诉我们应该保留哪些索引。在这种情况下，它表示应该保留分数最高的盒子 `0` 和 `1`。

让我们修改我们的预测函数，使用 NMS 清理 Faster R-CNN 的输出。我们还添加了一个 `min_score` 标志，我们可以用它来抑制不太可能是有意义的预测：
