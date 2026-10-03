
# RoI 池化与 Fast R-CNN

## RoI 池化

让我们固定输入 RoI 特征图的大小，该特征图进入神经网络。我们的任务是减少可变大小的 RoI 特征图到固定大小。如果固定的特征图大小设置为 *H*，*W*，并且我们的 RoI 在卷积特征图中对应于 (*r*，*c*，*h*，*w*)，我们将 *h* 和 *w* 分别划分为大小为 *h*/*H* 和 *w*/*W* 的相等大小的块，并对这些块应用最大池化以获得 *H*，*W* 特征图。

回到我们的例子，让我们固定 *H* = *W* = 4。我们期望的固定特征图大小是 (4,4,512)。因此，当我们的 RoI 是 (0, 0, 256, 256) 时，我们的 RoI 特征嵌入是 (16, 16, 512)：*h* = *w* = 16。我们将 16 × 16 区域划分为四个 (16/4, 16/4) 区域，并对每个区域执行最大池化以获得固定大小的 (4,4,512) 特征。同样，当我们的 RoI 是 (0, 0, 128, 128) 时，*h* = *w* = 8。我们将 8 × 8 区域划分为四个 (8/4, 8/4) 区域，并对这些区域执行最大池化以获得固定大小的 (4,4,512) 特征。

精明的读者会注意到，我们已仔细选择 RoI，使其是 *H* 和 *W* 的倍数，从而 *h*/*H* 和 *w*/*W* 分别得到整数值。但在现实中，这种情况很少发生。*h*/*H* 和 *w*/*W* 往往是浮点数。在这种情况下我们该怎么办？答案是*量化*：也就是说，我们选择最接近 *h*/*H* 和 *w*/*W* 的整数（在原始实现中为 floor 操作）。`ROIAlign` 已经改进了这一点，它使用双线性插值而不是量化。我们在这里不深入探讨 `ROIAlign` 的细节。

实际上，如果我们有一个大的 RoI，我们将特征图划分为一定数量的较大区域并执行最大池化。当我们有一个小的 RoI 时，我们将特征图划分为一定数量的较小区域并执行最大池化。用于池化的区域大小可以改变，但输出大小保持固定。RoI 池化输出的维度不依赖于输入特征图的大小或 RoI 的大小：它仅由我们将 RoI 划分的部分数量决定—*H* 和 *W*（见图 11.18）。

![图片](img/CH11_F18_Chaudhury.png)

图 11.18 一个具有两个不同大小感兴趣区域的卷积特征图。RoI 池化通过最大池化在单个过程中从每个 RoI 中提取固定大小的输出特征图（在此图像中为 2 × 2）。这使得我们能够为每个 RoI 提取固定大小的代表性特征向量，然后将其输入到进一步的分类器和回归层中进行分类和定位。

因此，RoI 池化的目的是对非均匀大小的输入执行最大池化以获得固定大小的特征图。Fast R-CNN 和 Faster R-CNN 使用 7 × 7 作为固定特征图大小。

## Fast R-CNN 架构

给定卷积特征图和一组 RoI，我们已经看到 RoI 池化层如何从特征图中提取固定长度的特征向量。每个特征向量被输入到一系列 FC 层中，最终分支成两个兄弟输出层：一个分类器，它为 *K* 个对象类别加上一个“背景”类别产生 softmax 概率估计，另一个回归器为 *K* 个对象类别中的每一个产生四个实数值。

## 为 Fast R-CNN 生成 GT

对于每张图像，我们有一个由 RPN 生成的 RoI 列表和一个 GT 边界框列表。我们如何为每个 RoI 生成 GT 和回归目标？想法与我们的 RPN 相同：我们使用 IoU 分数。算法如下：

1. 计算所有 RoI 和 GT 框之间的 IoU。
2. 对于每个 RoI，确定具有最高 IoU 的 GT 边界框。
3. 如果最高的 IoU 大于一个阈值（0.5），则将相应的 GT 标签分配给 RoI。
4. 如果 IoU 在 [0.1, 0.5] 之间，则分配背景标签。使用 0.1 的下限确保某些与 GT 有较小交集的 RoI 被选为背景。这有助于选择背景的困难示例；这是一种困难负样本挖掘的形式。

注意：完整的 Fast R-CNN RoI 头部的功能代码，可通过 Jupyter Notebook 执行，可在 [`mng.bz/nY48`](http://mng.bz/nY48) 找到。

### 列出 11.23 PyTorch 代码用于 Fast R-CNN RoI 头部

```py
class Fast_RCNN_ROI_Head(nn.Module):
    def __init__(self, num_classes, H, W, subsample=16, embedding_size=512):

        super(Fast_RCNN_ROI_Head, self).__init__()

        self.num_classes = num_classes
        self.H = H
        self.W = W
        self.embedding_size = embedding_size
        self.subsample = 16

        self.roi_head_classifier = nn.Sequential(
            nn.Linear(H*W*embedding_size, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
        )
        self.cls = torch.nn.Linear(4096, num_classes+1) ①
        self.reg = torch.nn.Linear(4096, (num_classes+1)*4)

    def forward(self, x, rois):                         ②
        assert x.shape[0] == 1 # This code only supports batch size of 1
        roi_pooled_features = torchvision.ops.roi_pool(
            x, [rois], output_size=(self.H, self.W), spatial_scale=1/subsample)
        roi_pooled_features = roi_pooled_features.view(
            -1, self.H*self.W*self.embedding_size)
        fc_out = self.roi_head_classifier(roi_pooled_features)
        roi_cls_scores = self.cls(fc_out)
        roi_loc = self.reg(fc_out)

        return roi_cls_scores, roi_loc                  ③
```

① `num_classes + background`

② `x` : (1, c, h, w) 张量，表示卷积特征图。`rois`: (n, 4) 张量，表示 RoI 的边界框

③ `roi_cls_scores`: (n, num_classes+1) 张量，表示每个 RoI 的分类得分。`roi_loc`: (n, (num_classes + 1) * 4) 张量，表示每个 RoI 的回归得分

## 训练 Fast R-CNN

RPN 为每张图像生成大约 2,000 个 RoI。由于计算限制，我们无法使用所有 *N* 个 RoI。相反，我们从中采样一个子集。训练 minibatch 是分层采样的，首先采样 *K* 个图像，然后从每个图像中采样 *R*/*K* 个 RoI。在 FRCNN 中，*R* 设置为 128。对于这次讨论，我们假设 *K* = 1：也就是说，每个 minibatch 只有一个图像。那么，对于单个图像的 RoI，我们如何从中采样 128 个 RoI 呢？

一个简单的解决方案是随机采样 128 个 RoI。然而，这会遇到我们之前讨论过的相同数据不平衡问题：我们最终采样背景的频率远高于类别。为了解决这个问题，我们采用与之前类似的采样策略。特别是，对于单个图像，我们采样 128 个 RoI，使得背景与对象的比率为 0.75:0.25。如果少于 32 个 RoI 包含对象，我们将使用更多的背景 RoI 填充 minibatch。

## 分配目标到 RoI 框

正如 RPN 的情况一样，我们为包含对象的全部 RoI 生成回归目标，即 GT 框相对于感兴趣区域的偏移量。对于所有背景 RoI，回归目标不适用。

## Fast R-CNN 损失函数

我们已经定义了 Fast R-CNN 网络以及如何为其输出生成标签和回归目标。我们需要讨论使我们能够训练 Fast R-CNN 的损失函数。正如预期的那样，有两个损失项：

+ *分类损失* — 我们使用任何标准分类器中使用的标准交叉熵损失。
+ *回归损失* — 回归损失仅适用于对象 RoI：背景 RoI 不参与回归。这里我们使用与 RPN 中相同的平滑 L1 损失。

因此，单个 RoI 的整体损失可以定义为以下：

![](img/eq_11-06.png)

方程 11.6

其中 *p* 是 RoI 的预测标签，*u* 是 RoI 的真实标签，*t[u]* = (*t[x]*, *t[y]*, *t[w]*, *t[h]*) 是类别 *u* 的回归预测，*v* = (*v[x]*, *v[y]*, *v[w]*, *v[h]*) 是 RoI 的回归目标。因此，整体损失可以定义为

![](img/eq_11-07.png)

方程 11.7

其中 *p[i]* 是 RoI *i* 的预测概率，*p[i]*^* 是 RoI *i* 的真实标签；*t[i]* = (*t[x]*, *t[y]*, *t[w]*, *t[h]*) 是对应于类别 *p[i]*^* 的 RoI *i* 的回归预测，*t[i]*^* = (*t[x]*^*, *t[y]*^*, *t[w]*^*, *t[h]*^*) 是 RoI *i* 的回归目标，*N[pos]* 是对象 RoI（非背景 RoI）的数量。

注意：完整的 Fast R-CNN 损失函数的代码，可通过 Jupyter Notebook 执行，可以在 [`mng.bz/nY48`](http://mng.bz/nY48) 找到。

### 列表 11.24 PyTorch 代码用于 Fast R-CNN 损失函数

```py
def rcnn_loss(
    roi_cls_scores,                              ①
    roi_loc,                                     ②
    roi_labels,                                  ③
    rcnn_loc_targets,                            ④
    lambda_ = 1):

    classification_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.SmoothL1Loss(reduction="sum")

    cls_loss = classification_criterion(roi_cls_scores, roi_labels)

    pos_roi_idxes = torch.where(roi_labels>0)[0] ⑤
    pred_all_offsets = roi_loc[pos_roi_idxes]

    num_pos_rois = len(pos_roi_idxes)
    pred_all_offsets = pred_all_offsets.view(    ⑥
        num_pos_rois, -1, 4)

    pred_cls_offsets = pred_all_offsets[
        torch.arange(num_pos_rois) , roi_labels[pos_roi_idxes]]

    gt_offsets = rcnn_loc_targets[pos_roi_idxes]

    reg_loss = reg_criterion(pred_cls_offsets, gt_offsets) / num_pos_rois
    return {
        "rcnn_cls_loss": cls_loss,
        "rcnn_reg_loss": reg_loss,
        "rcnn_total_loss": cls_loss + lambda_* reg_loss

    }
```

① (128, num_classes) 张量：每个 RoI 的 RCNN 分类器得分

② (128, num_classes*4) 张量：每个类别、RoI 的 RCNN 回归器预测

③ (128,) 张量：每个 RoI 的真实类别

④ (128, 4) 张量：每个 RoI 的 RoI 回归器目标

⑤ 找到正 RoI

⑥ (n, num_classes*4) 到 n, num_classes, 4)

## Fast R-CNN 推理

我们已经讨论了如何训练 Fast R-CNN 模块。一旦模型训练完成，下一个问题是如何使用该模型生成输出类别和边界框。

Fast R-CNN 模型为每个 RoI 输出一个分类得分和回归偏移量。我们可以安全地忽略背景 RoI。对于其余的 RoI，选择概率最高的类别作为输出标签，并选择相应的偏移量。我们应用与 RPN 类似的后续处理步骤：

1. 我们使用 RoI 将偏移量转换回 (`xtl`, `ytl`, `xbr`, `ybr`) 格式。
2. 我们将输出边界框裁剪到图像边界内

我们面临一个与之前类似的问题：输出可能包含多个与同一对象对应的边界框。我们以相同的方式处理它，即使用 NMS。然而，有一个区别。在 RPN 的情况下，我们在所有由 RPN 预测的边界框上应用了全局 NMS。这里，NMS 仅应用于属于同一类的边界框。这是对所有类进行的，这在直观上应该是合理的：如果边界框代表不同的类，那么抑制高度重叠的边界框是没有意义的。

注意：完整的 Fast R-CNN 推理代码，可通过 Jupyter Notebook 执行，可在 [`mng.bz/nY48`](http://mng.bz/nY48) 找到。

### 列表 11.25 PyTorch 代码用于 Fast R-CNN 推理
