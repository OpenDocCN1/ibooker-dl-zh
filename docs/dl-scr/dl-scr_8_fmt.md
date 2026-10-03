
# 附录 A. 深入探讨

在这一部分，我们深入探讨了一些重要但不是必要理解的技术领域。

## 矩阵链式法则

首先解释一下为什么我们可以在第一章的链式法则表达式中用 *W*^T 替换 ∂ν/∂u(X)。

记住 *L* 实际上是：

```
σ(XW11) + σ(XW12) + σ(XW21) + σ(XW22) + σ(XW31) + σ(XW32)
```

这是一个简写，意味着：

```
σ(XW11) = σ(x11 × w11 + x12 × w21 + x13 × w31)
σ(XW12) = σ(x11 × w12 + x12 × w22 + x13 × w32)
```

等等。让我们仅关注其中一个表达式。如果我们对 σ(XW11) 求关于 X 的每个元素的偏导数（这最终是我们将要对所有六个组件的 L 做的事情）会是什么样子？

嗯，因为：

```
σ(XW11) = σ(x11 × w11 + x12 × w21 + x13 × w31)
```

不难看出，通过链式法则的简单应用，这个表达式对于 x1 的偏导数是：

```
∂σ/∂u(XW11) × w11
```

由于 XW[11] 表达式中 x[11] 乘以的唯一因素是 w[11]，因此对其他所有元素的偏导数为 0。

因此，计算 σ(XW[11]) 对于 X 的所有元素的偏导数给出了以下关于 ∂σ(XW11)/∂X 的整体表达式：

```
[ ∂σ/∂u(XW11) × w11   ∂σ/∂u(XW11) × w21   ∂σ/∂u(XW11) × w31 ]
[ 0                     0                     0                   ]
[ 0                     0                     0                   ]
```

同样，我们可以计算 XW[32] 的偏导数对于 X 的每个元素：

```
[ 0                     0                     0                   ]
[ 0                     0                     0                   ]
[ ∂σ/∂u(XW32) × w12   ∂σ/∂u(XW32) × w22   ∂σ/∂u(XW32) × w32 ]
```

现在我们有了所有组件，可以直接计算 ∂Λ/∂X(S)。我们只需计算与前述矩阵相同形式的六个矩阵并将结果相加。

请注意，数学再次变得混乱，尽管不是高级的。您可以跳过以下计算，直接转到结论，这实际上是一个简单的表达式。但是通过计算将使您更加欣赏结论是多么令人惊讶简单。生活还有什么不是用来欣赏的呢？

这里只有两个步骤。首先，我们将明确写出 ∂Λ/∂X(S) 是刚刚描述的六个矩阵的总和：



<math display="block"><mrow><mfrac><mrow><mi>∂</mi><mi>Λ</mi></mrow> <mrow><mi>∂</mi><mi>X</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>S</mi> <mo>)</mo></mrow> <mo>=</mo> <mfrac><mrow><mi>∂</mi><mi>σ</mi><mo>(</mo><mi>X</mi><msub><mi>W</mi> <mn>11</mn></msub> <mo>)</mo></mrow> <mrow><mi>∂</mi><mi>X</mi></mrow></mfrac> <mo>+</mo> <mfrac><mrow><mi>∂</mi><mi>σ</mi><mo>(</mo><mi>X</mi><msub><mi>W</mi> <mn>12</mn></msub> <mo>)</mo></mrow> <mrow><mi>∂</mi><mi>X</mi></mrow></mfrac> <mo>+</mo> <mfrac><mrow><mi>∂</mi><mi>σ</mi><mo>(</mo><mi>X</mi><msub><mi>W</mi> <mn>21</mn></msub> <mo>)</mo></mrow> <mrow><mi>∂</mi><mi>X</mi></mrow></mfrac> <mo>+</mo> <mfrac><mrow><mi>∂</mi><mi>σ</mi><mo>(</mo><mi>X</mi><msub><mi>W</mi> <mn>22</mn></msub> <mo>)</mo></mrow> <mrow><mi>∂</mi><mi>X</mi></mrow></mfrac> <mo>+</mo> <mfrac><mrow><mi>∂</mi><mi>σ</mi><mo>(</mo><mi>X</mi><msub><mi>W</mi> <mn>31</mn></msub> <mo>)</mo></mrow> <mrow><mi>∂</mi><mi>X</mi></mrow></mfrac> <mo>+</mo> <mfrac><mrow><mi>∂</mi><mi>σ</mi><mo>(</mo><mi>X</mi><msub><mi>W</mi> <mn>32</mn></msub> <mo>)</mo></mrow> <mrow><mi>∂</mi><mi>X</mi></mrow></mfrac> <mo>=</mo> <mfenced close="]" open="["><mtable><mtr><mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>11</mn></msub> <mo>)</mo></mrow> <mo>×</mo> <msub><mi>w</mi> <mn>11</mn></msub></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>11</mn></msub> <mo>)</mo></mrow> <mo>×</mo> <msub><mi>w</mi> <mn>21</mn></msub></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>11</mn></msub> <mo>)</mo></mrow> <mo>×</mo> <msub><mi>w</mi> <mn>31</mn></msub></mrow></mtd></mtr> <mtr><mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd></mtr></mtable></mfenced> <mo>+</mo> <mfenced close="]" open="["><mtable><mtr><mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>12</mn></msub> <mo>)</mo></mrow> <mo>×</mo> <msub><mi>w</mi> <mn>12</mn></msub></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>12</mn></msub> <mo>)</mo></mrow> <mo>×</mo> <msub><mi>w</mi> <mn>22</mn></msub></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>12</mn></msub> <mo>)</mo></mrow> <mo>×</mo> <msub><mi>w</mi> <mn>32</mn></msub></mrow></mtd></mtr> <mtr><mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd></mtr></mtable></mfenced> <mo>+</mo> <mfenced close="]" open="["><mtable><mtr><mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>21</mn></msub> <mo>)</mo></mrow> <mo>×</mo> <msub><mi>w</mi> <mn>11</mn></msub></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>21</mn></msub> <mo>)</mo></mrow> <mo>×</mo> <msub><mi>w</mi> <mn>21</mn></msub></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>21</mn></msub> <mo>)</mo></mrow> <mo>×</mo> <msub><mi>w</mi> <mn>31</mn></msub></mrow></mtd></mtr> <mtr><mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd></mtr></mtable></mfenced> <mo>+</mo> <mfenced close="]" open="["><mtable><mtr><mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>22</mn></msub> <mo>)</mo></mrow> <mo>×</mo> <msub><mi>w</mi> <mn>12</mn></msub></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>22</mn></msub> <mo>)</mo></mrow> <mo>×</mo> <msub><mi>w</mi> <mn>22</mn></msub></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>22</mn></msub> <mo>)</mo></mrow> <mo>×</mo> <msub><mi>w</mi> <mn>32</mn></msub></mrow></mtd></mtr> <mtr><mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd></mtr></mtable></mfenced> <mo>+</mo> <mfenced close="]" open="["><mtable><mtr><mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>31</mn></msub> <mo>)</mo></mrow> <mo>×</mo> <msub><mi>w</mi> <mn>11</mn></msub></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>31</mn></msub> <mo>)</mo></mrow> <mo>×</mo> <msub><mi>w</mi> <mn>21</mn></msub></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>31</mn></msub> <mo>)</mo></mrow> <mo>×</mo> <msub><mi>w</mi> <mn>31</mn></msub></mrow></mtd></mtr></mtable></mfenced> <mo>+</mo> <mfenced close="]" open="["><mtable><mtr><mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>32</mn></msub> <mo>)</mo></mrow> <mo>×</mo> <msub><mi>w</mi> <mn>12</mn></msub></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>32</mn></msub> <mo>)</mo></mrow> <mo>×</mo> <msub><mi>w</mi> <mn>22</mn></msub></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>32</mn></msub> <mo>)</mo></mrow> <mo>×</mo> <msub><mi>w</mi> <mn>32</mn></msub></mrow></mtd></mtr></mtable></mfenced></mrow></math>

现在让我们将这个总和合并成一个大矩阵。这个矩阵不会立即呈现出任何直观的形式，但实际上是计算前述总和的结果。



现在来了有趣的部分。回想一下：

```
W = [[w11, w12],
     [w21, w22],
     [w31, w32]]
```

嗯，`W` 隐藏在前述矩阵中——它只是被转置了。回想一下：

```
∂Λ/∂u(S) = [[∂σ/∂u(XW11), ∂σ/∂u(XW12)],
            [∂σ/∂u(XW21), ∂σ/∂u(XW22)],
            [∂σ/∂u(XW31), ∂σ/∂u(XW32)]]
```

结果表明，前述矩阵等同于：

```
∂Λ/∂X(S) = [[∂σ/∂u(XW11)×w11 + ∂σ/∂u(XW12)×w12, ∂σ/∂u(XW11)×w21 + ∂σ/∂u(XW12)×w22, ∂σ/∂u(XW11)×w31 + ∂σ/∂u(XW12)×w32],
            [∂σ/∂u(XW21)×w11 + ∂σ/∂u(XW22)×w12, ∂σ/∂u(XW21)×w21 + ∂σ/∂u(XW22)×w22, ∂σ/∂u(XW21)×w31 + ∂σ/∂u(XW22)×w32],
            [∂σ/∂u(XW31)×w11 + ∂σ/∂u(XW32)×w12, ∂σ/∂u(XW31)×w21 + ∂σ/∂u(XW32)×w22, ∂σ/∂u(XW31)×w31 + ∂σ/∂u(XW32)×w32]]
```



# 相对于权重矩阵的损失梯度

首先，我们推导损失函数 `Λ` 相对于输入 `X` 的梯度。假设 `S = σ(X × W)`，其中 `σ` 是逐元素激活函数，`W` 是权重矩阵，则：

```
∂Λ/∂X (X) = [∂σ/∂u (X·W₁₁), ∂σ/∂u (X·W₁₂);
             ∂σ/∂u (X·W₂₁), ∂σ/∂u (X·W₂₂);
             ∂σ/∂u (X·W₃₁), ∂σ/∂u (X·W₃₂)] × [w₁₁, w₂₁, w₃₁;
                                                w₁₂, w₂₂, w₃₂]
           = ∂Λ/∂u (S) × Wᵀ
```

此外，请记住，我们正在寻找填写以下方程中问号的内容：

```
∂Λ/∂X (X) = ∂Λ/∂u (S) × ?
```

嗯，事实证明那个东西就是 `W`。这个结果就像肉从骨头上掉下来一样。

还要注意，这与我们之前在一维中看到的结果相同；同样，这将成为一个解释为什么深度学习有效并且允许我们干净地实现它的结果。这是否意味着我们可以*实际*替换前面方程中的问号并说 `∂ν/∂X (X, W) = Wᵀ`？不，不完全是。但是如果我们将两个输入（`X` 和 `W`）相乘以得到结果 `N`，并将这些输入通过一些非线性函数 `σ` 以得到输出 `S`，那么我们*可以*说以下内容：

```
∂σ/∂X (X, W) = ∂σ/∂u (N) × Wᵀ
```

这个数学事实使我们能够使用矩阵乘法的符号高效计算和表达梯度更新。此外，我们可以类似地推理出以下内容：

```
∂σ/∂W (X, W) = Xᵀ × ∂σ/∂u (N)
```

# 相对于偏差项的损失梯度

接下来，我们将详细介绍在完全连接的神经网络中计算损失相对于偏差项的导数时，为什么要沿着 `axis=0` 求和。

在神经网络中添加偏差项发生在以下情境中：我们有一个由 `n` 行（批量大小）乘以 `f` 列（特征数量）的矩阵表示的数据批次，并且我们向每个 `f` 特征中添加一个单个数字。例如，在第二章中的神经网络示例中，我们有 13 个特征，偏差项 `B` 有 13 个数字；第一个数字将添加到 `M1 = np.dot(X, weights["W1"])` 的第一列中的每一行，第二个数字将添加到第二列中的每一行，依此类推。在网络的后续部分，`B2` 将包含一个数字，它将简单地添加到 `M2` 的单列中的每一行。因此，由于相同的数字将被添加到矩阵的每一行，所以在反向传播时，我们需要沿着表示每个偏差元素添加到的行的维度对梯度求和。这就是为什么我们沿着 `axis=0` 对 `dLdB1` 和 `dLdB2` 的表达式求和；例如，`dLdB1 = (dLdN1 × dN1dB1).sum(axis=0)`。图 A-1 提供了所有这些的视觉解释，并附有一些评论。

![神经网络图](img/dlfs_aa01.png)

###### 图 A-1。总结了为什么计算完全连接层的输出相对于偏差的导数涉及沿着轴=0 求和

# 通过矩阵乘法进行卷积

最后，我们将展示如何通过批量矩阵乘法来表达批量、多通道卷积操作，以便在 NumPy 中高效实现它。

要理解卷积是如何工作的，请考虑在完全连接神经网络的前向传播中发生的情况：

+   我们收到一个大小为 `[batch_size, in_features]` 的输入。
+   我们将其乘以一个大小为 `[in_features, out_features]` 的参数。
+   我们得到一个大小为 `[batch_size, out_features]` 的结果输出。

在卷积层中，相比之下：

+   我们收到一个大小为 `[batch_size, in_channels, img_height, img_width]` 的输入。
+   我们将其与一个大小为 `[in_channels, out_channels, param_height, param_width]` 的参数进行卷积。
+   我们得到一个大小为 `[batch_size, in_channels, img_height, img_width]` 的结果输出。

使卷积操作看起来更像常规前馈操作的关键是*首先从输入图像的每个通道中提取 `img_height × img_width` “图像补丁”*。一旦提取了这些补丁，输入就可以被重新整形，以便卷积操作可以通过 NumPy 的 `np.matmul` 函数表达为批量矩阵乘法。首先：

```py
def _get_image_patches(imgs_batch: ndarray,
                       fil_size: int):
    '''
    imgs_batch: [batch_size, channels, img_width, img_height]
    fil_size: int
    '''
    # pad the images
    imgs_batch_pad = np.stack([_pad_2d_channel(obs, fil_size // 2)
                              for obs in imgs_batch])
    patches = []
    img_height = imgs_batch_pad.shape[2]

    # For each location in the image...
    for h in range(img_height-fil_size+1):
        for w in range(img_height-fil_size+1):

            # ...get an image patch of size [fil_size, fil_size]
            patch = imgs_batch_pad[:, :, h:h+fil_size, w:w+fil_size]
            patches.append(patch)

    # Stack, getting an output of size
    # [img_height * img_width, batch_size, n_channels, fil_size, fil_size]
    return np.stack(patches)
```

然后我们可以按照以下方式计算卷积操作的输出：

1.  获取大小为 `[batch_size, in_channels, img_height x img_width, filter_size, filter_size]` 的图像块。
2.  将其重塑为 `[batch_size, img_height × img_width, in_channels × filter_size × filter_size]`。
3.  将参数重塑为 `[in_channels × filter_size × filter_size, out_channels]`。
4.  进行批量矩阵乘法后，结果将是 `[batch_size, img_height × img_width, out_channels]`。
5.  将其重塑为 `[batch_size, out_channels, img_height, img_width]`。

```py
def _output_matmul(input_: ndarray,
                   param: ndarray) -> ndarray:
    '''
    conv_in: [batch_size, in_channels, img_width, img_height]
    param: [in_channels, out_channels, fil_width, fil_height]
    '''

    param_size = param.shape[2]
    batch_size = input_.shape[0]
    img_height = input_.shape[2]
    patch_size = param.shape[0] * param.shape[2] * param.shape[3]

    patches = _get_image_patches(input_, param_size)

    patches_reshaped = (
      patches
      .transpose(1, 0, 2, 3, 4)
      .reshape(batch_size, img_height * img_height, -1)
      )

    param_reshaped = param.transpose(0, 2, 3, 1).reshape(patch_size, -1)

    output = np.matmul(patches_reshaped, param_reshaped)

    output_reshaped = (
      output
      .reshape(batch_size, img_height, img_height, -1)
      .transpose(0, 3, 1, 2)
    )

    return output_reshaped
```

这就是前向传播！对于反向传播，我们需要计算参数梯度和输入梯度。同样，我们可以借鉴全连接神经网络中的做法。首先是参数梯度，全连接神经网络的梯度是：

```py
np.matmul(self.inputs.transpose(1, 0), output_grad)
```

这应该激励我们如何通过卷积操作实现反向传播：这里，输入形状是 `[batch_size, in_channels, img_height, img_width]`，接收到的输出*梯度*将是 `[batch_size, out_channels, img_height, img_width]`。考虑到参数的形状是 `[in_channels, out_channels, param_height, param_width]`，我们可以通过以下步骤实现这种转换：

1.  首先，我们需要从输入图像中提取图像块，得到与上次相同的输出，形状为 `[batch_size, in_channels, img_height x img_width, filter_size, filter_size]`。
2.  然后，借鉴全连接情况下的乘法，将其重塑为形状为 `[in_channels × param_height × param_width, batch_size × img_height × img_width]`。
3.  然后，将原始形状为 `[batch_size, out_channels, img_height, img_width]` 的输出重塑为形状为 `[batch_size × img_height × img_width, out_channels]`。
4.  将它们相乘，得到形状为 `[in_channels × param_height × param_width, out_channels]` 的输出。
5.  将其重塑为最终的参数梯度，形状为 `[in_channels, out_channels, param_height, param_width]`。

这个过程的实现如下：



以下代码计算参数梯度：

```py
def _param_grad_matmul(input_: ndarray,
                       param: ndarray,
                       output_grad: ndarray):
    '''
 input_: [batch_size, in_channels, img_width, img_height]
 param: [in_channels, out_channels, fil_width, fil_height]
 output_grad: [batch_size, out_channels, img_width, img_height]
 '''

    param_size = param.shape[2]
    batch_size = input_.shape[0]
    img_size = input_.shape[2] ** 2
    in_channels = input_.shape[1]
    out_channels = output_grad.shape[1]
    patch_size = param.shape[0] * param.shape[2] * param.shape[3]

    patches = _get_image_patches(input_, param_sizes)

    patches_reshaped = (
        patches
        .reshape(batch_size * img_size, -1)
        )

    output_grad_reshaped = (
        output_grad
        .transpose(0, 2, 3, 1)
        .reshape(batch_size * img_size, -1)
    )

    param_reshaped = param.transpose(0, 2, 3, 1).reshape(patch_size, -1)

    param_grad = np.matmul(patches_reshaped.transpose(1, 0),
                           output_grad_reshaped)

    param_grad_reshaped = (
        param_grad
        .reshape(in_channels, param_size, param_size, out_channels)
        .transpose(0, 3, 1, 2)
    )

    return param_grad_reshaped
```

此外，我们遵循一个非常类似的步骤来获取输入梯度，受到全连接层中操作的启发，即：

```py
np.matmul(output_grad, self.param.transpose(1, 0))
```

以下代码计算输入梯度：

```py
def _input_grad_matmul(input_: ndarray,
                       param: ndarray,
                       output_grad: ndarray):

    param_size = param.shape[2]
    batch_size = input_.shape[0]
    img_height = input_.shape[2]
    in_channels = input_.shape[1]

    output_grad_patches = _get_image_patches(output_grad, param_size)

    output_grad_patches_reshaped = (
        output_grad_patches
        .transpose(1, 0, 2, 3, 4)
        .reshape(batch_size * img_height * img_height, -1)
    )

    param_reshaped = (
        param
        .reshape(in_channels, -1)
    )

    input_grad = np.matmul(output_grad_patches_reshaped,
                           param_reshaped.transpose(1, 0))

    input_grad_reshaped = (
        input_grad
        .reshape(batch_size, img_height, img_height, 3)
        .transpose(0, 3, 1, 2)
    )

    return input_grad_reshaped
```

这三个函数构成了`Conv2DOperation`的核心，具体是它的`_output`、`_param_grad`和`_input_grad`方法，你可以在书的 GitHub 仓库中的[`lincoln`库](https://oreil.ly/2KPdFay)中看到。
