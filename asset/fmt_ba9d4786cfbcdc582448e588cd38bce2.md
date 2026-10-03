
```py
def gram_matrix(x):
    if K.image_data_format() != 'channels_first':
        x = K.permute_dimensions(x, (2, 0, 1))
    features = K.batch_flatten(x)
    return K.dot(features, K.transpose(features))

def style_loss(layer_1, layer_2):
    gr1 = gram_matrix(layer_1)
    gr2 = gram_matrix(layer_1)
    return K.sum(K.square(gr1 - gr2))
```

与以前一样，我们希望一个预训练网络来完成工作。我们将在两个图像上使用它，一个是我们生成的图像，另一个是我们想要从中捕捉风格的图像——在这种情况下是克劳德·莫奈 1912 年的《睡莲》。因此，我们将创建一个包含两者的输入张量，并加载一个没有最终层的网络，该网络以此张量作为输入。我们将使用`VGG16`，因为它很简单，但任何预训练网络都可以：

```py
style_image = K.variable(preprocess_image(style_image_path,
                                          target_size=(1024, 768)))
result_image = K.placeholder(style_image.shape)
input_tensor = K.concatenate([result_image,
                              style_image], axis=0)

model = vgg16.VGG16(input_tensor=input_tensor,
                    weights='imagenet', include_top=False)
```

一旦我们加载了模型，我们就可以定义我们的损失变量。我们将遍历模型的所有层，并对其中名称中包含`_conv`的层（卷积层）收集`style_image`和`result_image`之间的`style_loss`：

```py
loss = K.variable(0.)
for layer in model.layers:
    if '_conv' in layer.name:
        output = layer.output
        loss += style_loss(output[0, :, :, :], output[1, :, :, :])
```

现在我们有了一个损失，我们可以开始优化。我们将使用`scipy`的`fmin_l_bfgs_b`优化器。该方法需要一个梯度和一个损失值来完成其工作。我们可以通过一次调用获得它们，因此我们需要缓存这些值。我们使用一个方便的辅助类`Evaluator`来做到这一点，它接受一个损失和一个图像：

```py
class Evaluator(object):
    def __init__(self, loss_total, result_image):
        grads = K.gradients(loss_total, result_image)
        outputs = [loss_total] + grads
        self.iterate = K.function([result_image], outputs)
        self.shape = result_image.shape

        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        outs = self.iterate([x.reshape(self.shape)])
        self.loss_value = outs[0]
        self.grad_values = outs[-1].flatten().astype('float64')
        return self.loss_value

    def grads(self, x):
        return np.copy(self.grad_values)
```

现在我们可以通过重复调用来优化图像：

```py
image, min_val, _ = fmin_l_bfgs_b(evaluator.loss, image.flatten(),
                                  fprime=evaluator.grads, maxfun=20)
```

经过大约 50 步后，生成的图像开始看起来相当合理。

## 讨论

在这个教程中，我们看到了格拉姆矩阵有效地捕捉了图像的风格。天真地，我们可能会认为匹配图像风格的最佳方法是直接匹配所有层的激活。但这种方法太过直接了。

也许不明显格拉姆矩阵方法会更好。其背后的直觉是，通过将给定层的每个激活与其他每个激活相乘，我们捕捉了神经元之间的相关性。这些相关性编码了风格，因为它是激活分布的度量，而不是绝对激活。

考虑到这一点，我们可以尝试一些事情。一个要考虑的事情是零值。将一个向量与其自身转置的点积将在任一乘数为零时产生零。这使得无法发现与零的相关性。由于零值经常出现，这是不太理想的。一个简单的修复方法是在进行点操作之前向特征添加一个增量。值为`-1`效果很好：

```py
return K.dot(features - 1, K.transpose(features - 1))
```

我们还可以尝试向表达式添加一个常数因子。这可以使结果更加平滑或夸张。同样，`-1`效果很好。

最后要考虑的是我们正在获取所有激活的格拉姆矩阵。这可能看起来有点奇怪——难道我们不应该只为每个像素的通道执行此操作吗？实际上正在发生的是，我们为每个像素的通道计算格拉姆矩阵，然后查看它们在整个图像上的相关性。这允许一种快捷方式：我们可以计算平均通道并将其用作格拉姆矩阵。这给我们一个捕捉平均风格的图像，因此更加规则。它也运行得更快：

```py
def gram_matrix_mean(x):
    x = K.mean(x, axis=1)
    x = K.mean(x, axis=1)
    features = K.batch_flatten(x)
    return K.dot(features - 1,
                 K.transpose(features - 1)) / x.shape[0].value
```

在这个教程中添加的总变差损失告诉网络保持相邻像素之间的差异。如果没有这个，结果将更加像素化和跳跃。在某种程度上，这种方法与我们用来控制网络层的权重或输出的正则化非常相似。总体效果类似于在输出像素上应用轻微模糊滤镜。

# 12.5 改进损失函数以增加图像的连贯性

## 问题

如何使从捕捉到的风格生成的图像不那么像素化？

## 解决方案

添加一个损失组件来控制图像的局部连贯性。

前一个配方中的图像看起来已经相当合理。然而，如果我们仔细看，似乎有些像素化。我们可以通过添加一个损失函数来引导算法，确保图像在局部上是连贯的。我们将每个像素与其左侧和下方的邻居进行比较。通过试图最小化这种差异，我们引入了一种对图像进行模糊处理的方法：

```py
def total_variation_loss(x, exp=1.25):
    _, d1, d2, d3 = x.shape
    a = K.square(x[:, :d1 - 1, :d2 - 1, :] - x[:, 1:, :d2 - 1, :])
    b = K.square(x[:, :d1 - 1, :d2 - 1, :] - x[:, :d1 - 1, 1:, :])
    return K.sum(K.pow(a + b, exp))
```

1.25 指数确定了我们惩罚异常值的程度。将这个添加到我们的损失中得到：

```py
loss_variation = total_variation_loss(result_image, h, w) / 5000
loss_with_variation = loss_variation + loss_style
evaluator_with_variation = Evaluator(loss_with_variation, result_image)
```

如果我们运行这个评估器 100 步，我们会得到一个非常令人信服的图片：

![没有图像的神经风格](img/dlcb_12in06.png)

## 讨论

在这个配方中，我们向我们的损失函数添加了最终组件，使图片在全局上看起来像内容图像。实际上，我们在这里所做的是优化生成的图像，使得上层的激活对应于内容图像，而下层的激活对应于风格图像。由于下层对应于风格，上层对应于内容，我们可以通过这种方式实现风格转移。

结果可能非常引人注目，以至于新手可能认为计算机现在可以创作艺术。但仍然需要调整，因为有些风格比其他风格更狂野，我们将在下一个配方中看到。

# 12.6 将风格转移到不同的图像

## 问题

如何将从一幅图像捕捉到的风格应用到另一幅图像上？

## 解决方案

使用一个损失函数，平衡一幅图像的内容和另一幅图像的风格。

在现有图像上运行前一个配方中的代码会很容易，而不是在噪声图像上运行，但结果并不那么好。起初似乎是将风格应用到现有图像上，但随着每一步，原始图像都会逐渐消失一点。如果我们继续应用算法，最终结果将基本相同，不管起始图像如何。

我们可以通过向我们的损失函数添加第三个组件来修复这个问题，考虑生成的图像与我们参考图像之间的差异：

```py
def content_loss(base, combination):
    return K.sum(K.square(combination - base))
```

现在我们需要将参考图像添加到我们的输入张量中：

```py
w, h = load_img(base_image_path).size
base_image = K.variable(preprocess_image(base_image_path))
style_image = K.variable(preprocess_image(style2_image_path, target_size=(h, w)))
combination_image = K.placeholder(style_image.shape)
input_tensor = K.concatenate([base_image,
                              style_image,
                              combination_image], axis=0)
```

我们像以前一样加载网络，并在网络的最后一层定义我们的内容损失。最后一层包含了网络所看到的最佳近似，所以这确实是我们想要保持不变的：

```py
loss_content = content_loss(feature_outputs[-1][0, :, :, :],
                            feature_outputs[-1][2, :, :, :])
```

我们将稍微改变风格损失，考虑网络中的层的位置。我们希望较低的层承载更多的权重，因为较低的层捕捉更多的纹理/风格，而较高的层更多地涉及图像的内容。这使得算法更容易平衡图像的内容（使用最后一层）和风格（主要使用较低的层）：

```py
loss_style = K.variable(0.)
for idx, layer_features in enumerate(feature_outputs):
    loss_style += style_loss(layer_features[1, :, :, :],
                             layer_features[2, :, :, :]) * (0.5 ** idx)
```

最后，我们平衡这三个组件：

```py
loss_content /= 40
loss_variation /= 10000
loss_total = loss_content + loss_variation + loss_style
```

在阿姆斯特丹的 Oude Kerk（古教堂）的图片上运行这个算法，以梵高的《星夜》作为风格输入，我们得到：

![梵高的 Oude Kerk](img/dlcb_12in07.png)

# 12.7 风格插值

## 问题

您已经捕捉到两幅图像的风格，并希望将这两者之间的风格应用到另一幅图像中。如何混合它们？

## 解决方案

使用一个损失函数，带有额外的浮点数，指示应用每种风格的百分比。

我们可以轻松地扩展我们的输入张量，以接受两种风格图像，比如夏季和冬季各一种。在我们像以前一样加载模型之后，我们现在可以为每种风格创建一个损失：

```py
loss_style_summer = K.variable(0.)
loss_style_winter = K.variable(0.)
for idx, layer_features in enumerate(feature_outputs):
    loss_style_summer += style_loss(layer_features[1, :, :, :],
                                    layer_features[-1, :, :, :]) * (0.5 ** idx)
    loss_style_winter += style_loss(layer_features[2, :, :, :],
                                    layer_features[-1, :, :, :]) * (0.5 ** idx)
```

然后我们引入一个占位符`summerness`，我们可以输入以获得所需的`summerness`损失：

```py
summerness = K.placeholder()
loss_total = (loss_content + loss_variation +
              loss_style_summer * summerness +
              loss_style_winter * (1 - summerness))
```

我们的`Evaluator`类没有一种传递`summerness`的方法。我们可以创建一个新类或者子类现有类，但在这种情况下，我们可以通过“猴子补丁”来解决：

```py
combined_evaluator = Evaluator(loss_total, combination_image,
                               loss_content=loss_content,
                               loss_variation=loss_variation,
                               loss_style=loss_style)
iterate = K.function([combination_image, summerness],
                     combined_evaluator.iterate.outputs)
combined_evaluator.iterate = lambda inputs: iterate(inputs + [0.5])
```

这将创建一个图像，其中有 50%的夏季风格，但我们可以指定任何值。

## 讨论

在损失变量中添加另一个组件允许我们指定两种不同风格之间的权重。当然，没有任何阻止我们添加更多风格图像并改变它们的权重。值得探索的是改变风格图像的相对权重；梵高的*星空*图像非常鲜明，其风格很容易压倒更加微妙的绘画风格。
