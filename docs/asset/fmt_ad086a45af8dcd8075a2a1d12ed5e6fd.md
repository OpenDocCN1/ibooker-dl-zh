
由于它生成的充满算法痕迹、鸟羽、狗脸和眼睛的奇异图片，`DeepDream`迅速成为互联网现象。这些痕迹是`DeepDream`的`ConvNet`在`ImageNet`上训练的结果，在`ImageNet`中，狗的品种和鸟的种类被大量过度代表。如果你尝试另一个在具有多数分布的其他对象的数据集上预训练的网络，例如汽车，你将在输出图像中看到汽车特征。

该项目最初是一个有趣的实验，旨在反向运行`CNN`并使用第 9.1 节中解释的相同的卷积滤波器可视化技术来可视化其激活图：反向运行卷积神经网络，对输入进行梯度上升，以最大化卷积神经网络上层特定滤波器的激活。`DeepDream`使用了这个相同的思想，但做了一些修改：

+ 输入图像——在滤波器可视化中，我们不使用输入图像。我们从一张空白图像（或稍微有点噪声的图像）开始，然后最大化卷积层的滤波器激活，以查看其特征。在`DeepDream`中，我们使用输入图像到网络，因为目标是将这些可视化的特征打印到图像上。

+ 最大化滤波器与层——在滤波器可视化中，正如其名所示，我们只最大化层内特定滤波器的激活。但在`DeepDream`中，我们的目标是最大化整个层的激活，以一次混合大量特征。

+ 八度音阶——在`DeepDream`中，输入图像以不同的尺度称为八度音阶进行处理，以提高可视化的特征质量。这个过程将在下一节中解释。

### 9.2.1 DeepDream 算法的工作原理

与过滤可视化技术类似，`DeepDream`使用在大数据集上预训练的网络。`Keras`库提供了许多可用的预训练卷积神经网络：`VGG16`、`VGG19`、`Inception`、`ResNet`等等。我们可以在`DeepDream`实现中使用这些网络中的任何一个；我们甚至可以在自己的数据集上训练一个自定义网络，并在`DeepDream`算法中使用它。直观地说，网络的选择以及它预训练的数据将影响我们的可视化，因为不同的卷积神经网络架构会导致不同的学习特征；当然，不同的训练数据集也会创建不同的特征。

`DeepDream`的创造者使用了`Inception`模型，因为他们发现实际上它产生的梦境看起来很美。所以在本章中，我们将使用`Inception v3`模型。我们鼓励你尝试不同的模型来观察差异。

`DeepDream`的整体思想是，我们将输入图像通过一个预训练的神经网络，如`Inception v3`模型。在某个层，我们计算梯度，它告诉我们如何改变输入图像以最大化该层的值。我们继续这样做，直到 10、20 或 40 次迭代，最终，输入图像中开始出现模式（图 9.11）。

![图片](img/9-11.png)

图 9.11 DeepDream 算法

这工作得很好，但是，如果预训练的网络是在相当小的图像尺寸上训练的，比如`ImageNet`，那么当我们的输入图像很大（比如 1000 × 1000）时，`DeepDream`算法会在图像中打印出很多小的模式，看起来很嘈杂而不是艺术。这是因为所有提取的特征都很小。为了解决这个问题，`DeepDream`算法以不同的尺度称为八度音阶处理输入图像。

八度只是一个表示区间的时髦词。想法是通过区间应用`DeepDream`算法于输入图像。我们首先将图像降级为几个不同的尺度。尺度数量是可配置的，你很快就会看到。对于每个区间，我们执行以下操作：

1. 注入细节：为了避免在每次连续放大后丢失大量图像细节，我们在每次放大过程之后将丢失的细节重新注入到图像中，以创建混合图像。

2. 应用`DeepDream`算法：将混合图像通过`DeepDream`算法。

3. 放大到下一个区间。

如图 9.12 所示，我们从一个大输入图像开始，然后将其降级两次，以在 3 次方频域中得到一个小图像。在应用`DeepDream`的第一个区间，我们不需要进行细节注入，因为输入图像是尚未放大过的源图像。我们将其通过`DeepDream`算法，然后放大输出。放大后，细节丢失，导致图像越来越模糊或像素化。这就是为什么在 2 次方频域中重新注入输入图像的细节，然后将混合图像通过`DeepDream`算法非常有价值。我们再次应用放大、细节注入和`DeepDream`的过程，以获得最终的结果图像。这个过程以递归方式运行，直到我们对输出艺术满意为止。

![](img/9-12.png)

图 9.12 DeepDream 过程：连续图像降级称为八度，细节重新注入，然后放大到下一个八度

我们将`DeepDream`参数设置为以下内容：

```
num_octave = 3           ❶

octave_scale = 1.4       ❷

iterations = 20          ❸
```

❶ 尺度数量

❷ 尺度之间的尺寸比。每个连续的尺度比前一个大 1.4 倍（即 40%更大）。

❸ 迭代次数

现在你已经了解了`DeepDream`算法的工作原理，让我们看看如何使用`Keras`实现`DeepDream`。

### 9.2.2 Keras 中的 DeepDream 实现

我们将要实现的`DeepDream`实现是基于 François Chollet 从官方`Keras`文档（[`keras.io/examples/generative/deep_dream/`](https://keras.io/examples/generative/deep_dream/））和他的书《Python 深度学习》中的代码。我们将在此代码适应`Jupyter Notebooks`后解释此代码：

```
import numpy as np
from keras.applications import inception_v3
from keras import backend as K
from keras.preprocessing.image import save_img

K.set_learning_phase(0)                                                     ❶

model = inception_v3.InceptionV3(weights='imagenet', include_top=False)     ❷
```

❶ 禁用所有训练操作，因为我们不会用该模型进行任何训练

❷ 下载预训练的`Inception v3`模型，不包括其顶部部分

现在，我们需要定义一个字典，指定用于生成梦境的层。为此，让我们打印出模型摘要以查看所有层并选择层名：

```
model.summary()
```

`Inception v3`非常深，摘要打印很长。为了简单起见，图 9.13 显示了网络的一些层。

你选择的精确层及其对最终损失的贡献对你在梦境图像中产生的视觉效果有重要影响，因此你希望这些参数易于配置。为了定义我们想要贡献于梦境创建的层，我们创建一个包含层名称及其相应权重的字典。层的权重越大，其对梦境的贡献级别越高：

```
layer_contributions = {
                        'mixed2': 0.4,
                        'mixed3': 2.,
                        'mixed4': 1.5,
                        'mixed5': 2.3,
                      }
```

![](img/9-13.png)

图 9.13 Inception v3 模型摘要的一部分

这些是我们尝试最大化激活的层名称。请注意，当你更改此字典中的层时，你将产生不同的梦境，因此我们鼓励你尝试不同的层及其相应的权重。对于这个项目，我们将通过向我们的字典添加四个层及其权重（`mixed2`、`mixed3`、`mixed4`和`mixed5`）来从一个相对任意的配置开始。作为一个指导，记得从本章前面的内容中，较低层可以用来生成边缘和几何图案，而高层则导致注入令人着迷的视觉图案，包括狗、猫和鸟的幻觉。

现在，让我们定义一个包含损失的张量：各层激活的 L2 范数的加权总和：

```
layer_dict = dict([(layer.name, layer) for layer in model.layers])      ❶

loss = K.variable(0.)                                                   ❷

for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output
    scaling = K.prod(K.cast(K.shape(activation), 'float32'))

    loss = loss + coeff * 
        K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling       ❸
```

❶ 将层名称映射到层实例的字典

❷ 通过添加层贡献到这个标量变量来定义损失

❸ 将层的特征 L2 范数添加到损失中。我们通过只涉及非边界像素来避免边界伪影。

接下来，我们计算损失，这是我们将在梯度上升过程中尝试最大化的量。在滤波器可视化中，我们希望最大化特定层中特定滤波器的值。在这里，我们将同时最大化多个层中所有滤波器的激活。具体来说，我们将最大化一组高级层激活的 L2 范数的加权总和：

```
dream = model.input                                           ❶

grads = K.gradients(loss, dream)[0]                           ❷

grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)                ❸

outputs = [loss, grads]                                       ❹
fetch_loss_and_grads = K.function([dream], outputs)           ❹

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values
def gradient_ascent(x, iterations, step, max_loss=None):      ❺
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x
```

❶ 包含生成图像的张量

❷ 计算梦境图像相对于损失的梯度

❸ 归一化梯度

❹ 设置一个`Keras`函数以检索给定输入图像的损失和梯度值

❺ 运行梯度上升过程多次迭代

现在我们已经准备好开发我们的`DeepDream`算法。过程如下：

1. 加载输入图像。

2. 定义从最小到最大的尺度数量。

3. 将输入图像调整到最小尺寸。

4. 对于每个尺度，从最小开始，应用以下：

    + 梯度上升函数

    + 上采样到下一个尺度

    + 在上采样过程中重新注入丢失的细节

    + 当我们回到原始大小时停止该过程。

首先，我们设置算法参数：

```
step = 0.01              ❶
num_octave = 3           ❷
octave_scale = 1.4       ❸
iterations = 20          ❹
max_loss = 10.
```

❶ 梯度上升步长

❷ 运行梯度上升的尺度数量

❸ 尺度之间的尺寸比

❹ 迭代次数

注意，调整这些超参数将允许你实现新的效果。

让我们定义我们想要用来创建梦境的输入图像。在这个例子中，我下载了一张旧金山金门大桥的图片（见图 9.14）；你也可以使用你自己的图片。图 9.15 显示了`DeepDream`的输出图像。

![图片](img/9-14.png)

图 9.14 示例输入图像

![图片](img/9-15.png)

图 9.15 DeepDream 输出

这里是`Keras`代码：

```
base_image_path = 'input.jpg'                      ❶
img = preprocess_image(base_image_path)
original_shape = img.shape[1:3]

successive_shapes = [original_shape]

for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)
successive_shapes = successive_shapes[::-1]
original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])
for shape in successive_shapes:
    print('Processing image shape', shape)

    img = resize_img(img, shape)
    img = gradient_ascent(img, iterations=iterations, step=step, 
                          max_loss=max_loss)
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img
    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)

    phil_img = deprocess_image(np.copy(img))
    save_img('deepdream_output/dream_at_scale_' + str(shape) + '.png', phil_img)

final_img = deprocess_image(np.copy(img))
save_img('final_dream.png', final_img)              ❷
```

❶ 定义输入图像的路径

❷ 将结果保存到磁盘

## 9.3 神经风格迁移

到目前为止，我们已经学习了如何在网络中可视化特定的过滤器。我们还学习了如何使用`DeepDream`算法操纵输入图像的特征来创建梦幻般的幻觉图像。在本节中，我们将探索一种新的艺术图像类型，即卷积神经网络可以通过神经风格迁移来创建的：将一种图像的风格转移到另一种图像的技术。

神经风格迁移算法的目标是将一张图像的风格（风格图像）应用到另一张图像的内容（内容图像）上。这里的风格指的是图像中的纹理、颜色和其他视觉模式。而内容则是图像的高级宏观结构。结果是包含内容图像的内容和风格图像的风格的综合图像。

例如，让我们看看图 9.16。内容图像中的对象（如海豚、鱼和植物）在综合图像中保持不变，但具有风格图像的特定纹理（蓝色和黄色的笔触）。

![图片](img/9-16.png)

图 9.16 神经风格迁移示例

神经风格迁移的想法是由 Leon A. Gatys 等人于 2015 年提出的。在此之前，与纹理生成紧密相关的风格迁移概念在图像处理社区中有着悠久的历史；但事实证明，基于深度学习的风格迁移实现提供了传统计算机视觉技术所无法比拟的结果，并引发了一场在创造性计算机视觉应用中的惊人复兴。

在创建艺术的不同神经网络技术（如`DeepDream`）中，风格迁移是我最喜欢的一种。`DeepDream`可以创建酷炫的幻觉图像，但有时可能会让人感到不安。此外，作为一个深度学习工程师，有意创造你心中所想的特定艺术品并不容易。另一方面，风格迁移可以使用艺术工程师将你从图像中想要的内容与你最喜欢的画作混合，创造出你想象中的东西。这是一个非常酷的技术，如果由艺术工程师使用，可以创造出与专业画家作品相媲美的美丽艺术。

实现风格迁移背后的主要思想与第二章中解释的所有深度学习算法的核心思想相同：我们首先定义一个损失函数来定义我们想要实现的目标，然后我们致力于优化这个函数。在风格迁移问题中，我们知道我们想要实现的目标是：在保留原始图像内容的同时，采用参考图像的风格。现在我们所需做的就是以数学形式定义内容和风格，然后定义一个合适的损失函数来最小化它。

定义损失函数的关键思想是要记住，我们想要保留一张图像的内容和另一张图像的风格：

+ 内容损失——计算内容图像和组合图像之间的损失。最小化这个损失意味着组合图像将包含更多来自原始图像的内容。

+ 风格损失——计算风格图像和组合图像之间的风格损失。最小化这个损失意味着组合图像将具有与风格图像相似的风格。

+ 噪声损失——这被称为总变差损失。它衡量了组合图像中的噪声。最小化这个损失会创建一个具有更高空间平滑性的图像。

这里是总损失的方程：
