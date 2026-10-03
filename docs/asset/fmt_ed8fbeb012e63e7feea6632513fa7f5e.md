
# 可视化中间激活

```py
images_per_row = 16
# Iterates over the activations (and the names of the corresponding
# layers)
for layer_name, layer_activation in zip(layer_names, activations):
    # The layer activation has shape (1, size, size, n_features).
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    # Prepares an empty grid for displaying all the channels in this
    # activation
    display_grid = np.zeros(
        ((size + 1) * n_cols - 1, images_per_row * (size + 1) - 1)
    )
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_index = col * images_per_row + row
            # This is a single channel (or feature).
            channel_image = layer_activation[0, :, :, channel_index].copy()
            # Normalizes channel values within the [0, 255] range.
            # All-zero channels are kept at zero.
            if channel_image.sum() != 0:
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype("uint8")
            # Places the channel matrix in the empty grid we prepared
            display_grid[
                col * (size + 1) : (col + 1) * size + col,
                row * (size + 1) : (row + 1) * size + row,
            ] = channel_image
    # Displays the grid for the layer
    scale = 1.0 / size
    plt.figure(
        figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0])
    )
    plt.title(layer_name)
    plt.grid(False)
    plt.axis("off")
    plt.imshow(display_grid, aspect="auto", cmap="viridis") 
```

代码清单 10.6：可视化每个中间激活中的每个通道

![](img/449cd612e2702fb4072369056fb379ca.png)

图 10.3：测试猫图片上每个层的每个通道的激活

这里有一些需要注意的事项：

+ 第一层充当各种边缘检测器的集合。在那个阶段，激活保留了初始图片中几乎所有信息。
+ 随着你向上移动，激活变得越来越抽象，视觉上可解释性越来越低。它们开始编码更高级的概念，如“猫耳”和“猫眼”。更高层的表示携带越来越少的关于图像视觉内容的信息，以及越来越多的与图像类别相关的信息。
+ 激活稀疏度随着层深度的增加而增加：在第一层，所有滤波器都由输入图像激活，但在后续层中，越来越多的滤波器变为空白。这意味着滤波器编码的图案在输入图像中没有找到。

我们刚刚观察到了深度神经网络学习到的表示的一个重要普遍特征：层中提取的特征随着层深度的增加而变得越来越抽象。更高层的激活携带越来越少关于特定输入的信息，而越来越多关于目标（在这种情况下，图像的类别：猫或狗）的信息。深度神经网络有效地充当一个*信息蒸馏管道*，原始数据进入（在这种情况下，RGB 图片），并被反复转换，以便过滤掉无关信息（例如，图像的具体视觉外观），并放大和细化有用信息（例如，图像的类别）。

这与人类和动物感知世界的方式类似：观察场景几秒钟后，人类可以记住其中包含哪些抽象对象（自行车、树），但无法记住这些对象的具体外观。事实上，如果你试图凭记忆画一辆通用的自行车，你很可能连大致的样子都画不出来，尽管你一生中见过成千上万辆自行车（例如，见图 10.4）。现在试试看：这种效果绝对是真实的。你的大脑已经学会了完全抽象其视觉输入——将其转化为高级视觉概念，同时过滤掉无关的视觉细节——这使得记住周围事物的外观变得极其困难。

![](img/8118eccf021ef647fb2a0c668c897a7c.png)

图 10.4：左：尝试凭记忆画一辆自行车。右：一个示意图自行车应该看起来像什么。

## 可视化 ConvNet 滤波器

检查 ConvNets 学习到的滤波器的一个简单方法是通过显示每个滤波器旨在响应的视觉模式。这可以通过在输入空间中进行*梯度上升*来实现，对 ConvNet 的输入图像的值应用*梯度下降*，以*最大化*特定滤波器的响应，从空白输入图像开始。结果输入图像将是选择滤波器响应最大的图像。

让我们用 Xception 模型的滤波器来尝试这个方法。过程很简单：我们将构建一个损失函数，该函数最大化给定卷积层中给定滤波器的值，然后我们将使用随机梯度下降来调整输入图像的值，以最大化这个激活值。这将是你的第二个低级梯度下降循环的例子（第一个在第二章中）。我们将为 TensorFlow、PyTorch 和 Jax 展示它。

首先，让我们实例化在 ImageNet 数据集上训练的 Xception 模型。我们可以再次使用 KerasHub 库，就像我们在第八章中所做的那样。

```py
import keras_hub

# Instantiates the feature extractor network from pretrained weights
model = keras_hub.models.Backbone.from_preset(
    "xception_41_imagenet",
)
# Loads the matching preprocessing to scale our input images
preprocessor = keras_hub.layers.ImageConverter.from_preset(
    "xception_41_imagenet",
    image_size=(180, 180),
) 
```

代码列表 10.7：实例化 Xception 卷积基

我们对模型的卷积层感兴趣——`Conv2D`和`SeparableConv2D`层。我们需要知道它们的名称，以便检索它们的输出。让我们按深度顺序打印它们的名称。

```py
for layer in model.layers:
    if isinstance(layer, (keras.layers.Conv2D, keras.layers.SeparableConv2D)):
        print(layer.name) 
```

列表 10.8：打印 Xception 中所有卷积层的名称

你会注意到这里的`SeparableConv2D`层都命名为类似`block6_sepconv1`、`block7_sepconv2`等——Xception 被结构化为块，每个块包含几个卷积层。

现在，让我们创建第二个模型，该模型返回特定层的输出——“特征提取器”模型。由于我们的模型是功能 API 模型，它是可检查的：你可以查询其某个层的`output`并在新模型中重用它。无需复制整个 Xception 代码。

```py
# You could replace this with the name of any layer in the Xception
# convolutional base.
layer_name = "block3_sepconv1"
# This is the layer object we're interested in.
layer = model.get_layer(name=layer_name)
# We use model.input and layer.output to create a model that, given an
# input image, returns the output of our target layer.
feature_extractor = keras.Model(inputs=model.input, outputs=layer.output) 
```

列表 10.9：返回特定输出的特征提取器模型

要使用此模型，我们只需在输入数据上调用它即可，但我们应该小心应用我们模型特有的图像预处理，以确保我们的图像被缩放到与 Xception 预训练数据相同的范围。

```py
activation = feature_extractor(preprocessor(img_tensor)) 
```

列表 10.10：使用特征提取器

让我们使用我们的特征提取器模型定义一个函数，该函数返回一个标量值，量化给定输入图像在层中激活给定滤波器的程度。这是我们将在梯度上升过程中最大化的损失函数：

```py
from keras import ops

# The loss function takes an image tensor and the index of the filter
# we consider (an integer).
def compute_loss(image, filter_index):
    activation = feature_extractor(image)
    # We avoid border artifacts by only involving nonborder pixels in
    # the loss: we discard the first 2 pixels along the sides of the
    # activation.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    # Returns the mean of the activation values for the filter
    return ops.mean(filter_activation) 
```

一个有助于梯度上升过程顺利进行的非明显技巧是将梯度张量通过除以其 L2 范数（张量中值的平方和的平方根）进行归一化。这确保了对输入图像所做的更新幅度始终在相同的范围内。

让我们设置梯度上升步骤函数。任何涉及梯度的操作都需要调用后端 API，例如 TensorFlow 中的`GradientTape`、PyTorch 中的`.backward()`和 JAX 中的`jax.grad()`。让我们排列出三个后端中每个后端的代码片段，从 TensorFlow 开始。

### TensorFlow 中的梯度上升

对于 TensorFlow，我们只需打开一个`GradientTape`作用域，在其中计算损失以检索所需的梯度。我们将使用`@tf.function`装饰器来加速计算：

```py
import tensorflow as tf

@tf.function
def gradient_ascent_step(image, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        # Explicitly watches the image tensor, since it isn't a
        # TensorFlow Variable (only Variables are automatically watched
        # in a gradient tape)
        tape.watch(image)
        # Computes the loss scalar, indicating how much the current
        # image activates the filter
        loss = compute_loss(image, filter_index)
    # Computes the gradients of the loss with respect to the image
    grads = tape.gradient(loss, image)
    # Applies the "gradient normalization trick"
    grads = ops.normalize(grads)
    # Moves the image a little bit in a direction that activates our
    # target filter more strongly
    image += learning_rate * grads
    # Returns the updated image, so we can run the step function in a
    # loop
    return image 
```

列表 10.11：通过随机梯度上升进行损失最大化：TensorFlow

### PyTorch 中的梯度上升

在 PyTorch 的情况下，我们使用`loss.backward()`和`image.grad`来获取相对于输入图像的损失梯度，如下所示。

```py
import torch

def gradient_ascent_step(image, filter_index, learning_rate):
    # Creates a copy of "image" that we can get gradients for.
    image = image.clone().detach().requires_grad_(True)
    loss = compute_loss(image, filter_index)
    loss.backward()
    grads = image.grad
    grads = ops.normalize(grads)
    image = image + learning_rate * grads
    return image 
```

列表 10.12：通过随机梯度上升进行损失最大化：PyTorch

由于图像张量在每个迭代中都会被重新创建，因此无需重置梯度。

### JAX 中的梯度上升

在 JAX 的情况下，我们使用`jax.grad()`来获取一个函数，该函数返回相对于输入图像的损失梯度。

```py
import jax

grad_fn = jax.grad(compute_loss)

@jax.jit
def gradient_ascent_step(image, filter_index, learning_rate):
    grads = grad_fn(image, filter_index)
    grads = ops.normalize(grads)
    image += learning_rate * grads
    return image 
```

列表 10.13：通过随机梯度上升进行损失最大化：JAX

### 过滤器可视化循环

现在您已经拥有了所有部件。让我们将它们组合成一个 Python 函数，该函数接受一个过滤器索引作为输入，并返回一个表示在目标层中最大化指定过滤器激活的模式张量。
