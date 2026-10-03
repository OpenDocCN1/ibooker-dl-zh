
在这里，`|x1 - x2|`是两个图像之间的平均像素绝对差异，`|D(x1) - D(x2)|`是评论家预测之间的绝对差异。基本上，我们要求评论家的预测在两个图像之间变化的速率有限（即，梯度的绝对值必须在任何地方最多为 1）。我们可以看到这应用于利普希茨连续的一维函数中，如图 4-11 所示——在任何位置放置锥体时，线都不会进入锥体。换句话说，线在任何点上升或下降的速率都有限制。

![](img/gdl2_0411.png)

###### 图 4-11。利普希茨连续函数（来源：[维基百科](https://oreil.ly/Ki7ds)）

###### 提示

对于那些想深入了解为什么只有在强制执行这个约束时，Wasserstein 损失才有效的数学原理的人，Jonathan Hui 提供了[一个优秀的解释](https://oreil.ly/devy5)。

## 强制利普希茨约束

在原始的 WGAN 论文中，作者展示了通过在每个训练批次后将评论家的权重剪辑到一个小范围`[-0.01, 0.01]`内来强制执行利普希茨约束的可能性。

这种方法的批评之一是，评论家学习的能力大大降低，因为我们正在剪辑它的权重。事实上，即使在原始的 WGAN 论文中，作者们也写道，“权重剪辑显然是一种强制利普希茨约束的可怕方式。”强大的评论家对于 WGAN 的成功至关重要，因为没有准确的梯度，生成器无法学习如何调整其权重以生成更好的样本。

因此，其他研究人员寻找了其他方法来强制执行利普希茨约束，并提高 WGAN 学习复杂特征的能力。其中一种方法是带有梯度惩罚的 Wasserstein GAN。

在引入这种变体的论文中，作者展示了如何通过在评论家的损失函数中直接包含*梯度惩罚*项来强制执行利普希茨约束，如果梯度范数偏离 1，模型将受到惩罚。这导致了一个更加稳定的训练过程。

在下一节中，我们将看到如何将这个额外项构建到我们评论家的损失函数中。

## 梯度惩罚损失

图 4-12 是 WGAN-GP 评论家的训练过程的图表。如果我们将其与图 4-5 中原始鉴别器训练过程进行比较，我们可以看到的关键添加是梯度惩罚损失作为整体损失函数的一部分，与真实和虚假图像的 Wasserstein 损失一起。

![](img/gdl2_0412.png)

###### 图 4-12。WGAN-GP 评论家训练过程

梯度惩罚损失衡量了预测梯度的范数与输入图像之间的平方差异和 1 之间的差异。模型自然倾向于找到确保梯度惩罚项最小化的权重，从而鼓励模型符合利普希茨约束。

在训练过程中无法计算每个地方的梯度，因此 WGAN-GP 只在少数几个点处评估梯度。为了确保平衡混合，我们使用一组插值图像，这些图像位于连接真实图像批次和虚假图像批次的线上随机选择的点上，如图 4-13 所示。

![](img/gdl2_0413.png)

###### 图 4-13。图像之间的插值

在示例 4-8 中，我们展示了如何在代码中计算梯度惩罚。

##### 示例 4-8。梯度惩罚损失函数

```py
def gradient_penalty(self, batch_size, real_images, fake_images):
    alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0) # ①
    diff = fake_images - real_images
    interpolated = real_images + alpha * diff # ②

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = self.critic(interpolated, training=True) # ③

    grads = gp_tape.gradient(pred, [interpolated])[0] # ④
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3])) # ⑤
    gp = tf.reduce_mean((norm - 1.0) ** 2) # ⑥
    return gp
```

①

批次中的每个图像都会得到一个介于 0 和 1 之间的随机数，存储为向量`alpha`。

②

计算一组插值图像。

③

批评家被要求对这些插值图像进行评分。

④

根据输入图像计算预测的梯度。

⑤

计算这个向量的 L2 范数。

⑥

该函数返回 L2 范数与 1 之间的平均平方距离。

## 训练 WGAN-GP

使用 Wasserstein 损失函数的一个关键优势是我们不再需要担心平衡批评家和生成器的训练——事实上，当使用 Wasserstein 损失时，必须在更新生成器之前将批评家训练到收敛，以确保生成器更新的梯度准确。这与标准 GAN 相反，标准 GAN 中重要的是不要让鉴别器变得太强。

因此，使用 Wasserstein GAN，我们可以简单地在生成器更新之间多次训练批评家，以确保它接近收敛。通常使用的比例是每次生成器更新三到五次批评家更新。

我们现在介绍了 WGAN-GP 背后的两个关键概念——Wasserstein 损失和包含在批评家损失函数中的梯度惩罚项。包含所有这些想法的 WGAN 模型的训练步骤显示在示例 4-9 中。

##### 示例 4-9。训练 WGAN-GP

```py
def train_step(self, real_images):
    batch_size = tf.shape(real_images)[0]

    for i in range(3): # ①
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)
        )

        with tf.GradientTape() as tape:
            fake_images = self.generator(
                random_latent_vectors, training = True
            )
            fake_predictions = self.critic(fake_images, training = True)
            real_predictions = self.critic(real_images, training = True)

            c_wass_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(
                real_predictions
            ) # ②
            c_gp = self.gradient_penalty(
                batch_size, real_images, fake_images
            ) # ③
            c_loss = c_wass_loss + c_gp * self.gp_weight # ④

        c_gradient = tape.gradient(c_loss, self.critic.trainable_variables)
        self.c_optimizer.apply_gradients(
            zip(c_gradient, self.critic.trainable_variables)
        ) # ⑤

    random_latent_vectors = tf.random.normal(
        shape=(batch_size, self.latent_dim)
    )
    with tf.GradientTape() as tape:
        fake_images = self.generator(random_latent_vectors, training=True)
        fake_predictions = self.critic(fake_images, training=True)
        g_loss = -tf.reduce_mean(fake_predictions) # ⑥

    gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
    self.g_optimizer.apply_gradients(
        zip(gen_gradient, self.generator.trainable_variables)
    ) # ⑦

    self.c_loss_metric.update_state(c_loss)
    self.c_wass_loss_metric.update_state(c_wass_loss)
    self.c_gp_metric.update_state(c_gp)
    self.g_loss_metric.update_state(g_loss)

    return {m.name: m.result() for m in self.metrics}
```

①

执行三次批评家更新。

②

为批评家计算 Wasserstein 损失——虚假图像和真实图像的平均预测之间的差异。

③

计算梯度惩罚项（参见示例 4-8）。

④

批评家损失函数是 Wasserstein 损失和梯度惩罚的加权和。

⑤

更新批评家的权重。

⑥

为生成器计算 Wasserstein 损失。

⑦

更新生成器的权重。

# WGAN-GP 中的批归一化

在训练 WGAN-GP 之前我们应该注意的最后一个考虑是批归一化不应该在批评家中使用。这是因为批归一化会在同一批次中的图像之间创建相关性，使得梯度惩罚损失效果不佳。实验证明，即使在批评家中没有批归一化，WGAN-GP 仍然可以产生出色的结果。

我们现在已经涵盖了标准 GAN 和 WGAN-GP 之间的所有关键区别。回顾一下：

+   WGAN-GP 使用 Wasserstein 损失。

+   WGAN-GP 使用标签 1 表示真实和-1 表示虚假进行训练。

+   在批评家的最后一层没有 Sigmoid 激活。

+   在评论者的损失函数中包含一个梯度惩罚项。

+   对生成器进行多次更新之前多次训练评论者。

+   评论中没有批量归一化层。

## WGAN-GP 的分析

让我们看一下生成器在训练 25 个时期后的一些示例输出（图 4-14）。

![](img/gdl2_0414.png)

###### 图 4-14\. WGAN-GP 面部示例

模型已经学会了面部的重要高级属性，没有出现模式坍塌的迹象。

我们还可以看到模型的损失函数随时间的演变（图 4-15）—评论者和生成器的损失函数都非常稳定和收敛。

如果我们将 WGAN-GP 的输出与上一章的 VAE 输出进行比较，我们可以看到 GAN 图像通常更清晰—特别是头发和背景之间的定义。这在一般情况下是正确的；VAE 倾向于产生模糊颜色边界的柔和图像，而众所周知，GAN 倾向于产生更清晰、更明确定义的图像。

![](img/gdl2_0415.png)

###### 图 4-15\. WGAN-GP 损失曲线：评论者损失（`epoch_c_loss`）分解为 Wasserstein 损失（`epoch_c_wass`）和梯度惩罚损失（`epoch_c_gp`）

同样，GAN 通常比 VAE 更难训练，并需要更长的时间达到令人满意的质量。然而，今天许多最先进的生成模型都是基于 GAN 的，因为在 GPU 上训练大规模 GAN 并花费更长时间的回报是显著的。

# 条件 GAN（CGAN）

到目前为止，在本章中，我们已经构建了能够从给定的训练集生成逼真图像的 GAN。然而，我们无法控制我们想要生成的图像类型—例如，男性或女性的面孔，或大砖或小砖。我们可以从潜在空间中随机采样一个点，但我们无法轻松地了解在选择潜在变量的情况下将产生什么样的图像。

在本章的最后部分，我们将把注意力转向构建一个能够控制输出的 GAN—所谓的*条件 GAN*。这个想法最早是在 2014 年由 Mirza 和 Osindero 在“条件生成对抗网络”中首次提出的，是对 GAN 架构的一个相对简单的扩展。

# 运行此示例的代码

此示例的代码可以在位于书籍存储库中的*notebooks/04_gan/03_cgan/cgan.ipynb*的 Jupyter 笔记本中找到。

代码已经从由 Sayak Paul 创建的优秀[CGAN 教程](https://oreil.ly/Ey11I)中调整，该教程可在 Keras 网站上找到。

## CGAN 架构

在这个例子中，我们将在面部数据集的*金发*属性上对我们的 CGAN 进行条件化。也就是说，我们可以明确指定我们是否想要生成一张有金发的图像。这个标签作为 CelebA 数据集的一部分提供。

高级 CGAN 架构如图 4-16 所示。

![](img/gdl2_0416.png)

###### 图 4-16\. CGAN 中生成器和评论者的输入和输出

标准 GAN 和 CGAN 之间的关键区别在于，在 CGAN 中，我们向生成器和评论者传递与标签相关的额外信息。在生成器中，这只是作为一个独热编码向潜在空间样本附加。在评论者中，我们将标签信息作为额外通道添加到 RGB 图像中。我们通过重复独热编码向量来填充与输入图像相同形状的方式来实现这一点。

CGAN 之所以有效是因为评论者现在可以访问有关图像内容的额外信息，因此生成器必须确保其输出与提供的标签一致，以继续愚弄评论者。如果生成器生成了与图像标签不符的完美图像，评论者将能够简单地判断它们是假的，因为图像和标签不匹配。

###### 提示

在我们的示例中，我们的独热编码标签长度为 2，因为有两个类别（金发和非金发）。但是，您可以有任意数量的标签——例如，您可以在 Fashion-MNIST 数据集上训练一个 CGAN，以输出 10 种不同的时尚物品之一，通过将长度为 10 的独热编码标签向量合并到生成器的输入中，并将 10 个额外的独热编码标签通道合并到评论家的输入中。

我们需要对架构进行的唯一更改是将标签信息连接到生成器和评论家的现有输入中，如示例 4-10 所示。

##### 示例 4-10。CGAN 中的输入层

```py
critic_input = layers.Input(shape=(64, 64, 3)) # ①
label_input = layers.Input(shape=(64, 64, 2))
x = layers.Concatenate(axis = -1)([critic_input, label_input])
...
generator_input = layers.Input(shape=(32,)) # ②
label_input = layers.Input(shape=(2,))
x = layers.Concatenate(axis = -1)([generator_input, label_input])
x = layers.Reshape((1,1, 34))(x)
...
```

①

图像通道和标签通道分别传递给评论家并连接。

②

潜在向量和标签类别分别传递给生成器，并在重塑之前连接。

## 训练 CGAN

我们还需要对 CGAN 的`train_step`进行一些更改，以匹配生成器和评论家的新输入格式，如示例 4-11 所示。

##### 示例 4-11。CGAN 的`train_step`
