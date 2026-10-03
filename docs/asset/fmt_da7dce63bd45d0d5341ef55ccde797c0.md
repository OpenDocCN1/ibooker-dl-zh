
### 7. 对于结合模型，我们只训练生成器。

### 8. 判别器确定转换图像的有效性

### 9. 结合模型训练生成器欺骗判别器

最后一点需要从前面的代码中澄清：`combined`模型的输出以六个列表的形式出现。这是因为我们总是得到有效性（来自判别器）、重建和身份损失——一个用于 A-B-A，一个用于 B-A-B 循环——因此是六个。前两个是平方误差，其余是平均绝对误差。相对权重受前面描述的`lambda`因素影响。

#### 9.7.2. 构建生成器

接下来，我们在清单 9.4 中构建生成器代码，该代码使用我们在第 9.5.2 节中描述的跳跃连接。这是 U-Net 架构。这个架构比一些实现中使用的 ResNet 架构更简单。在我们的生成器函数中，我们首先定义辅助函数：

1. 将`conv2d()`函数定义为如下：
   - 标准的 2D 卷积层
   - Leaky ReLU 激活
   - 实例归一化[^8]

   > [^8] 实例归一化类似于第四章中的批归一化，除了我们不是基于整个批次的全部信息进行归一化，而是对每个通道中的每个特征图分别进行归一化。实例归一化通常会产生更好的图像质量，适用于风格迁移或图像到图像翻译等任务——这正是 CycleGAN 所需要的！

2. 定义`deconv2d()`函数作为一个转置卷积（即 *反卷积*）层，它执行以下操作：

   > [^9] 这里，*转置卷积*是——有人认为——一个更准确的术语。然而，只需将其视为卷积的相反，或反卷积。

   - 提升输入层`input_layer`
   - 如果我们设置了 dropout 率，可能会应用 dropout
   - 总是应用`InstanceNormalization`
   - 更重要的是，在其输出层和从图 9.4 的下采样部分对应维度的层之间创建跳过连接

   ##### 注意
   在步骤 2d 中，我们使用简单的`UpSampling2D`，这不是一个学习参数，而是使用最近邻插值。

然后我们创建实际的生成器：

1. 将输入（128 × 128 × 3）分配给`d0`。
2. 通过卷积层`d1`运行，得到 64 × 64 × 32 层。
3. 将`d1`（64 × 64 × 32）应用`conv2d`得到 32 × 32 × 64（`d2`）。
4. 将`d2`（32 × 32 × 64）应用`conv2d`得到 16 × 16 × 128（`d3`）。
5. 将`d3`（16 × 16 × 128）应用`conv2d`得到 8 × 8 × 256（`d4`）。
6. `u1`：上采样`d4`并在`d3`和`u1`之间创建跳过连接。
7. `u2`：上采样`u1`并在`d2`和`u2`之间创建跳过连接。
8. `u3`：上采样`u2`并在`d1`和`u3`之间创建跳过连接。
9. `u4`：使用常规上采样得到 128 × 128 × 64 图像。
10. 使用常规的 2D 卷积去除额外的特征图，只得到 128 × 128 × 3（高度 × 宽度 × 颜色通道）

##### 列表 9.4. 构建生成器

```
def build_generator(self):
    """U-Net Generator"""

    def conv2d(layer_input, filters, f_size=4):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size,
                   strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4,
                 dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1,
                   padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = InstanceNormalization()(u)
        u = Concatenate()([u, skip_input])
        return u

    d0 = Input(shape=self.img_shape)     # 1 图像输入
    d1 = conv2d(d0, self.gf)             # 2 下采样
    d2 = conv2d(d1, self.gf * 2)         # 2 下采样
    d3 = conv2d(d2, self.gf * 4)         # 2 下采样
    d4 = conv2d(d3, self.gf * 8)         # 2 下采样

    u1 = deconv2d(d4, d3, self.gf * 4)   # 3 上采样
    u2 = deconv2d(u1, d2, self.gf * 2)   # 3 上采样
    u3 = deconv2d(u2, d1, self.gf)       # 3 上采样

    u4 = UpSampling2D(size=2)(u3)
    output_img = Conv2D(self.channels, kernel_size=4,
                        strides=1, padding='same', activation='tanh')(u4)

    return Model(d0, output_img)
```

#### 9.7.3. 构建判别器

现在是判别器方法，它使用一个辅助函数来创建由 2D 卷积、`LeakyReLU`和可选的`InstanceNormalization`组成的层。

我们以以下方式应用这些层，如图列表 9.5 所示：

1. 我们将输入图像（128 × 128 × 3）分配给`d1`（64 × 64 × 64）。
2. 我们将`d1`（64 × 64 × 64）分配给`d2`（32 × 32 × 128）。
3. 我们将`d2`（32 × 32 × 128）分配给`d3`（16 × 16 × 256）。
4. 我们将`d3`（16 × 16 × 256）分配给`d4`（8 × 8 × 512）。
5. 我们将`d4`（8 × 8 × 512）通过`conv2d`展平为 8 × 8 × 1。

##### 列表 9.5. 构建判别器

```
def build_discriminator(self):

    def d_layer(layer_input, filters, f_size=4, normalization=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size,
                   strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if normalization:
            d = InstanceNormalization()(d)
        return d

    img = Input(shape=self.img_shape)

    d1 = d_layer(img, self.df, normalization=False)
    d2 = d_layer(d1, self.df * 2)
    d3 = d_layer(d2, self.df * 4)
    d4 = d_layer(d3, self.df * 8)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    return Model(img, validity)
```

#### 9.7.4. 训练 CycleGAN

所有网络编写完毕后，我们现在将实现创建训练循环的方法。对于 CycleGAN 训练算法，每个训练迭代的细节如下。

**CycleGAN 训练算法**

*对于*每个训练迭代*执行*

1. 训练判别器：
   - 从每个域（*imgs[A]* 和 *imgs[B]*）中随机抽取一个图像的小批量。
   - 使用生成器 *G[AB]* 将 *imgs[A]* 转换为域 B，反之亦然，使用 *G[BA]*。
   - 计算 *D[A]*(*imgs[A]*, 1) 和 *D[A]*(*G[BA]*(*imgs[B]*), 0) 以获取 A 中真实图像和从 B 转换的图像的损失。然后将这两个损失相加。*D[A]* 中的 1 和 0 作为标签。
   - 计算 *D[B]*(*imgs[B]*, 1) 和 *D[B]*(*G[AB]*(*imgs[A]*), 0) 以获取 B 中真实图像和从 A 转换的图像的损失。然后将这两个损失相加。*D[B]* 中的 1 和 0 作为标签。
2. 将步骤 c 和 d 的损失相加以获得总判别器损失。训练生成器：
   - 我们使用组合模型来
     - 输入来自域 A (*imgs[A]*) 和 B (*imgs[B]*) 的图像
     - 输出如下
       1. A 的有效性：*D[A]*(*G[BA]*(*imgs[B]*))
       2. B 的有效性：*D[B]*(*G[AB]*(*imgs[A]*))
       3. 重建 A：*G[BA]*(*G[AB]*(*imgs[A]*))
       4. 重建 B：*G[AB]*(*G[BA]*(*imgs[B]*))
       5. A 的身份映射：*G[BA]*(*imgs[A]*))
       6. B 的身份映射：*G[AB]*(*imgs[B]*))
   - 然后，我们根据循环一致性损失、身份损失和对抗损失更新两个生成器的参数
     - 标量（判别器概率）的均方误差 (MSE)
     - 均方绝对误差 (MAE) 用于图像（无论是重建的还是身份映射的）

*结束 for*

下面的列表实现了这个 CycleGAN 训练算法。

##### 列表 9.6. 训练 CycleGAN

```
def train(self, epochs, batch_size=1, sample_interval=50):

    start_time = datetime.datetime.now()

    valid = np.ones((batch_size,) + self.disc_patch)                  # 1 对抗损失真实值
    fake = np.zeros((batch_size,) + self.disc_patch)

    for epoch in range(epochs):
        for batch_i, (imgs_A, imgs_B) in enumerate(
            self.data_loader.load_batch(batch_size)):

            fake_B = self.g_AB.predict(imgs_A)                        # 2 现在我们开始训练判别器。这些行将图像转换为相反的域。
            fake_A = self.g_BA.predict(imgs_B)                        # 2

            dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)     # 3 训练判别器（原始图像 = 真实 / 转换 = 假）
            dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)      # 3
            dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)        # 3

            dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)     # 3
            dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)      # 3
            dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)        # 3

            d_loss = 0.5 * np.add(dA_loss, dB_loss)                   # 4 总判别器损失

            g_loss = self.combined.train_on_batch([imgs_A, imgs_B],   # 5 训练生成器
                                                  [valid, valid,
                                                   imgs_A, imgs_B,
                                                   imgs_A, imgs_B])
            if batch_i % sample_interval == 0:                        # 6 如果达到保存间隔 => 保存生成的图像样本
                self.sample_images(epoch, batch_i)                    # 7 这个函数与您遇到过的类似，并在 GitHub 仓库中明确说明。
```

#### 9.7.5. 运行 CycleGAN

我们已经编写了所有这些复杂的代码，现在准备好实例化一个 CycleGAN 对象，并查看一些从采样图像中获得的结果：

```
gan = CycleGAN()
gan.train(epochs=100, batch_size=64, sample_interval=10)
```

图 9.7 展示了我们辛勤工作的部分结果。

##### 图 9.7. 苹果转换为橙子，橙子转换为苹果。这些是我们 Jupyter 笔记本中直接显示的结果。（结果可能因随机种子、TensorFlow 和 Keras 的实现以及超参数而略有不同。）

![](img/09fig07_alt.jpg)

### 9.8. 扩展、增强和应用

当你运行这些结果时，我们希望你会像我们一样印象深刻。由于绝对惊人的结果，许多研究人员纷纷改进这项技术。本节详细介绍了 CycleGAN 的扩展，然后讨论了一些 CycleGAN 的应用。

#### 9.8.1. 增强 CycleGAN

“从未配对数据中学习多对多映射的增强 CycleGAN”是对标准 CycleGAN 的一个真正巧妙的扩展，它在翻译过程中注入了潜在空间信息。在斯德哥尔摩的 ICML 2018 会议上提出，增强 CycleGAN 为我们提供了额外的变量，这些变量驱动着生成过程。[^10] 就像我们在条件 GAN 的情况下使用潜在空间一样，我们可以在 CycleGAN 的设置中使用它，超越 CycleGAN 本身所做的工作。

> [^10]
