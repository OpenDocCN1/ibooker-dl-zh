
我们希望最小化的值（一如既往）是数据的负对数似然：

<math alttext="script upper L equals minus double-struck upper E Subscript x tilde normal d normal a normal t normal a Baseline left-bracket log p Subscript theta Baseline left-parenthesis bold x right-parenthesis right-bracket" display="block"><mrow><mi>ℒ</mi> <mo>=</mo> <mo>-</mo> <msub><mi>𝔼</mi> <mrow><mi>x</mi><mo>∼</mo> <mi>data</mi></mrow></msub> <mfenced separators="" open="[" close="]"><mo form="prefix">log</mo> <msub><mi>p</mi> <mi>θ</mi></msub> <mrow><mo>(</mo> <mi>𝐱</mi> <mo>)</mo></mrow></mfenced></mrow></math>

当 <math alttext="p Subscript theta Baseline left-parenthesis bold x right-parenthesis"><mrow><msub><mi>p</mi> <mi>θ</mi></msub> <mrow><mo>(</mo> <mi>𝐱</mi> <mo>)</mo></mrow></mrow></math> 具有玻尔兹曼分布的形式，能量函数为 <math alttext="upper E Subscript theta Baseline left-parenthesis bold x right-parenthesis"><mrow><msub><mi>E</mi> <mi>θ</mi></msub> <mrow><mo>(</mo> <mi>𝐱</mi> <mo>)</mo></mrow></mrow></math> ，可以证明该值的梯度可以写成以下形式（Oliver Woodford 的“对比散度笔记”进行完整推导）：⁵

<math alttext="StartLayout 1st Row 1st Column normal nabla Subscript theta Baseline script upper L 2nd Column equals 3rd Column double-struck upper E Subscript x tilde normal d normal a normal t normal a Baseline left-bracket normal nabla Subscript theta Baseline upper E Subscript theta Baseline left-parenthesis bold x right-parenthesis right-bracket minus double-struck upper E Subscript x tilde normal m normal o normal d normal e normal l Baseline left-bracket normal nabla Subscript theta Baseline upper E Subscript theta Baseline left-parenthesis bold x right-parenthesis right-bracket EndLayout" display="block"><mtable displaystyle="true"><mtr><mtd columnalign="right"><mrow><msub><mi>∇</mi> <mi>θ</mi></msub> <mi>ℒ</mi></mrow></mtd> <mtd><mo>=</mo></mtd> <mtd columnalign="left"><mrow><msub><mi>𝔼</mi> <mrow><mi>x</mi><mo>∼</mo> <mi>data</mi></mrow></msub> <mfenced separators="" open="[" close="]"><msub><mi>∇</mi> <mi>θ</mi></msub> <msub><mi>E</mi> <mi>θ</mi></msub> <mrow><mo>(</mo> <mi>𝐱</mi> <mo>)</mo></mrow></mfenced> <mo>-</mo> <msub><mi>𝔼</mi> <mrow><mi>x</mi><mo>∼</mo> <mi>model</mi></mrow></msub> <mfenced separators="" open="[" close="]"><msub><mi>∇</mi> <mi>θ</mi></msub> <msub><mi>E</mi> <mi>θ</mi></msub> <mrow><mo>(</mo> <mi>𝐱</mi> <mo>)</mo></mrow></mfenced></mrow></mtd></mtr></mtable></math>

这在直觉上是有很多意义的——我们希望训练模型输出真实观察的大负能量分数，并为生成的假观察输出大正能量分数，以便这两个极端之间的对比尽可能大。

换句话说，我们可以计算真实和假样本的能量分数之间的差异，并将其用作我们的损失函数。

要计算假样本的能量分数，我们需要能够从分布 <math alttext="p Subscript theta Baseline left-parenthesis bold x right-parenthesis"><mrow><msub><mi>p</mi> <mi>θ</mi></msub> <mrow><mo>(</mo> <mi>𝐱</mi> <mo>)</mo></mrow></mrow></math> 中精确抽样，但由于不可解的分母，这是不可能的。相反，我们可以使用 Langevin 采样过程生成一组能量分数较低的观察。这个过程需要运行无限多步才能产生完美样本（显然是不切实际的），因此我们运行一些小步数，假设这足以产生有意义的损失函数。

我们还维护一个来自先前迭代的样本缓冲区，这样我们可以将其用作下一批的起点，而不是纯随机噪声。生成采样缓冲区的代码如示例 7-5 所示。

##### 示例 7-5。`缓冲区`

```py
class Buffer:
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.examples = [
            tf.random.uniform(shape = (1, 32, 32, 1)) * 2 - 1
            for _ in range(128)
        ] # ①

    def sample_new_exmps(self, steps, step_size, noise):
        n_new = np.random.binomial(128, 0.05) # ②
        rand_imgs = (
            tf.random.uniform((n_new, 32, 32, 1)) * 2 - 1
        )
        old_imgs = tf.concat(
            random.choices(self.examples, k=128-n_new), axis=0
        ) # ③
        inp_imgs = tf.concat([rand_imgs, old_imgs], axis=0)
        inp_imgs = generate_samples(
            self.model, inp_imgs, steps=steps, step_size=step_size, noise = noise
        ) # ④
        self.examples = tf.split(inp_imgs, 128, axis = 0) + self.examples # ⑤
        self.examples = self.examples[:8192]
        return inp_imgs
```

①

采样缓冲区用一批随机噪声初始化。

②

平均而言，每次有 5%的观察是从头开始生成的（即，随机噪声）。

③

其余的随机从现有缓冲区中提取。

④

这些观察被连接并通过 Langevin 采样器运行。

⑤

生成的样本被添加到缓冲区中，缓冲区被修剪为最多 8,192 个观察。

图 7-5 显示了对比散度的一个训练步骤。真实观察的分数被算法推低，而假观察的分数被拉高，每一步之后都不考虑对这些分数进行归一化。

![](img/gdl2_0705.png)

###### 图 7-5。对比散度的一步

我们可以编写对比散度算法的训练步骤，如示例 7-6 所示，在自定义 Keras 模型中。

##### 示例 7-6。使用对比散度训练的 EBM

```py
class EBM(models.Model):
    def __init__(self):
        super(EBM, self).__init__()
        self.model = model
        self.buffer = Buffer(self.model)
        self.alpha = 0.1
        self.loss_metric = metrics.Mean(name="loss")
        self.reg_loss_metric = metrics.Mean(name="reg")
        self.cdiv_loss_metric = metrics.Mean(name="cdiv")
        self.real_out_metric = metrics.Mean(name="real")
        self.fake_out_metric = metrics.Mean(name="fake")

    @property
    def metrics(self):
        return [
            self.loss_metric,
            self.reg_loss_metric,
            self.cdiv_loss_metric,
            self.real_out_metric,
            self.fake_out_metric
        ]

    def train_step(self, real_imgs):
        real_imgs += tf.random.normal(
            shape=tf.shape(real_imgs), mean = 0, stddev = 0.005
        ) # ①
        real_imgs = tf.clip_by_value(real_imgs, -1.0, 1.0)
        fake_imgs = self.buffer.sample_new_exmps(
            steps=60, step_size=10, noise = 0.005
        ) # ②
        inp_imgs = tf.concat([real_imgs, fake_imgs], axis=0)
        with tf.GradientTape() as training_tape:
            real_out, fake_out = tf.split(self.model(inp_imgs), 2, axis=0) # ③
            cdiv_loss = tf.reduce_mean(fake_out, axis = 0) - tf.reduce_mean(
                real_out, axis = 0
            ) # ④
            reg_loss = self.alpha * tf.reduce_mean(
                real_out ** 2 + fake_out ** 2, axis = 0
            ) # ⑤
            loss = reg_loss + cdiv_loss
        grads = training_tape.gradient(loss, self.model.trainable_variables) # ⑥
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )
        self.loss_metric.update_state(loss)
        self.reg_loss_metric.update_state(reg_loss)
        self.cdiv_loss_metric.update_state(cdiv_loss)
        self.real_out_metric.update_state(tf.reduce_mean(real_out, axis = 0))
        self.fake_out_metric.update_state(tf.reduce_mean(fake_out, axis = 0))
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, real_imgs): # ⑦
        batch_size = real_imgs.shape[0]
        fake_imgs = tf.random.uniform((batch_size, 32, 32, 1)) * 2 - 1
        inp_imgs = tf.concat([real_imgs, fake_imgs], axis=0)
        real_out, fake_out = tf.split(self.model(inp_imgs), 2, axis=0)
        cdiv = tf.reduce_mean(fake_out, axis = 0) - tf.reduce_mean(
            real_out, axis = 0
        )
        self.cdiv_loss_metric.update_state(cdiv)
        self.real_out_metric.update_state(tf.reduce_mean(real_out, axis = 0))
        self.fake_out_metric.update_state(tf.reduce_mean(fake_out, axis = 0))
        return {m.name: m.result() for m in self.metrics[2:]}

ebm = EBM()
ebm.compile(optimizer=optimizers.Adam(learning_rate=0.0001), run_eagerly=True)
ebm.fit(x_train, epochs=60, validation_data = x_test,)
```

①

为真实图像添加少量随机噪声，以避免模型过度拟合训练集。

②

一组假图像从缓冲区中抽样。

③

真实和假图像通过模型运行以产生真实和假分数。

④

对比散度损失简单地是真实和假观察的分数之间的差异。

⑤

添加正则化损失以避免分数变得过大。

⑥

通过反向传播计算网络权重相对于损失函数的梯度。

⑦

`test_step` 用于在验证过程中计算一组随机噪声和训练集中的数据之间的对比散度。它可以作为衡量模型训练效果的指标（见下一节）。

## 能量基模型的分析

训练过程中的损失曲线和支持指标显示在 Figure 7-6 中。

![](img/gdl2_0706.png)

###### 图 7-6。EBM 训练过程的损失曲线和指标

首先，注意到在训练步骤中计算的损失在各个周期中大致保持不变且较小。虽然模型不断改进，但与训练集中的真实图像进行比较的缓冲区中生成的图像质量也在提高，因此我们不应该期望训练损失显著下降。

因此，为了评估模型性能，我们还建立了一个验证过程，该过程不从缓冲区中采样，而是对一组随机噪声进行评分，并将其与训练集中的示例进行比较。如果模型正在改进，我们应该看到对比散度随着周期的增加而下降（即，它在区分随机噪声和真实图像方面变得更好），如 Figure 7-6 中所示。

从 EBM 生成新样本只需运行 Langevin 采样器进行大量步骤，从一个静止状态（随机噪声）开始，如 Example 7-7 中所示。观测被迫 *下坡*，沿着相对于输入的评分函数的梯度，以便在噪声中出现一个合理的观测。

##### 示例 7-7。使用 EBM 生成新观测

```py
start_imgs = np.random.uniform(size = (10, 32, 32, 1)) * 2 - 1
gen_img = generate_samples(
    ebm.model,
    start_imgs,
    steps=1000,
    step_size=10,
    noise = 0.005,
    return_img_per_step=True,
)
```

在经过 50 个周期的训练后，采样器生成的一些观测示例显示在 Figure 7-7 中。

![](img/gdl2_0707.png)
