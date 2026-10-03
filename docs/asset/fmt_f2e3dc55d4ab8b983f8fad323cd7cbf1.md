
我们应用`Conv2DTranspose`层来沿一个轴扩展张量的大小，使其与`N_BARS`的长度相同。

我们使用`Reshape`层去除不必要的额外维度。

我们使用卷积操作而不是要求两个独立的向量进入网络的原因是，我们希望网络学习如何以一种一致的方式让一个小节跟随另一个小节。使用神经网络沿着时间轴扩展输入向量意味着模型有机会学习音乐如何跨越小节流动，而不是将每个小节视为完全独立于上一个的。

### 和弦、风格、旋律和 groove

现在让我们更仔细地看一下喂给生成器的四种不同输入：

**和弦**

和弦输入是一个长度为`Z_DIM`的单一噪声向量。这个向量的作用是控制音乐随时间的总体进展，跨越轨道共享，因此我们使用`TemporalNetwork`将这个单一向量转换为每个小节的不同潜在向量。请注意，虽然我们称这个输入为和弦，但它实际上可以控制音乐中每个小节变化的任何内容，比如一般的节奏风格，而不是特定于任何特定轨道。

**风格**

风格输入也是长度为`Z_DIM`的向量。这个向量在不经过转换的情况下传递，因此在所有小节和轨道上都是相同的。它可以被视为控制乐曲整体风格的向量（即，它会一致地影响所有小节和轨道）。

**旋律**

旋律输入是一个形状为`[N_TRACKS, Z_DIM]`的数组—也就是说，我们为每个轨道提供长度为`Z_DIM`的随机噪声向量。

这些向量中的每一个都通过轨道特定的`TemporalNetwork`，其中轨道之间的权重不共享。输出是每个轨道的每个小节的长度为`Z_DIM`的向量。因此，模型可以使用这些输入向量来独立地微调每个小节和轨道的内容。

**Groove**

groove 输入也是一个形状为`[N_TRACKS, Z_DIM]`的数组，即每个轨道的长度为`Z_DIM`的随机噪声向量。与旋律输入不同，这些向量不通过时间网络，而是直接传递，就像风格向量一样。因此，每个 groove 向量将影响轨道的整体属性，跨越所有小节。

我们可以总结每个 MuseGAN 生成器组件的责任，如表 11-1 所示。

**表 11-1\. MuseGAN 生成器的组件**

|  | 输出在小节之间不同吗？ | 输出在部分之间不同吗？ |
| --- | --- | --- |
| 风格 | Ｘ | Ｘ |
| Groove | Ｘ | ✓ |
| 和弦 | ✓ | Ｘ |
| 旋律 | ✓ | ✓ |

MuseGAN 生成器的最后一部分是*小节生成器*—让我们看看如何使用它来将和弦、风格、旋律和 groove 组件的输出粘合在一起。

### 小节生成器

小节生成器接收四个潜在向量——来自和弦、风格、旋律和 groove 组件。这些被连接起来产生长度为`4 * Z_DIM`的输入向量。输出是单个轨道的单个小节的钢琴卷表示—即，形状为`[1, n_steps_per_bar, n_pitches, 1]`的张量。

小节生成器只是一个使用卷积转置层来扩展输入向量的时间和音高维度的神经网络。我们为每个轨道创建一个小节生成器，轨道之间的权重不共享。构建`BarGenerator`的 Keras 代码在示例 11-7 中给出。

**示例 11-7\. 构建`BarGenerator`**

```py
def BarGenerator():

    input_layer = layers.Input(shape=(Z_DIM * 4,), name='bar_generator_input') # ①

    x = layers.Dense(1024)(input_layer) # ②
    x = layers.BatchNormalization(momentum = 0.9)(x)
    x = layers.Activation('relu')(x)
    x = layers.Reshape([2,1,512])(x)

    x = conv_t(x, f=512, k=(2,1), s=(2,1), a= 'relu',  p = 'same', bn = True) # ③
    x = conv_t(x, f=256, k=(2,1), s=(2,1), a= 'relu', p = 'same', bn = True)
    x = conv_t(x, f=256, k=(2,1), s=(2,1), a= 'relu', p = 'same', bn = True)
    x = conv_t(x, f=256, k=(1,7), s=(1,7), a= 'relu', p = 'same', bn = True) # ④
    x = conv_t(x, f=1, k=(1,12), s=(1,12), a= 'tanh', p = 'same', bn = False) # ⑤

    output_layer = layers.Reshape([1, N_STEPS_PER_BAR , N_PITCHES ,1])(x) # ⑥

    return models.Model(input_layer, output_layer)
```

① bar 生成器的输入是长度为`4 * Z_DIM`的向量。

② 通过一个`Dense`层后，我们重新塑造张量以准备进行卷积转置操作。

③ 首先我们沿着时间步长轴扩展张量…​

④ …​然后沿着音高轴。

⑤ 最终层应用了 tanh 激活，因为我们将使用 WGAN-GP（需要 tanh 输出激活）来训练网络。

⑥ 张量被重塑以添加两个大小为 1 的额外维度，以准备与其他小节和轨道连接。

### 将所有内容整合在一起

最终，MuseGAN 生成器接受四个输入噪声张量（和弦、风格、旋律和节奏），并将它们转换为一个多轨多小节乐谱。构建 MuseGAN 生成器的 Keras 代码在示例 11-8 中提供。

**示例 11-8。构建 MuseGAN 生成器**

```py
def Generator():
    chords_input = layers.Input(shape=(Z_DIM,), name='chords_input') # ①
    style_input = layers.Input(shape=(Z_DIM,), name='style_input')
    melody_input = layers.Input(shape=(N_TRACKS, Z_DIM), name='melody_input')
    groove_input = layers.Input(shape=(N_TRACKS, Z_DIM), name='groove_input')

    chords_tempNetwork = TemporalNetwork() # ②
    chords_over_time = chords_tempNetwork(chords_input)

    melody_over_time = [None] * N_TRACKS
    melody_tempNetwork = [None] * N_TRACKS
    for track in range(N_TRACKS):
        melody_tempNetwork[track] = TemporalNetwork() # ③
        melody_track = layers.Lambda(lambda x, track = track: x[:,track,:])(
            melody_input
        )
        melody_over_time[track] = melody_tempNetworktrack

    barGen = [None] * N_TRACKS
    for track in range(N_TRACKS):
        barGen[track] = BarGenerator() # ④

    bars_output = [None] * N_BARS
    c = [None] * N_BARS
    for bar in range(N_BARS): # ⑤
        track_output = [None] * N_TRACKS

        c[bar] = layers.Lambda(lambda x, bar = bar: x[:,bar,:])(chords_over_time)
        s = style_input

        for track in range(N_TRACKS):

            m = layers.Lambda(lambda x, bar = bar: x[:,bar,:])(
                melody_over_time[track]
            )
            g = layers.Lambda(lambda x, track = track: x[:,track,:])(
                groove_input
            )

            z_input = layers.Concatenate(
                axis = 1, name = 'total_input_bar_{}_track_{}'.format(bar, track)
            )([c[bar],s,m,g])

            track_output[track] = barGentrack

        bars_output[bar] = layers.Concatenate(axis = -1)(track_output)

    generator_output = layers.Concatenate(axis = 1, name = 'concat_bars')(
        bars_output
    ) # ⑥

    return models.Model(
        [chords_input, style_input, melody_input, groove_input], generator_output
    ) # ⑦

generator = Generator()
```

① 定义生成器的输入。

② 通过时间网络传递和弦输入。

③ 通过时间网络传递旋律输入。

④ 为每个轨道创建一个独立的小节生成器网络。

⑤ 循环遍历轨道和小节，为每种组合创建一个生成的小节。

⑥ 将所有内容连接在一起形成单个输出张量。

⑦ MuseGAN 模型接受四个不同的噪声张量作为输入，并输出一个生成的多轨多小节乐谱。

## MuseGAN 评论家

与生成器相比，评论家的架构要简单得多（这在 GAN 中经常是这样）。

评论家试图区分生成器创建的完整多轨多小节乐谱和巴赫赞美诗的真实节选。它是一个卷积神经网络，主要由将乐谱折叠成单个输出预测的`Conv3D`层组成。

### Conv3D 层

到目前为止，在本书中，我们只使用了适用于三维输入图像（宽度、高度、通道）的`Conv2D`层。在这里，我们必须使用`Conv3D`层，它们类似于`Conv2D`层，但接受四维输入张量（`n_bars`、`n_steps_per_bar`、`n_pitches`、`n_tracks`）。

我们在评论家中不使用批量归一化层，因为我们将使用 WGAN-GP 框架来训练 GAN，这是不允许的。

构建评论家的 Keras 代码在示例 11-9 中给出。

**示例 11-9。构建 MuseGAN 评论家**

```py
def conv(x, f, k, s, p):
    x = layers.Conv3D(filters = f
                , kernel_size = k
                , padding = p
                , strides = s
                , kernel_initializer = initializer
                )(x)
    x = layers.LeakyReLU()(x)
    return x

def Critic():
    critic_input = layers.Input(
        shape=(N_BARS, N_STEPS_PER_BAR, N_PITCHES, N_TRACKS),
        name='critic_input'
    ) # ①

    x = critic_input
    x = conv(x, f=128, k = (2,1,1), s = (1,1,1), p = 'valid') # ②
    x = conv(x, f=128, k = (N_BARS - 1,1,1), s = (1,1,1), p = 'valid')

    x = conv(x, f=128, k = (1,1,12), s = (1,1,12), p = 'same') # ③
    x = conv(x, f=128, k = (1,1,7), s = (1,1,7), p = 'same')

    x = conv(x, f=128, k = (1,2,1), s = (1,2,1), p = 'same') # ④
    x = conv(x, f=128, k = (1,2,1), s = (1,2,1), p = 'same')
    x = conv(x, f=256, k = (1,4,1), s = (1,2,1), p = 'same')
    x = conv(x, f=512, k = (1,3,1), s = (1,2,1), p = 'same')

    x = layers.Flatten()(x)

    x = layers.Dense(1024, kernel_initializer = initializer)(x)
    x = layers.LeakyReLU()(x)

    critic_output = layers.Dense(
        1, activation=None, kernel_initializer = initializer
    )(x) # ⑤

    return models.Model(critic_input, critic_output)

critic = Critic()
```

① 评论家的输入是一个多轨多小节乐谱数组，每个形状为`[N_BARS, N_STEPS_PER_BAR, N_PITCHES, N_TRACKS]`。

② 首先，我们沿着小节轴折叠张量。由于我们使用的是 4D 张量，所以在评论家中应用`Conv3D`层。

③ 接下来，我们沿着音高轴折叠张量。

④ 最后，我们沿着时间步轴折叠张量。

⑤ 输出是一个具有单个单元且没有激活函数的`Dense`层，这是 WGAN-GP 框架所需的。

## MuseGAN 的分析

我们可以通过生成一个乐谱，然后调整一些输入噪声参数来查看对输出的影响来进行一些实验。

生成器的输出是一个值范围在[-1, 1]的数组（由于最终层的 tanh 激活函数）。为了将其转换为每个轨道的单个音符，我们选择每个时间步的所有 84 个音高中具有最大值的音符。在原始的 MuseGAN 论文中，作者使用了阈值 0，因为每个轨道可以包含多个音符；然而，在这种情况下，我们可以简单地取最大值来确保每个时间步每个轨道恰好有一个音符，这与巴赫赞美诗的情况相同。

图 11-16 显示了模型从随机正态分布的噪声向量生成的乐谱（左上角）。我们可以通过欧几里德距离找到数据集中最接近的乐谱，并检查我们生成的乐谱是否是数据集中已经存在的音乐片段的副本——最接近的乐谱显示在其正下方，我们可以看到它与我们生成的乐谱并不相似。

现在让我们玩弄输入噪声来微调我们生成的乐谱。首先，我们可以尝试改变和弦噪声向量——图 11-16 中左下角的乐谱显示了结果。我们可以看到每个轨道都已经改变，正如预期的那样，而且两个小节展现出不同的特性。在第二小节中，贝斯线更加动态，顶部音乐线的音高比第一小节更高。这是因为影响两个小节的潜在向量是不同的，因为输入和弦向量通过了一个时间网络。

**图 11-16。MuseGAN 预测乐谱的示例，显示训练数据中最接近的真实乐谱以及通过改变输入噪声而影响生成乐谱的情况**

## 总结

当我们改变风格向量（右上角）时，两个小节以类似的方式改变。整个乐段的风格已经从原始生成的乐谱中改变，以一种一致的方式（即，相同的潜在向量被用来调整所有轨道和小节）。

我们还可以通过旋律和节奏输入单独改变轨道。在图 11-16 中间右侧的乐谱中，我们可以看到仅改变顶部音乐线的旋律噪声输入的效果。所有其他部分保持不变，但顶部音符发生了显著变化。此外，我们可以看到顶部音乐线两个小节之间的节奏变化：第二小节比第一小节更动态，包含比第一小节更快的音符。

最后，图中右下角的乐谱显示了当我们仅改变贝斯的节奏输入参数时预测的乐谱。同样，所有其他部分保持不变，但贝斯部分是不同的。此外，贝斯的整体模式在小节之间保持相似，这是我们所期望的。

这展示了如何每个输入参数可以直接影响生成的音乐序列的高级特征，就像我们之前能够调整 VAE 和 GAN 的潜在向量以改变生成图像的外观一样。模型的一个缺点是必须预先指定要生成的小节数。为了解决这个问题，作者展示了模型的一个扩展，允许将先前的小节作为输入馈入，使模型能够通过不断将最近预测的小节作为额外输入来生成长形乐谱。

在本章中，我们探讨了两种不同类型的音乐生成模型：Transformer 和 MuseGAN。

Transformer 的设计类似于我们在第九章中看到的用于文本生成的网络。音乐和文本生成有很多共同点，通常可以同时用于两者的类似技术。我们通过将两个输入和输出流（音符和持续时间）纳入 Transformer 架构来扩展了 Transformer 架构。我们看到模型能够通过准确生成巴赫音乐来学习关于调式和音阶等概念。
