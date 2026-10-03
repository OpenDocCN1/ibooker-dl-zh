
```py
image_dir = 'BBBC005_v1_images'
label_dir = 'BBBC005_v1_ground_truth'
rows = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
		'M', 'N', 'O', 'P')
blurs = (1, 4, 7, 10, 14, 17, 20, 23, 26, 29, 32, 35, 39, 42, 45, 48)
files = []
labels = []
for f in os.listdir(label_dir):
 if f.endswith('.TIF'):
  for row, blur in zip(rows, blurs):
   fname = f.replace('_F1', '_F%d'%blur).replace('_A', '_%s'%row)
   files.append(os.path.join(image_dir, fname))
   labels.append(os.path.join(label_dir, f))
loader = dc.data.ImageLoader()
dataset = loader.featurize(files, labels)
```

现在我们已经加载和处理了我们的数据集，让我们开始构建一些深度学习模型。和以前一样，我们将把这个数据集分成训练、验证和测试集：

```py
splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
	  dataset, seed=123)
```

我们可以使用什么架构来进行图像分割任务？这不仅仅是使用一个简单的卷积架构，因为我们的输出本身需要是一幅图像（分割掩模）。幸运的是，过去已经有一些适合这个任务的架构。U-Net 架构使用一系列堆叠的卷积来逐渐“下采样”然后“上采样”源图像，如图 7-14 所示。这种架构在图像分割任务中表现良好。

![生物医学图像分割的 U-Net 架构表示。](img/dlls_0714.png)

###### 图 7-14。生物医学图像分割的 U-Net 架构表示。（改编自[弗莱堡大学](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)。）

现在让我们在 DeepChem 中实现 U-Net：

```py
learning_rate = dc.models.tensorgraph.optimizers.ExponentialDecay(0.01, 0.9, 250)
model = dc.models.TensorGraph(learning_rate=learning_rate, 
							  model_dir='segmentation')
features = layers.Feature(shape=(None, 520, 696, 1)) / 255.0
labels = layers.Label(shape=(None, 520, 696, 1)) / 255.0
# Downsample three times.
conv1 = layers.Conv2D(16, kernel_size=5, stride=2, in_layers=features)
conv2 = layers.Conv2D(32, kernel_size=5, stride=2, in_layers=conv1)
conv3 = layers.Conv2D(64, kernel_size=5, stride=2, in_layers=conv2)
# Do a 1x1 convolution.
conv4 = layers.Conv2D(64, kernel_size=1, stride=1, in_layers=conv3)
# Upsample three times.
concat1 = layers.Concat(in_layers=[conv3, conv4], axis=3)
deconv1 = layers.Conv2DTranspose(32, kernel_size=5, stride=2, in_layers=concat1)
concat2 = layers.Concat(in_layers=[conv2, deconv1], axis=3)
deconv2 = layers.Conv2DTranspose(16, kernel_size=5, stride=2, in_layers=concat2)
concat3 = layers.Concat(in_layers=[conv1, deconv2], axis=3)
deconv3 = layers.Conv2DTranspose(1, kernel_size=5, stride=2, in_layers=concat3)
# Compute the final output.
concat4 = layers.Concat(in_layers=[features, deconv3], axis=3)
logits = layers.Conv2D(1, kernel_size=5, stride=1, activation_fn=None, 
					   in_layers=concat4)
output = layers.Sigmoid(logits)
model.add_output(output)
loss = layers.ReduceSum(layers.SigmoidCrossEntropy(in_layers=(labels, logits)))
model.set_loss(loss)
```

这种架构比用于细胞计数的架构稍微复杂一些，但我们使用相同的基本代码结构和堆叠卷积层来实现我们想要的架构。和以前一样，让我们使用一个预训练模型来尝试这个架构。下载预训练模型的指导可在书的[代码库](https://github.com/deepchem/DeepLearningLifeSciences)中找到。一旦你把预训练权重放好，你可以像以前一样加载权重：

```py
model.restore()
```

让我们使用这个模型创建一些掩模。调用`model.predict_on_batch()`允许我们预测一批输入的输出掩模。我们可以通过将我们的掩模与地面真实掩模进行比较并检查重叠部分来检查我们预测的准确性：

```py
scores = []
for x, y, w, id in test_dataset.itersamples():
 y_pred = model.predict_on_batch([x]).squeeze()
 scores.append(np.mean((y>0) == (y_pred>0.5)))
print(np.mean(scores))
```

这应该返回大约 0.9899。这意味着将近 99% 的像素被正确预测了！这是一个不错的结果，但我们应该强调这只是一个玩具学习任务。一个简单的图像处理算法加上亮度阈值可能也能做得差不多。不过，这里展示的原则应该能够适用于更复杂的图像数据集。

好的，现在我们已经使用预训练模型进行了探索，让我们从头开始训练一个 U-Net 模型，进行 50 个 epochs 的训练，看看我们能得到什么结果：

```py
model.fit(train_dataset, nb_epoch=50, checkpoint_interval=100)
```

和之前一样，这种训练需要大量的计算资源，在一台性能良好的 GPU 上可能需要几个小时。在 CPU 系统上训练这个模型可能并不可行。一旦模型训练完成，尝试运行结果并看看你得到了什么。你能达到预训练模型的准确率吗？

## 计算实验

细胞计数和分割是相当直观的任务，因此并不奇怪机器学习模型能够在这类数据集上表现良好。人们可能会合理地问，机器学习模型是否只能做到这些。

幸运的是，答案是否定的！机器学习模型能够捕捉到数据集中微妙的信号。例如，一项研究表明，深度学习模型能够预测原始图像中荧光标记的输出。值得停下来考虑这个结果有多么令人惊讶。正如我们在“为显微镜准备生物样本”中看到的，荧光染色可能是一个相当复杂的过程。令人惊讶的是，深度学习可能能够减少一些必要的准备工作。

这是一个令人兴奋的结果，但值得注意的是，这仍然是一个早期的结果。还需要做大量工作来“加固”这些技术，以便它们能够更广泛地应用。

# 结论

在本章中，我们向您介绍了显微镜学的基础知识以及一些基本的显微系统机器学习方法。我们对现代显微镜学的一些基本问题（尤其是应用于生物问题）进行了广泛介绍，并讨论了深度学习已经产生影响的领域，以及它未来可能产生更大影响的领域。

我们还提供了显微镜学中一些物理和生物学的全面概述，并试图传达为什么这些信息对于那些主要关注构建有效处理显微镜图像和模型管道的开发人员可能是有用的。了解物理原理，比如衍射极限，将使您能够理解为什么使用不同的显微技术以及深度学习可能对该领域的未来至关重要。了解生物样本制备技术将帮助您理解在设计实用显微系统时需要跟踪的元数据和注释类型。

虽然我们对深度学习技术在显微镜学中的潜在应用感到非常兴奋，但我们需要强调这些方法存在一些注意事项。首先，一些最近的研究强调了视觉卷积模型的脆弱性。简单的伪影可能会让这些模型出错并导致重大问题。例如，一个停车标志的图像可能会被轻微扰动，以至于模型将其分类为绿色交通灯。这对于自动驾驶汽车来说将是灾难性的！

鉴于这些证据，值得问一下深度显微镜模型存在哪些潜在的陷阱。深度显微镜模型是否可能只是从记忆中的先前数据点中填充？即使这并非其性能的全部解释，但至少部分深度模型的能力可能来自这种记忆数据的反刍。这很可能会导致虚假相关性的插补。因此，在对微观数据集进行科学分析时，停下来质疑你的结果是否是由模型人为因素或真实生物现象造成的将至关重要。我们将在接下来的章节中为您提供一些工具，以便您批判性地审查模型，从而更好地确定您的模型实际学到了什么。

在下一章中，我们将探讨深度学习在医学中的应用。我们将重复本章中涵盖的许多视觉深度学习技能。

¹ 欧阳伟等人。“深度学习大大加速了超分辨率定位显微镜。”*自然生物技术*36（2018 年 4 月）：460–468。[*https://doi.org/10.1038/nbt.4106*](https://doi.org/10.1038/nbt.4106)。

² 陶鑫等人。“用于深度图像去模糊的尺度循环网络。”[*https://arxiv.org/pdf/1802.01770.pdf*](https://arxiv.org/pdf/1802.01770.pdf)。2018 年。

³ 克里斯滕森，埃里克。“体外标记：预测未标记图像中的荧光标记。”[*https://github.com/google/in-silico-labeling*](https://github.com/google/in-silico-labeling)。

⁴ 罗森菲尔德，阿米尔，理查德·泽梅尔和约翰·K·索索斯。“房间里的大象。”[*https://arxiv.org/abs/1808.03305*](https://arxiv.org/abs/1808.03305)。2018 年。
