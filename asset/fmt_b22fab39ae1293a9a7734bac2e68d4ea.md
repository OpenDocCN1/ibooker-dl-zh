
随着人工智能自动化越来越多的工作，不足为奇它最终也可以自动设计人工智能架构。`NAS`方法利用强化学习将小型架构块连接在一起，直到它们能够最大化目标函数；换句话说，我们的验证准确性。当前最先进的网络都基于`NAS`，使人类设计的架构相形见绌。这一领域的研究始于 2017 年，2018 年更加注重使训练更快。现在有了`AutoKeras`（Haifeng Jin 等人），我们也可以以相对容易的方式在我们的特定数据集上应用这种最先进的技术。

使用`AutoKeras`生成新的模型架构只需提供我们的图像和相关标签，以及一个完成作业的时间限制。在内部，它实现了几种优化算法，包括贝叶斯优化方法来搜索最佳架构：

```py
!pip3 install autokeras
!pip3 install graphviz
from keras.datasets import mnist
from autokeras.image.image_supervised import ImageClassifier

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))

clf = ImageClassifier(path=".",verbose=True, augment=False)
clf.fit(x_train, y_train, time_limit= 30 * 60) # 30 minutes
clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
y = clf.evaluate(x_test, y_test)
print(y)

# Save the model as a pickle file
clf.export_autokeras_model("model.pkl")

visualize('.')
```

训练后，我们都渴望了解新模型架构的样子。与我们通常看到的大多数干净的图像不同，这个看起来相当难以理解或打印出来。但我们相信的是，它能产生高准确性。

## 总结

在本章中，我们看到了一系列工具和技术，帮助探索改进`CNN`准确性的机会。通过建立迭代实验的案例，您了解到调整超参数如何带来最佳性能。由于有这么多超参数可供选择，我们随后看了自动化方法，包括`AutoKeras`、`AutoAugment`和`Keras Tuner`。最重要的是，本章的核心代码结合了多个工具，存储在书的`GitHub`上（请参阅[*http://PracticalDeepLearning.ai*](http://PracticalDeepLearning.ai)），可以通过一行更改轻松调整到 100 多个数据集，并在浏览器中在线运行。此外，我们编制了一份可操作提示清单，以及在线托管的交互式实验，帮助您的模型获得一点额外优势。我们希望本章涵盖的内容能够使您的模型更加健壮，减少偏见，使其更易解释，并最终有助于负责任地发展人工智能。
