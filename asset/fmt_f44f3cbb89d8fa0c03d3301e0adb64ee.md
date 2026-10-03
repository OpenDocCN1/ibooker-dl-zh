
# PyTorch 与 TabNet

`PyTorch with TabNet` 通过在 Colab 中无需任何麻烦地工作，在表格数据特定的库中脱颖而出。像 `fastai` 一样，使用 `TabNet`，你只需几行代码就可以在表格数据上定义和训练一个深度学习模型。与 `fastai` 不同，`TabNet` 使用的是 `Scikit-learn` 用户容易理解的常规 API。与其他堆栈相比，`TabNet` 训练模型所需的时间更长。此外，作为一个专门为表格数据设计的库，`TabNet` 的用户社区比 `fastai` 和 `Keras` 小，这意味着如果你遇到问题，其他人已经遇到过并已在 Stack Overflow 上记录了修复的可能性较小。

# PyTorch 与 Lightning Flash

`PyTorch with Lightning Flash` 可以快速训练模型，并且一旦运行起来，API 就很简单。`Lightning` 拥有一个庞大的社区——不如 `Keras` 那么大，但比 `fastai` 大。然而，在 Colab 上使用 `Lightning Flash` 处理表格数据的问题领域并不大，我们刚刚才让 `Lightning Flash` 在 Colab 上运行起来。

在本章中，我们探索了一系列用于表格数据的深度学习堆栈，比较了每种方法的优缺点，并讨论了为什么我们没有探索一些其他堆栈。在下一章中，我们将回顾用于表格数据的深度学习的最佳实践。

## 摘要

+   有两个低级深度学习框架：`TensorFlow` 和 `PyTorch`。
+   `TensorFlow` 在工业中应用得更频繁。
+   `PyTorch` 是研究中的主要深度学习框架。
+   `Keras` 是 `TensorFlow` 的高级 API。
+   `fastai` 既是 `PyTorch` 的一般用途、高级 API，也是一个表格数据库。
+   `PyTorch Lightning` 是一个高级 API，它抽象了一些 `PyTorch` 的细节。`Lightning Flash` 是基于 `Lightning` 的表格数据库。
+   `Lightning Flash` 和 `fastai` 都为 `PyTorch` 提供了一些与 `Keras` 为 `TensorFlow` 提供的相同的好处，通过抽象底层 `PyTorch` 框架的一些方面。
+   `TabNet` 是一个既适用于 `TensorFlow` 也适用于 `PyTorch` 的表格数据库。
+   `SAINT` 是 `TensorFlow` 的表格数据库。
+   `PyTorch Tabular` 是 `PyTorch` 的表格数据库。
+   在所有可用的选择中，`TensorFlow` 与 `Keras`、`PyTorch` 与 `fastai`、`PyTorch` 与 `TabNet` 以及 `PyTorch` 与 `Lightning Flash` 都是 Colab 上进行表格数据深度学习的有效选项。
