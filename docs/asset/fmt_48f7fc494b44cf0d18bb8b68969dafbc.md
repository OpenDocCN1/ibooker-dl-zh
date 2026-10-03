
`tensorflow-datasets`是使用最简单的，但是它有限，因为它只支持有限的数据集。

## 练习答案

**练习 1:** 功能 API。由于有两个输出层，我们不能使用串行 API。没有必要使用子类化 API，因为我们需要的一切都可以使用 Keras 层完成。

**练习 2:** `labels_ds.map(lambda x: x if x != -1)`。你也可以使用 `tf.Dataset.filter()` 方法（即 `labels_ds.filter(lambda x: x != -1)`）。

**练习 3:** `tfds.load("caltech101", with_info=True)`
