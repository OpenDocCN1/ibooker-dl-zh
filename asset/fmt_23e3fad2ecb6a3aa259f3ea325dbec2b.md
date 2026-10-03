
```py
# Takes a NumPy array or a tensor for your current backend and returns
# a tensor for your current backend
predictions = model(new_inputs) 
```

然而，这会一次性处理`new_inputs`中的所有输入，如果你正在查看大量数据（特别是，可能需要比你的 GPU 更多的内存），这可能不可行。

进行推理的更好方法是使用`predict()`方法。它将遍历数据的小批量，并返回一个包含预测的 NumPy 数组。而且与`__call__`不同，它还可以处理 TensorFlow `Dataset`对象：

```py
# Takes a NumPy array or a Dataset and returns a NumPy array
predictions = model.predict(new_inputs, batch_size=128) 
```

例如，如果我们使用`predict()`对之前训练的线性模型的一些验证数据进行预测，我们得到的是与模型对每个输入样本预测相对应的标量分数：

```py
>>> predictions = model.predict(val_inputs, batch_size=128)
>>> print(predictions[:10])
[[0.3590725 ]
 [0.82706255]
 [0.74428225]
 [0.682058  ]
 [0.7312616 ]
 [0.6059811 ]
 [0.78046083]
 [0.025846  ]
 [0.16594526]
 [0.72068727]]
```

目前，你只需要了解这么多关于 Keras 模型的知识。到目前为止，你已经准备好进入下一章，使用 Keras 解决现实世界的机器学习问题。

## 摘要

+   TensorFlow、PyTorch 和 JAX 是三种流行的用于数值计算和自动微分的基础层框架。它们都有自己的做事方式、优势和劣势。

+   Keras 是构建和训练神经网络的顶层 API。它可以与 TensorFlow、PyTorch 或 JAX 一起使用——只需选择你最喜欢的后端即可。

+   Keras 的核心类是`Layer`。一个层封装了一些权重和一些计算。层被组装成模型。

+   在开始训练模型之前，你需要选择一个优化器、一个损失函数和一些指标，你可以通过`model.compile()`方法指定它们。

+   要训练一个模型，你可以使用`fit()`方法，它会为你运行小批量梯度下降。你还可以用它来监控验证数据上的损失和指标，这是一组模型在训练期间没有见过的输入。

+   模型训练完成后，你可以使用`model.predict()`方法对新输入生成预测。

### 脚注

1.  R. E. Wengert，“一个简单的自动导数评估程序”，ACM 通讯，第 7 卷第 8 期（1964 年）。[[↩]](#footnote-link-1)

1.  注意，PyTorch 是一个中间案例：虽然它主要是一个底层框架，但它也包括自己的层和自己的优化器。然而，如果你将 PyTorch 与 Keras 结合使用，那么你将只与低级的 PyTorch API 交互，例如张量操作。[[↩]](#footnote-link-2)
