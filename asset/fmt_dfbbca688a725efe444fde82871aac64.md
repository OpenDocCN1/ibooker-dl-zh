
```py
# Instantiate model, optimizer, and hyperparameter(s)
in_dim, feature_dim, out_dim = 784, 256, 10
lr=1e-3
loss_fn = nn.CrossEntropyLoss()
epochs=40
classifier = BaseClassifier(in_dim, feature_dim, out_dim)
optimizer = optim.SGD(classifier.parameters(), lr=lr)

def train(classifier=classifier,
          optimizer=optimizer,
          epochs=epochs,
          loss_fn=loss_fn):

  classifier.train()
  loss_lt = []
  for epoch in range(epochs):
    running_loss = 0.0
    for minibatch in train_loader:
      data, target = minibatch
      data = data.flatten(start_dim=1)
      out = classifier(data)
      computed_loss = loss_fn(out, target)
      computed_loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      # Keep track of sum of loss of each minibatch
      running_loss += computed_loss.item()
    loss_lt.append(running_loss/len(train_loader))
    print("Epoch: {} train loss: {}".format(epoch+1, 
          running_loss/len(train_loader)))

  plt.plot([i for i in range(1,epochs+1)], loss_lt)
  plt.xlabel("Epoch")
  plt.ylabel("Training Loss")
  plt.title(
      "MNIST Training Loss: optimizer {}, lr {}".format("SGD", lr))
  plt.show()

  # Save state to file as checkpoint
  torch.save(classifier.state_dict(), 'mnist.pt')

def test(classifier=classifier,
          loss_fn = loss_fn):
  classifier.eval()
  accuracy = 0.0
  computed_loss = 0.0

  with torch.no_grad():
      for data, target in test_loader:
          data = data.flatten(start_dim=1)
          out = classifier(data)
          _, preds = out.max(dim=1)

          # Get loss and accuracy
          computed_loss += loss_fn(out, target)
          accuracy += torch.sum(preds==target)

      print("Test loss: {}, test accuracy: {}".format(
          computed_loss.item()/(len(test_loader)*64), 
          accuracy*100.0/(len(test_loader)*64)))
```

此外，请注意，在训练和测试函数的开始分别调用`classifier.train()`和`classifier.eval()`。对这些函数的调用向 PyTorch 后端传达了模型是处于训练模式还是推理模式。您可能会想知道为什么我们需要调用`classifier.train()`和`classifier.eval()`，如果在训练和测试时间神经网络的行为没有区别。尽管在我们的第一次尝试中是这样，但其他神经架构的训练和测试模式并不一定相同。例如，如果在模型架构中添加了 dropout 层，则在测试阶段需要忽略 dropout 层。我们在这里添加`train()`和`eval()`的调用，因为通常认为这样做是一个好习惯。

作为第一步，我们需要为模型训练设置一些起始超参数。我们从稍微保守的学习率`1e-4`开始，并在 40 个 epochs 或整个数据集的迭代后检查训练损失曲线和测试准确率。图 5-2 显示了通过 epochs 的训练损失曲线的图表。

![](img/fdl2_0502.png)

###### 图 5-2。我们看到欠拟合的迹象，因为模型在训练集上的表现未能平稳下来，这意味着我们尚未进入局部最优解

我们可以看到这个损失曲线在训练结束时并没有接近平稳，我们希望看到的是一个以足够学习率训练的模型开始出现的情况。虽然我们没有验证集来确认我们的怀疑，但我们有充分的理由怀疑更高的学习率会有所帮助。将学习率设置为稍微更积极的`1e-3`后，我们观察到训练损失曲线更符合我们所期望看到的情况（图 5-3）。

![](img/fdl2_0503.png)

###### 图 5-3。损失曲线的平稳是我们对于问题的合适学习率所期望看到的更多的情况

损失曲线仅在训练结束时开始平稳下来。这种趋势表明模型很可能处于欠拟合训练数据和过拟合训练数据之间的最佳状态，就像我们之前的尝试一样。在测试集上对 40 个 epochs 训练的模型进行评估，准确率达到 91％！虽然这与今天 MNIST 的顶尖表现者相去甚远，后者主要使用卷积神经分类器，但这是一个很好的开始。我们建议您尝试对代码进行一些扩展，例如增加隐藏层的数量并替换更复杂的优化器。

# 总结

在这一章中，我们涵盖了 PyTorch 及其功能的基础知识。具体来说，我们学习了 PyTorch 中张量的概念，以及这些张量如何存储数值信息。此外，我们还学习了如何通过张量操作来操作张量，访问张量中的数据，并设置一些重要的属性。我们还讨论了 PyTorch 中的梯度以及它们如何存储在张量中。我们通过 PyTorch `nn`模块部分的标准`nn`功能构建了我们的第一个神经网络。将基于`nn`的方法与仅使用 PyTorch 张量的方法进行比较，显示了`nn`模块提供的有效抽象，使其易于使用。最后，在最后一节中，我们将所有学到的知识结合起来，在 PyTorch 提供的测试集上训练和测试了一个 MNIST 手写数字前馈神经分类器，准确率达到了 91%。虽然我们涵盖了许多基础知识，并为您提供了您需要动手实践的知识，但我们只是触及了 PyTorch API 所提供的所有内容的表面—我们鼓励您进一步探索并改进我们在本节中构建的模型。我们建议您访问 PyTorch 文档以了解更多信息，并构建自己的神经网络，包括尝试其他架构，应用于各种在线数据集，如 CIFAR-10 图像识别数据集。在下一节中，我们将涵盖神经网络实现，这是当今另一个最流行的深度学习框架之一。
