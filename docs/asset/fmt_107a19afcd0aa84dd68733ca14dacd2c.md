
> **重要提示**：使用命名列表时，列表的顺序不能保证保留下来。请务必通过位置或名称跟踪项目，但不能混合使用两者。

## 示例 7.11 通过提供命名输入和目标数组列表来训练模型

```r
model %>%
  compile(optimizer = "rmsprop",
          loss = c(priority = "mean_squared_error",
                   department = "categorical_crossentropy"),
          metrics = c(priority = "mean_absolute_error",
                      department = "accuracy"))

model %>%
  fit(list(title = title_data,
           text_body = text_body_data,
           标签 = tags_data),
      list(priority = priority_data,
           department = department_data), epochs = 1)

model %>%
  评估(list(title = title_data,
           text_body = text_body_data,
           标签 = tags_data),
      list(priority = priority_data,
           department = department_data))
```

![图片](img/f0194-02.jpg)

```r
c(priority_preds, department_preds) %<-%
  预测(model, list(title = title_data,
                  text_body = text_body_data,
                  标签 = tags_data))
```

### 函数 API 的威力：访问层连接性

功能模型是一个明确的图形数据结构。这使得可以检查层如何连接，并重复使用以前的图形节点（它们是层输出）作为新模型的一部分。它还很好地适应了大多数研究人员在思考深度神经网络时使用的“心理模型”：层的图形。这使两个重要用例成为可能：模型可视化和特征提取。

让我们可视化刚刚定义的模型的连接性（模型的*拓扑*）。您可以使用`plot()`方法将功能模型绘制为图形（参见图 7.2）：

```r
plot(model)
```

![图片](img/f0194-01.jpg)

**图 7.2 由我们的票务分类器模型`plot(model)`生成的图表**

您可以在模型的每一层中添加此图表的输入和输出形状，这在调试期间可能会有所帮助（参见图 7.3）：

```r
plot(model, show_shapes = TRUE)
```

![图片](img/f0195-01.jpg)

**图 7.3 添加形状信息的模型图**

张量形状中的“None”表示批处理大小：该模型允许任意大小的批处理。

访问层连接性也意味着您可以检查和重复使用图中的单个节点（层调用）。`model$layers`模型属性提供了组成模型的层的列表，对于每个层，您可以查询`layer$input`和`layer$output`。

**列表 7.12 在功能模型中检索层的输入或输出**

```r
str(model$layers)
```

```
7 的列表
 $ :<keras.engine.input_layer.InputLayer object at 0x7fc962da63a0>
 $ :<keras.engine.input_layer.InputLayer object at 0x7fc962da6430>
 $ :<keras.engine.input_layer.InputLayer object at 0x7fc962da68e0>
 $ :<keras.layers.merge.Concatenate object at 0x7fc962d2e130>
 $ :<keras.layers.core.dense.Dense object at 0x7fc962da6c40>
 $ :<keras.layers.core.dense.Dense object at 0x7fc962da6340>
 $ :<keras.layers.core.dense.Dense object at 0x7fc962d331f0>
```

```r
str(model$layers[[4]]$input)
```

```
3 的列表
 $ :<KerasTensor: shape=(None, 10000) dtype=float32 (created by layer
 ![图片](img/common01.jpg) 'title')>
 $ :<KerasTensor: shape=(None, 10000) dtype=float32 (created by layer
 ![图片](img/common01.jpg) 'text_body')>
 $ :<KerasTensor: shape=(None, 100) dtype=float32 (created by layer 'tags')>
```

```r
str(model$layers[[4]]$output)
```

```
<KerasTensor: shape=(None, 20100) dtype=float32 (created by layer
![图片](img/common01.jpg) 'concatenate')>
```

这使您能够进行*特征提取*，创建重用另一个模型的中间特征的模型。

假设你想要在前一个模型中添加另一个输出——你想要估计给定问题票的解决时间，一种困难评级。你可以通过三个类别的分类层来实现这一点：“快速”，“中等”和“困难”。你不需要从头开始重新创建和重新训练一个模型。你可以从你以前模型的中间特征开始，因为你可以像这样访问它们。

**列表 7.13 通过重用中间层输出创建一个新模型**

```r
features <- model$layers[[5]]$output➊
difficulty <- features %>%
  layer_dense(3, activation = "softmax", name = "difficulty")

new_model <- keras_model(
  inputs = list(title, text_body, tags),
  outputs = list(priority, department, difficulty)
)
```

➊ **`layer[[5]]`是我们的中间密集层。您也可以使用`get_layer()`按名称检索层。**

让我们绘制我们的新模型（见图 7.4）：

```r
plot(new_model, show_shapes = TRUE)
```

![图像](img/f0196-01.jpg)

**图 7.4 我们的新模型绘图：更新的工单分类器**

### 7.2.3 子类化 Model 类

你应该知道的最后一个模型构建模式是最高级的模型子类化。你在第三章学习过如何使用`new_layer_class()`子类化`Layer`类并创建自定义层。使用`new_model_class()`子类化`Model`类非常相似：

+   在`initialize()`方法中，定义模型将使用的层。
+   在`call()`方法中，定义模型的前向传播，重用先前创建的层。
+   实例化你的子类，并在数据上调用它以创建它的权重

### 重写我们之前的示例作为子类模型

让我们看一个简单的例子：我们将使用`new_model_class()`重新实现客户支持票务管理模型来定义一个`Model`子类。

**列表 7.14 一个简单的子类模型**

```r
CustomerTicketModel <- new_model_class(
  classname = "CustomerTicketModel",
  initialize = function(num_departments) {
    super$initialize()➊
    self$concat_layer <- layer_concatenate()
    self$mixing_layer <-➋
      layer_dense(units = 64, activation = "relu")
    self$priority_scorer <-
      layer_dense(units = 1, activation = "sigmoid")
    self$department_classifier <-
      layer_dense(units = num_departments, activation = "softmax")
  },
  call = function(inputs) {➌
    title <- inputs$title➍
    text_body <- inputs$text_body
    tags <- inputs$tags
    features <- list(title, text_body, tags) %>%
      self$concat_layer() %>%
      self$mixing_layer()
    priority <- self$priority_scorer(features)
    department <- self$department_classifier(features)
    list(priority, department)
  }
)
```

➊ **别忘了调用`super$initialize()`！**

➋ **在构造函数中定义子层。请注意，我们在这里指定了`units`参数名称，以便我们得到一个层实例。**

➌ **在`call()`方法中定义前向传播。**

➍ **对于输入，我们将提供一个带有名称的列表给模型。**

我们在第三章中实现了`Model`的最简版本。要注意的主要事项是，我们正在定义一个自定义类，即我们的模型，它是`Model`子类。正如您将在接下来的章节中看到的那样，`Model`提供了许多方法和功能，您可以选择加入其中。

定义模型之后，您可以实例化它。请注意，它只会在首次在某些数据上调用它时创建其权重，就像层子类一样：

```r
model <- CustomerTicketModel(num_departments = 4)

c(priority, department) %<-% model(list(title = title_data,
                                        text_body = text_body_data,
                                        tags = tags_data))
```

模型本质上是一种层（`Layer`）。这意味着，使用`create_layer_wrapper()`就可以轻松地使模型具有与`%>%`很好地组合的能力，就像这样：

```r
inputs <- list(title = title_data,
               text_body = text_body_data,
               tags = tags_data)

layer_customer_ticket_model <- create_layer_wrapper(CustomerTicketModel)

outputs <- inputs %>%
  layer_customer_ticket_model(num_departments = 4)

c(priority, department) %<-% outputs
```

到目前为止，所有内容看起来与`Layer`子类化非常相似，这是您在第三章中遇到的工作流程。那么，“层”的子类和“模型”的子类之间有什么区别呢？答案很简单：一个“层”是用来创建模型的构建块，而一个“模型”是您将实际培训、导出推理等的最高级对象。换句话说，模型具有`fit()`、`evaluate()`和`predict()`方法，而层则没有。除此之外，这两个类几乎完全相同。（另一个区别是您可以将模型保存在磁盘上的文件中，我们将在几个章节中介绍这个功能。）您可以编译和训练`Model`子类，就像`Sequential`模型或`Functional`模型一样：

```r
model %>%
  compile(optimizer = "rmsprop",
          loss = c("mean_squared_error",
                   "categorical_crossentropy"),➊
          metrics = c("mean_absolute_error", "accuracy"))

x <- list(title = title_data,➋
          text_body = text_body_data,
          tags = tags_data)
y <- list(priority_data, department_data)➌
model %>% fit(x, y, epochs = 1)
model %>% evaluate(x, y)
```

![Image](img/f0198-01.jpg)

```r
c(priority_preds, department_preds) %<-% {
  model %>% predict(x)
}
```

➊ **您作为损失和指标参数传递的内容的结构必须严格匹配`call()`方法返回的内容，即两个元素的列表。**

➋ **输入数据的结构必须严格匹配`call()`方法所期望的内容，即一个命名列表，其中包括标题（`title`）、正文（`text_body`）和标签（`tags`）。当按名称匹配时，列表顺序将被忽略！**

➌ **目标数据的结构必须严格匹配`call()`方法返回的内容，即两个元素的列表。**

`Model`子类化流程是构建模型最灵活的方法。它使您能够构建无法表示为层的有向无环图的模型，例如，模型的`call()`方法在`for`循环内使用层，甚至会递归调用它们。任何事情都可能发生——您有权决定。

### 警告：子类化模型不支持的内容

这种自由是有代价的：通过子类模型，您需要负责更多的模型逻辑，这意味着您的潜在错误范围要大得多。因此，您将需要更多的调试工作。您正在开发一个新的类对象，而不仅仅是将乐高积木拼凑在一起。

`Functional`和子类模型在性质上也有很大不同。`Functional`模型是一个显式数据结构——层的图形，您可以查看、检查和修改。而子类模型是一组 R 代码——一个具有`call()`方法的类，该方法是一个 R 函数。这就是子类化工作流程的灵活性来源——您可以编写任何功能性代码——但它也引入了新的限制。

例如，因为层之间连接的方式被隐藏在`call()`方法的主体内部，您无法访问该信息。调用`summary()`将不显示层连接性，并且您无法通过`plot()`绘制模型拓扑。同样，如果您有一个子类模型，您无法访问层图的节点来进行特征提取，因为简单地没有图。一旦模型被实例化，其正向传递就变成了一个完全的黑盒。

### 7.2.4 混合和匹配不同的组件

关键是，选择这些模式之一——`Sequential`模型、`Functional` API 或`Model`子类化——不会将您排除在其他模式之外。Keras API 中的所有模型都可以与彼此平滑地互操作，无论它们是`Sequential`模型、`Functional`模型还是从头编写的子类模型。它们都是同一工作流谱系的一部分。例如，您可以在`Functional`模型中使用一个子类化的层或模型。

**列表 7.15 创建一个包含子类化模型的 Functional 模型**

```r
ClassifierModel <- new_model_class(
  classname = "Classifier",
  initialize = function(num_classes = 2) {
    super$initialize()
    if (num_classes == 2) {
      num_units <- 1
      activation <- "sigmoid"
    } else {
      num_units <- num_classes
      activation <- "softmax"
    }
    self$dense <- layer_dense(units = num_units, activation = activation)
  },
  call = function(inputs)
    self$dense(inputs)
)

inputs <- layer_input(shape = c(3))
classifier <- ClassifierModel(num_classes = 10)
outputs <- inputs %>%
  layer_dense(64, activation = "relu") %>%
  classifier()
model <- keras_model(inputs = inputs, outputs = outputs)
```

相反地，您可以在子类化层或模型中使用`Functional`模型的一部分。

**列表 7.16 创建一个包含 Functional 模型的子类模型**

```r
inputs <- layer_input(shape = c(64))
outputs <- inputs %>% layer_dense(1, activation = "sigmoid")
binary_classifier <- keras_model(inputs = inputs, outputs = outputs)

MyModel <- new_model_class(
  classname = "MyModel",
  initialize = function(num_classes = 2) {
    super$initialize()
    self$dense <- layer_dense(units = 64, activation = "relu")
    self$classifier <- binary_classifier
  },
  call = function(inputs) {
    inputs %>%
      self$dense() %>%
      self$classifier()
  }
)

model <- MyModel()
```

### 7.2.5 记住：用对的工具做对的事情

你已经学习了一系列构建 Keras 模型的工作流程，从最简单的序列模型到最高级的模型子类化。什么时候应该使用其中一种？每种都有其优缺点，选择最适合当前工作的一种。

总的来说，功能 API 为您提供了易于使用和灵活性的很好的平衡。它还直接给您访问层连接性的能力，对于模型绘制或特征提取等用例非常有用。如果您*可以*使用功能 API，即，如果您的模型可以表示为层的有向无环图，那么我建议您使用功能 API 而不是模型子类化。
