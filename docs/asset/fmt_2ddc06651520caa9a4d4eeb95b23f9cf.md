
# 初始实例化

初始实例化中的最后两项是损失函数和优化器。本例中使用的损失函数是`bcewithlogits`损失。要更详细地解释为什么使用这种损失，请参考“感知器分类器”，它描述了模型。简而言之，最合适的损失函数的二元分类是二进制交叉熵（BCE）损失和更数值稳定对`BCEWithLogitsLoss`的模型不适用 Sigmoid 函数输出比一对`BCELoss`模型，并应用 sigmoid 函数的输出。我们使用的优化器是 Adam 优化器。一般来说，Adam 与其他优化器相比具有很强的竞争力，在撰写本文时，还没有令人信服的证据表明可以使用任何其他优化器来替代 Adam。我们鼓励您通过尝试其他优化器并注意性能来验证这一点。

**示例 3-19：实例化数据集，模型，损失，优化器和训练状态**

```py
import torch.optim as optim

def make_train_state(args):
    return {'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1}
train_state = make_train_state(args)

if not torch.cuda.is_available():
    args.cuda = False
args.device = torch.device("cuda" if args.cuda else "cpu")

# dataset and vectorizer
dataset = ReviewDataset.load_dataset_and_make_vectorizer(args.review_csv)
vectorizer = dataset.get_vectorizer()

# model
classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab))
classifier = classifier.to(args.device)

# loss and optimizer
loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
```

# 训练循环

训练循环使用来自初始实例化的对象来更新模型参数，以便随着时间的推移进行改进。更具体地说，训练循环由两个循环组成：一个内循环覆盖数据集中的小批量，另一个外循环重复内循环若干次。在内部循环中，计算每个小批量的损失，并使用优化器更新模型参数。

为了更全面地介绍所发生的事情，让我们从下面的代码片段顶部开始。在第一行中，我们使用`for`循环，它的范围跨越各个周期。周期的数量是一个可以设置的超参数。它控制训练例程应该对数据集进行多少次传递。在实践中，您应该使用类似于早期停止标准的东西来在循环结束之前终止它。在补充资料中，我们向您展示了如何做到这一点。

在`for`循环的顶部，有几个例程定义和实例化。首先设置训练状态的周期索引，然后设置数据集的分割（首先是`"train"`，然后是`"val"`，当我们想在周期结束时测量模型性能，最后是`"test"`，当我们想评估模型的最终性能）。考虑到我们是如何构造数据集的，应该总是在调用`generate_batch()`之前设置拆分。创建`batch_generator`之后，将实例化两个浮动，以跟踪批量之间的损失和准确性。有关这里使用的“运行平均公式”的更多细节，请参阅 Wikipedia 的`moving average`页面。最后，调用分类器的`.train()`方法，表示模型处于“训练模式”，模型参数是可变的。这也支持像丢弃这样的正则化机制。

训练循环的下一部分是迭代`batch_generator`中的训练批，并执行更新模型参数的基本操作。在每个批量迭代中，首先使用`optimizer.zero_grad()`方法重置优化器的梯度。然后，从模型中计算输出。接下来，损失函数用于计算模型输出与监督目标（真正的类标签）之间的损失。在此之后，对损失对象（而不是损失函数对象）调用`loss.backward()`方法，导致梯度传播到每个参数。最后，优化器使用这些传播的梯度来使用`optimizer.step()`方法执行参数更新。这五个步骤是梯度下降的基本步骤。除此之外，还有一些用于记帐和跟踪的额外操作。具体来说，损失和精度值（作为常规 Python 变量存储）被计算出来，然后用于更新运行损失和运行精度变量。

在训练分割批量的内部循环之后，有两个簿记和实例化操作。具体来说，首先用最终的损失和精度值更新训练状态。然后，创建一个新的批量生成器、运行损失和运行精度。验证数据的循环几乎与训练数据相同，因此重用相同的变量。有一个主要的区别：调用分类器的`.eval()`方法，它执行与分类器的`.train()`方法相反的操作。`eval()`方法使模型参数不可变，且不可丢失。求值模式还禁止计算梯度的损失并将其传播回参数。这很重要，因为我们不希望模型根据验证数据调整参数。相反，我们希望这些数据作为模型执行情况的度量。如果其测量性能之间存在着很大的差异在训练数据和验证数据的测量性能，很可能模型过度拟合训练数据，你应该调整模型或训练程序（比如设置阻止早期，我们使用补充笔记本对于这个例子）。

在对验证数据进行迭代并保存由此产生的验证损失和精度值之后，外部`for`循环就完成了。我们在本书中实现的每个训练例程都将遵循非常相似的设计模式。事实上，所有梯度下降算法都遵循相似的设计模式。在您习惯了从头开始编写这个循环之后，您将会学会如何使用它执行梯度下降！示例 3-20 给出了代码。

**示例 3-20：粗糙的训练循环**

```py
for epoch_index in range(args.num_epochs):
    train_state['epoch_index'] = epoch_index

    # Iterate over training dataset

    # setup: batch generator, set loss and acc to 0, set train mode on
    dataset.set_split('train')
    batch_generator = generate_batches(dataset,
                                       batch_size=args.batch_size,
                                       device=args.device)
    running_loss = 0.0
    running_acc = 0.0
    classifier.train()

    for batch_index, batch_dict in enumerate(batch_generator):
        # the training routine is 5 steps:

        # step 1\. zero the gradients
        optimizer.zero_grad()

        # step 2\. compute the output
        y_pred = classifier(x_in=batch_dict['x_data'].float())

        # step 3\. compute the loss
        loss = loss_func(y_pred, batch_dict['y_target'].float())
        loss_batch = loss.item()
        running_loss += (loss_batch - running_loss) / (batch_index + 1)

        # step 4\. use loss to produce gradients
        loss.backward()

        # step 5\. use optimizer to take gradient step
        optimizer.step()

        # -----------------------------------------
        # compute the accuracy
        acc_batch = compute_accuracy(y_pred, batch_dict['y_target'])
        running_acc += (acc_batch - running_acc) / (batch_index + 1)

    train_state['train_loss'].append(running_loss)
    train_state['train_acc'].append(running_acc)

    # Iterate over val dataset

    # setup: batch generator, set loss and acc to 0; set eval mode on
    dataset.set_split('val')
    batch_generator = generate_batches(dataset,
                                       batch_size=args.batch_size,
                                       device=args.device)
    running_loss = 0.
    running_acc = 0.
    classifier.eval()

    for batch_index, batch_dict in enumerate(batch_generator):

        # step 1\. compute the output
        y_pred = classifier(x_in=batch_dict['x_data'].float())

        # step 2\. compute the loss
        loss = loss_func(y_pred, batch_dict['y_target'].float())
        loss_batch = loss.item()
        running_loss += (loss_batch - running_loss) / (batch_index + 1)

        # step 3\. compute the accuracy
        acc_batch = compute_accuracy(y_pred, batch_dict['y_target'])
        running_acc += (acc_batch - running_acc) / (batch_index + 1)

    train_state['val_loss'].append(running_loss)
    train_state['val_acc'].append(running_acc)
```

## 评估，推断和检查

在您有了一个经过训练的模型之后，接下来的步骤是要么评估它是如何处理一些保留下来的数据的，要么使用它对新数据进行推断，要么检查模型的权重，看看它学到了什么。在本节中，我们将向您展示所有三个步骤。

### 在测试数据上评估

为了评估外置测试集上的数据，代码与我们在上一个示例中看到的训练例程中的验证循环完全相同，但有一个细微的区别：分割设置为`"test"`而不是`"val"`。数据集的两个分区之间的区别在于，测试集应该尽可能少地运行。每次您在测试集上运行一个训练过的模型，做出一个新的模型决策（例如改变层的大小），并在测试集上重新测量新的再训练模型时，您都是在向测试数据倾斜您的建模决策。换句话说，如果您足够频繁地重复这个过程，那么测试集作为真正交付数据的精确度量将变得毫无意义。示例 3-21 对此进行了更深入的研究。

**示例 3-21：测试集评估**

```py
Input[0]
dataset.set_split('test')
batch_generator = generate_batches(dataset,
                                   batch_size=args.batch_size,
                                   device=args.device)
running_loss = 0.
running_acc = 0.
classifier.eval()

for batch_index, batch_dict in enumerate(batch_generator):
    # compute the output
    y_pred = classifier(x_in=batch_dict['x_data'].float())

    # compute the loss
    loss = loss_func(y_pred, batch_dict['y_target'].float())
    loss_batch = loss.item()
    running_loss += (loss_batch - running_loss) / (batch_index + 1)

    # compute the accuracy
    acc_batch = compute_accuracy(y_pred, batch_dict['y_target'])
    running_acc += (acc_batch - running_acc) / (batch_index + 1)

train_state['test_loss'] = running_loss
train_state['test_acc'] = running_acc
Input[1]
print("Test loss: {:.3f}".format(train_state['test_loss']))
print("Test Accuracy: {:.2f}".format(train_state['test_acc']))
Output[1]
Test loss: 0.297
Test Accuracy: 90.55
```

### 推断和分类新的数据点

评价模型的另一种方法是对新数据进行推断，并对模型是否有效进行定性判断。我们可以在示例 3-22 中看到这一点。

**示例 3-22：打印样本评论的预测**

```py
Input[0]
def predict_rating(review, classifier, vectorizer,
                   decision_threshold=0.5):
    """Predict the rating of a review

    Args:
        review (str): the text of the review
        classifier (ReviewClassifier): the trained model
        vectorizer (ReviewVectorizer): the corresponding vectorizer
        decision_threshold (float): The numerical boundary which
            separates the rating classes
    """

    review = preprocess_text(review)
    vectorized_review = torch.tensor(vectorizer.vectorize(review))
    result = classifier(vectorized_review.view(1, -1))

    probability_value = F.sigmoid(result).item()

    index =  1
    if probability_value < decision_threshold:
        index = 0

    return vectorizer.rating_vocab.lookup_index(index)

test_review = "this is a pretty awesome book"
prediction = predict_rating(test_review, classifier, vectorizer)
print("{} -> {}".format(test_review, prediction)
Output[0]
this is a pretty awesome book -> positive
```

### 检查模型权重

最后，了解模型在完成训练后是否表现良好的最后一种方法是检查权重，并对权重是否正确做出定性判断。如示例 3-23 所示，使用感知器和压缩的单热编码，这是一种相当简单的方法，因为每个模型的权重与词汇表中的单词完全对应。

**示例 3-23：打印对应分类器权重的单词**
