
```py
import nltk
import numpy as np
from utils import get_batches, compute_pca, get_dict
import re
from matplotlib import pyplot

with open("hamlet.txt", "r", encoding="utf-8") as f:
    data = f.read()                              #1

data = re.sub(r"[,!?;-]", ".", data)         #2
data = nltk.word_tokenize(data)
data = [ch.lower() for ch in data if ch.isalpha() or ch == "."]
print("Number of tokens:", len(data), "\n", data[500:515])

fdist = nltk.FreqDist(word for word in data)     #3
print("Size of vocabulary:", len(fdist))
print("Most Frequent Tokens:", fdist.most_common(20))

word2Ind, Ind2word = get_dict(data)     #4
V = len(word2Ind)
print("Size of vocabulary:", V)

print("Index of the word 'king':", word2Ind["king"])
print("Word which has index 2743:", Ind2word[2743])

def initialize_model(N, V, random_seed=1):     #5
    """
    Inputs:
        N: dimension of hidden vector
        V: dimension of vocabulary
        random_seed: seed for consistent results in tests
    Outputs:
        W1, W2, b1, b2: initialized weights and biases
    """
    np.random.seed(random_seed)

    W1 = np.random.rand(N, V)
    W2 = np.random.rand(V, N)
    b1 = np.random.rand(N, 1)
    b2 = np.random.rand(V, 1)

    return W1, W2, b1, b2

def softmax(z):      #6
    """
    Inputs:
        z: output scores from the hidden layer
    Outputs:
        yhat: prediction (estimate of y)
    """
    yhat = np.exp(z) / np.sum(np.exp(z), axis=0)
    return yhat

def forward_prop(x, W1, W2, b1, b2):      #7
    """
    Inputs:
        x: average one-hot vector for the context
        W1,W2,b1,b2: weights and biases to be learned
    Outputs:
        z: output score vector
    """
    h = W1 @ x + b1
    h = np.maximum(0, h)
    z = W2 @ h + b2
    return z, h

def compute_cost(y, yhat, batch_size):     #8
    logprobs = np.multiply(np.log(yhat), y) + np.multiply(
        np.log(1 - yhat), 1 - y
    )
    cost = -1 / batch_size * np.sum(logprobs)
    cost = np.squeeze(cost)
    return cost

def back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size):    #9
    """
    Inputs:
        x:  average one hot vector for the context
        yhat: prediction (estimate of y)
        y:  target vector
        h:  hidden vector (see eq. 1)
        W1, W2, b1, b2:  weights and biases
        batch_size: batch size
    Outputs:
        grad_W1, grad_W2, grad_b1, grad_b2:  gradients of weights and biases
    """
    l1 = np.dot(W2.T, yhat - y)
    l1 = np.maximum(0, l1)
    grad_W1 = np.dot(l1, x.T) / batch_size
    grad_W2 = np.dot(yhat - y, h.T) / batch_size
    grad_b1 = np.sum(l1, axis=1, keepdims=True) / batch_size
    grad_b2 = np.sum(yhat - y, axis=1, keepdims=True) / batch_size

    return grad_W1, grad_W2, grad_b1, grad_b2

def gradient_descent(data, word2Ind, N, V, num_iters, alpha=0.03):     #10
    """
    This is the gradient_descent function
        Inputs:
            data:      text
            word2Ind:  words to Indices
            N:         dimension of hidden vector
            V:         dimension of vocabulary
            num_iters: number of iterations
        Outputs:
            W1, W2, b1, b2:  updated matrices and biases

        """
        W1, W2, b1, b2 = initialize_model(N, V, random_seed=8855)
        batch_size = 128
        iters = 0
        C = 2
        for x, y in get_batches(data, word2Ind, V, C, batch_size):
            z, h = forward_prop(x, W1, W2, b1, b2)
            yhat = softmax(z)
            cost = compute_cost(y, yhat, batch_size)
            if (iters + 1) % 10 == 0:
                print(f"iters: {iters+1} cost: {cost:.6f}")
            grad_W1, grad_W2, grad_b1, grad_b2 = back_prop(
                x, yhat, y, h, W1, W2, b1, b2, batch_size
            )
            W1 = W1 - alpha * grad_W1
            W2 = W2 - alpha * grad_W2
            b1 = b1 - alpha * grad_b1
            b2 = b2 - alpha * grad_b2
            iters += 1
            if iters == num_iters:
                break
            if iters % 100 == 0:
                alpha *= 0.66

        return W1, W2, b1, b2

C = 2      #11
N = 50
word2Ind, Ind2word = get_dict(data)
V = len(word2Ind)
num_iters = 150
print("Call gradient_descent")
W1, W2, b1, b2 = gradient_descent(data, word2Ind, N, V, num_iters)
# Call gradient descent
# Iters: 10 loss: 0.525015
# Iters: 20 loss: 0.092373
# Iters: 30 loss: 0.050474
# Iters: 40 loss: 0.034724
# Iters: 50 loss: 0.026468
# Iters: 60 loss: 0.021385
# Iters: 70 loss: 0.017941
# Iters: 80 loss: 0.015453
# Iters: 90 loss: 0.012099
# Iters: 100 loss: 0.012099
# Iters: 110 loss: 0.011253
# Iters: 120 loss: 0.010551
# Iters: 130 loss: 0.009932
# Iters: 140 loss: 0.009382
# Iters: 150 loss: 0.008889
```

#1 创建我们的训练语料库

#2 通过删除标点符号、按单词标记化并转换为小写字母字符来稍微清理数据

#3 获取我们的词袋，以及一个分布

#4 创建两个字典以加快转换时间和跟踪词汇

#5 在这里，我们使用一个层和两个参数创建我们的神经网络。

#6 创建我们的最终分类层，使得所有可能性加起来等于 1

#7 定义了通过我们的模型向前移动的行为，以及一个激活函数

#8 定义了如何确定真实值和模型预测之间的距离

#9 定义了如何通过模型向后移动并收集梯度

#10 将所有内容组合在一起并进行训练

#11 训练模型

CBoW 示例是我们第一个展示机器学习中完整且有效训练循环的代码示例。在这个过程中，请特别注意训练循环中的步骤，特别是激活函数 ReLU。由于我们预期你对各种 ML 范式至少是熟悉的，包括不同的激活函数，我们在这里不会解释 ReLU。我们将讨论何时应该使用它以及何时不应该使用它。ReLU 虽然解决了梯度消失问题，但并没有解决梯度爆炸问题，并且会破坏模型中所有的负比较。更好的情境变体包括指数线性单元（ELU），它允许负数归一化到 alpha，以及广义高斯线性单元（GEGLU）/Swish 门控线性单元（SWIGLU），它们在越来越复杂的场景中表现良好，如语言。然而，人们经常使用 ReLU，并不是因为它们在某个情境下是最好的，而是因为它们易于理解、编码和直观，甚至比它们所取代的激活函数（如 sigmoid 或 tanh）更直观。

许多内容最终都会通过包和其他类似工具进行抽象化，但了解底层发生的事情对于将 LLMs 投入生产的你来说将非常有帮助。你应该能够以某种确定性预测不同模型在不同情况下的行为。下一节将深入探讨这些抽象化之一——在这种情况下，是连续建模技术创建的抽象化。

### 2.2.5 嵌入

回顾到我们关于语言的特征，应该很容易理解为什么连续风格的文本建模是一个突破。嵌入将我们创建的、不包含任何意义的标记化向量，尝试根据对文本的观察插入意义，例如词序和出现在相似上下文中的子词。尽管主要的意义模式是搭配（共定位，相邻出现的词），但它们证明是有用的，甚至显示出与人类编码的词义的一些相似性。

Word2Vec 中的一个经典示例，也是最早的预训练向量嵌入之一，就是取“king”（国王）的向量，减去“man”（男人）的向量，加上“woman”（女人）的向量，然后找到与这个和最近的邻居向量，就是“queen”（王后）这个词的向量。这在我们看来是有道理的，因为它模仿了人类的语义。其中一个已经被提到几次的主要区别是：语用学。人类使用语用上下文来告知语义意义，理解到你说“我需要食物”并不意味着你实际上没有食物就会处于身体危险之中。嵌入没有受到任何纯使用之外的影响，这感觉就像人类学习的方式一样，而且在这里有很好的论据。唯一确定的是，如果我们能以某种方式给模型提供更具代表性的数据，这可能为更有效的嵌入打开大门，但这是一个鸡生蛋、蛋生鸡的问题，因为更有效的嵌入会带来更好的模型性能。

在列出 2.5 时，我们深入探讨了如何使用`pyplot`可视化嵌入。我们将在后面的章节中更深入地探讨嵌入。这对于模型可解释性和在预训练步骤中的验证都很有帮助。如果你看到你的语义相似的嵌入在图上相对较近，你很可能是在正确的方向上。

##### 列出 2.5：嵌入可视化

```py
words = [          #1
    "King",
    "Queen",
    "Lord",
    "Man",
    "Woman",
    "Prince",
    "Ophelia",
    "Rich",
    "Happy",
]              
embs = (W1.T + W2) / 2.0
idx = [word2Ind[word] for word in words]
X = embs[idx, :]
print(X.shape, idx)

result = compute_pca(X, 2)
pyplot.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()
```

#1 列出 2.4 完成后，并执行了梯度下降

如图 2.4 所示，这是我们从 CBoW 模型中训练出的一个成功但非常稀疏的嵌入表示。使这些语义表示（嵌入）更密集是我们在这个领域可以看到的主要改进之处，尽管已经进行了许多实验，其中更密集的语义意义被通过指导和不同的思维链技术用更大的语用上下文所取代。我们将在后面讨论思维链（CoT）和其他技术。现在，让我们转向讨论为什么我们的连续嵌入技术甚至可以成功，鉴于基于频率的模型通常很难与现实相关联。所有这一切都始于半个多世纪前的 MLP。

![figure](img/2-4.png)

##### 图 2.4：词嵌入的可视化技术。可视化嵌入对于模型可解释性很重要。

### 2.2.6  多层感知器

MLPs 是“机器擅长做一件事，所以希望我们能够只用一些擅长这件事的机器来做出擅长很多事情的机器”这一观点的体现。MLP 神经网络中的每一个权重和偏差都擅长做一件事，这可能是检测一个或多个特征。因此，我们将它们绑定在一起以检测更大、更复杂的特征。MLPs 是大多数神经网络架构中的主要构建块。架构之间的关键区别，如卷积神经网络和循环神经网络，主要源于数据加载方法以及处理在模型层中流动的标记化和嵌入数据，而不是单个层的功能，尤其是全连接层。

列表 2.6 提供了一个更动态的神经网络类，它可以拥有你任务所需的所有层和参数。我们使用 PyTorch 提供了一个更定义明确和显式的类，以便你能够以任何你想要的方式实现 MLP，无论是从头开始还是在一个流行的框架中。

##### 列表 2.6 多层感知器 PyTorch 类实现
