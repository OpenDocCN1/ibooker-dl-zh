
# 8.9 Rasa 框架的关键概念

*   **动作** — 聊天机器人系统可以执行的动作。一个简单的动作可能是一个预先准备好的文本响应（例如在问候时返回`hello`）。在我们的深度学习模型部署中，我们将动作定义为 Python 中的`actions.py`文件中的`ActionPredictDelayComplete`类。此动作接受 Rasa 从用户输入中提取的槽位值，填写未由槽位指定的值，将值通过管道运行，将管道的输出应用于训练好的模型，并最终根据训练模型的预测为用户编写响应。

*   **槽位** — 一组键和值，用于捕获用户的基本输入。在我们的深度学习模型部署中，为模型期望的所有输入列（路线、方向、小时、月份等）定义了槽位。

*   **故事** — 用户与聊天机器人之间对话的抽象，可以表示多次来回交流。在我们深度学习模型部署中的主要故事是一个简单的交流：用户询问一次旅行是否会延误，机器人提供响应以指示延误或无延误。

图 8.11 显示了用于训练 Rasa 模型的关键文件以及每个文件中定义的 Rasa 对象。当您训练 Rasa 模型（您将在第 8.10 节中这样做）时，Rasa 使用`nlu.md`和`stories.md`文件中的训练数据来训练模型。

![CH08_F11_Ryan](img/CH08_F11_Ryan.png)

图 8.11 定义在 Rasa 中的关键文件和对象

Rasa 框架中的另一个关键文件是`actions.py`，该文件包含 Python 自定义操作。如果 Facebook Messenger 是部署的漂亮面孔，Rasa 的 NLP 能力是其可爱的声音，那么`actions.py`就是部署的大脑。让我们更详细地看看`actions.py`中的代码，该代码获取 Rasa 设置的槽位值。

Rasa 与`actions.py`之间的连接是`actions.py`中自定义类中的跟踪器结构。`tracker.get_slot()`方法允许您获取 Rasa 设置的槽位值，或者如果 Rasa 没有设置槽位值，则为`None`。`actions.py`中的此循环遍历从 Rasa 传递的槽位值，并加载与槽位值相对应的评分数据框列，如果没有设置槽位值，则使用默认值，如下一列表所示。

列表 8.6 从 Rasa 加载槽位值到数据框的代码

```
for col in score_cols:
    if tracker.get_slot(col) != None:                      ❶ 
        if tracker.get_slot(col) == "today":
            score_df.at[0,col] = score_default[col]        ❷ 
        else:
            score_df.at[0,col] = tracker.get_slot(col)     ❸ 
    else:
        score_df.at[0,col] = score_default[col]            ❹ 
```

❶ 如果设置了槽位，则使用其值。

❷ 如果日期由 Rasa 设置为今天，则使用默认值，即当前日期。

❸ 否则，对于将要评分的数据框，将值设置为等于 Rasa 的槽位值。

❹ 如果 Rasa 没有设置值（例如日期和时间），则使用默认值，即当前时间/日期。

本节简要概述了 Rasa 的一些关键概念。有关更多详细信息，您可以在 [`rasa.com`](https://rasa.com/) 上了解 Rasa 及其架构。

## 8.10 使用 Rasa 在 Facebook Messenger 中部署您的模型的步骤

本节描述了使用 Facebook Messenger 部署您的模型的步骤。完成这些步骤后，您将拥有一个部署的深度学习模型，您可以从 Facebook Messenger 中查询它。

当您克隆与本书相关的 GitHub 仓库 ([`mng.bz/v95x`](http://mng.bz/v95x)) 时，您将在`deploy`子目录中找到与 Facebook Messenger 部署相关的文件，如下所示。

列表 8.7 与 Facebook Messenger 部署相关的代码

```
│ actions.py              ❶ 
│   config.yml              ❷ 
│   credentials.yml         ❸ 
│   custom_classes.py       ❹ 
│   deploy_config.yml       ❺ 
│   domain.yml              ❻ 
│   endpoints.yml           ❼ 
│   one_off_scoring.py  
│   __init__.py
│ 
├── data
│       nlu.md              ❽ 
│       stories.md          ❾ 
│ 
└── models                  ❿ 
```

❶ 包含 Rasa 模型自定义操作的文件

❷ Rasa 配置文件

❸ Rasa 凭证文件

❹ 包含管道类的定义

❺ `actions.py`的配置文件：管道文件名、模型文件名和调试设置

❻ Rasa 领域文件：指定意图、槽位和操作

❼ Rasa 端点文件：指定自定义操作的端点 URL。

❽ Rasa `nlu.md`文件：Rasa 模型的单轮对话训练数据

❾ Rasa `stories.md`文件：Rasa 模型的多轮对话训练数据

❿ 包含 Rasa 模型的目录

在步骤 1 到 4 中，您将通过安装 Python（如果您尚未在本地系统上安装它）和 Rasa 开源聊天机器人环境来完成基本设置：

1.  如果您本地系统上尚未安装 Python 3.7，请安装它 ([`www.python.org/downloads`](https://www.python.org/downloads))。

    注意：Python 3.8 中 TensorFlow 依赖项与 Rasa 存在问题，因此请确保您使用的是 Python 3.7，以避免在 Rasa 安装步骤中出现问题。另外请注意，Rasa 与 TensorFlow 2 不兼容，因此您用于 Facebook Messenger 部署的 Python 环境需要与您用于训练 Keras 模型的 Python 环境分开。

1.  安装开源聊天机器人框架 Rasa ([`rasa.com/docs/rasa/user-guide/installation`](https://rasa.com/docs/rasa/user-guide/installation))：

    ```
    pip install rasa
    ```

1.  如果您在 Windows 上，并且 Rasa 安装失败，显示需要 C++ 的消息，您可以下载并安装 Visual C++ Build Tools ([`mng.bz/4BA5`](http://mng.bz/4BA5))。安装构建工具后，重新运行 Rasa 安装：

1.  前往您克隆的仓库中的部署目录。

1.  在部署目录中运行以下命令以设置基本的 Rasa 环境：

    ```
    rasa init
    ```

1.  在您的部署目录中运行以下命令以在 Rasa 的 Python 环境中调用`actions.py`。如果您收到任何关于缺少库的消息，请运行`pip install`以添加缺少的库：

    ```
    rasa run actions
    ```

1.  在步骤 6 到 13 中，您将设置 ngrok（用于连接本地系统上的部署环境与 Facebook Messenger）并设置您需要连接到 Facebook Messenger 的 Facebook 应用和 Facebook 页面：

1.  安装 ngrok ([`ngrok.com/download`](https://ngrok.com/download))。

1.  在您安装 ngrok 的目录中，调用 ngrok 使您的本地主机在端口 5005 上可供 Facebook Messenger 使用。以下是 Windows 的命令：

    ```
    .\ngrok http 5005 
    ```

1.  记下 ngrok 输出的 https 转发 URL，如图 8.12 所示；您将需要该 URL 来完成步骤 13。

    ![CH08_F12_Ryan](img/CH08_F12_Ryan.png)

    图 8.12 调用 ngrok 的输出

1.  在部署目录中运行以下命令以训练 Rasa 模型：

    ```
    rasa train
    ```

1.  按照以下说明 [`mng.bz/Qxy1`](http://mng.bz/Qxy1) 添加新的 Facebook 应用。记下页面访问令牌和应用程序密钥；您需要在步骤 11 中使用这些值更新`credentials.yml`文件。

1.  更新部署目录中的`credentials.yml`文件，以设置验证令牌（您选择的字符串值）和密钥以及页面访问令牌（在步骤 10 中您设置的 Facebook 应用设置期间提供）：

    ```
    facebook:  
      verify: <verify token that you choose>
      secret: <app secret from Facebook app setup>
      page-access-token: <page access token from Facebook app setup>
    ```

1.  在部署目录中运行以下命令以启动 Rasa 服务器，使用您在步骤 11 中设置的`credentials.yml`中的凭据：

    ```
    rasa init
    ```

1.  在第 10 步中创建的 Facebook 应用中，选择 Messenger -> 设置，滚动到 Webhooks 部分，并点击编辑回调 URL。将回调 URL 值的初始部分替换为你在第 7 步调用 ngrok 时记录的 https 转发 URL。在验证令牌字段中输入你在第 11 步设置的`credentials.yml`中的验证令牌，然后点击验证并保存，如图 8.13 所示。

    ![CH08_F13_Ryan](img/CH08_F13_Ryan.png)

    图 8.13 设置 Facebook Messenger 的 webhook 回调 URL

1.  最后，在第 14 步和第 15 步中，验证你在 Facebook Messenger 中的部署：

1.  在 Facebook Messenger（移动或网页应用）中，搜索你在第 10 步中创建的 Facebook 页面的 ID，并向该 ID 发送以下消息：

    ```
    Will Queen west be delayed
    ```

1.  如果你的部署成功，你将看到如图 8.14 所示的响应，这是从我本地系统提供的。不用担心预测是否为延迟；只需确认你收到了响应。

    ![CH08_F14_Ryan](img/CH08_F14_Ryan.png)

    图 8.14 使用 Facebook Messenger 成功进行模型部署的合理性测试

## 8.11 管道简介

现在我们已经完成了模型的部署过程，我们需要回顾一个重要的过程部分，它使得准备用户输入以便模型能够对其生成预测成为可能：管道。通过使用管道，我们可以将完全相同的准备步骤（例如，将分类列中的值分配给数值标识符）应用于用户输入的电车行程详情，这些步骤与我们训练 Keras 模型时应用于训练数据时相同。

让我们看看用户在 Facebook Messenger 部署中期望如何输入对电车行程延迟预测的请求，并将其与训练模型期望作为预测输入的内容进行比较。图 8.15 展示了用户请求与模型期望输入之间的差距。

![CH08_F15_Ryan](img/CH08_F15_Ryan.png)

图 8.15 如何从用户输入转换为模型期望的格式

我们需要有一种方便的方式来准备用户提供的用于新预测的数据，使其符合训练模型期望的格式。正如我们在第 8.6 节中看到的，Rasa 通过从用户的请求中提取必要信息并推断缺失信息，帮助我们走了一半的路，如图 8.16 所示。

![CH08_F16_Ryan](img/CH08_F16_Ryan.png)

图 8.16 如何从 Rasa 输出转换为模型期望的格式

我们如何将 Rasa 从用户输入中提取的数据点转换为模型期望的格式？特别是，我们如何将诸如路线、月份和星期几这样的分类值转换为模型期望的整数标识符？

一种方法是在训练代码中包含函数，这些函数将评分代码可用的分类值编码。在评分代码中，我们可以调用这些相同的函数来对新数据点（我们希望应用模型的数据点）中的分类值（如路线和方向）进行编码。问题是如何确保在评分过程中使用与我们在训练数据集期间使用的相同映射。例如，如果“2019”在训练过程中被映射为“5”，那么我们如何确保在评分过程中“2019”发生相同的映射？我们可以将训练过程中使用的编码器对象进行 pickle，然后在使用训练模型评分新数据时解 pickle 并应用这些相同的编码器，但这个过程会繁琐且容易出错。我们需要的是一个方便的方式来封装我们用来准备训练数据的数据准备过程，以便我们可以将相同的流程应用于将被训练模型评分的新数据点。实现这一目标的有效方法之一是使用 scikit-learn 提供的管道功能([`mng.bz/X0Pl`](https://shortener.manning.com/X0Pl))。

scikit-learn 中的管道功能使得将所有数据转换（以及如果您愿意，还包括模型本身）封装在一个对象中成为可能，您可以将其作为一个整体进行训练。当您训练了一个管道后，您可以在评分新数据点时应用它；管道会处理新数据点上的所有数据转换，然后应用模型以获得结果。

scikit-learn 中的管道功能是为了与 scikit-learn 中包含的经典机器学习算法一起使用而设计的，包括支持向量机、逻辑回归和随机森林。虽然可以创建一个包含 Keras 深度学习模型的 scikit-learn 管道，但我还没有能够成功地创建一个多输入 Keras 模型（如电车延误预测模型）的此类管道。因此，当我们在这本书中应用 scikit-learn 管道时，它们仅涵盖数据准备步骤，并不封装模型本身。

此外，为了解决在应用最终数据准备步骤（将数据集从 Pandas dataframe 转换为 numpy 数组字典）后将数据集分为训练、验证和测试时的问题，我们使用两个管道串联。第一个管道编码分类值（并处理任何剩余的缺失值），第二个管道将数据集从 Pandas dataframe 转换为模型期望的 numpy 数组字典。请参阅第 8.12 节，了解构成这两个管道的关键代码元素。

图 8.17 显示了用户如何输入将被训练模型评分的新数据点，Rasa 如何解释这些数据点，然后通过管道处理，以便它们以正确的格式供训练的 Keras 模型进行预测。

![CH08_F17_Ryan](img/CH08_F17_Ryan.png)

图 8.17 从用户输入到模型期望的完整流程

在将训练模型应用于新数据点之前，应用于用户输入的新数据点的管道与在训练模型之前应用于数据集的管道相同，如图 8.18 所示。

![CH08_F18_Ryan](img/CH08_F18_Ryan.png)

图 8.18 训练数据通过与评分数据相同的管道

在本节中，我们介绍了管道的概念，并从高层次上展示了它们在训练和部署过程中的位置。在下一节中，我们将深入研究定义管道的代码。

## 8.12 模型训练阶段定义管道

既然我们已经了解了管道的整体目的，让我们来看看实现街道延误预测项目中使用的管道的代码细节。在本节中，我们将回顾如何在`streetcar_model_training`笔记本中定义管道。在第 8.13 节中，我们将回顾管道在模型部署过程中的评分过程中应用于新数据点的细节。

scikit-learn 中的管道功能附带了一组可现成使用的转换类，或者您可以通过创建新的类作为核心管道类的子类来创建自己的自定义转换器。对于街道延误预测项目，我们通过从 scikit-learn 提供的类中派生新的 Python 类来创建自定义转换器。您可以在 [custom_classes.py](https://github.com/ryanmark1867/deep_learning_for_structured_data/blob/master/notebooks/custom_classes.py) 中找到这些类的定义：

+   `encode_categorical` — 对类别列（如路线、方向和年份）进行编码。

+   `prep_for_keras_input` — 将数据集从 Pandas 数据框转换为 Keras 模型期望的格式：一个 numpy 数组的字典。

+   `fill_empty` — 用占位符值替换空值。

+   `encode_text` — 对文本列进行编码（未用于街道延误项目）。

您可能会想知道为什么这些类定义与代码的其他部分分开在一个单独的文件中。有两个原因：

+   这些类需要由包含训练模型代码的`streetcar_model_training`笔记本和包含 Facebook Messenger 部署评分代码的`actions.py`使用。由于这两个 Python 程序都需要访问相同的类定义，因此将类定义放在一个单独的文件中，这两个程序都可以导入，是有意义的。

+   如果直接将类定义包含在评分代码文件中，类定义将无法正确解析。将类定义放在单独的文件中允许它们在将类导入评分代码 [actions.py](https://github.com/ryanmark1867/deep_learning_for_structured_data/blob/master/deploy/actions.py) 时正确解析。导入语句如下：
