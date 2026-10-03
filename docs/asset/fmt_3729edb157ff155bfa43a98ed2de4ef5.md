
```
def serialize(self, h5file):
    h5file.create_group('encoder')
    h5file['encoder'].attrs['name'] = self.encoder.name()
    h5file['encoder'].attrs['board_width'] = self.encoder.board_width
    h5file['encoder'].attrs['board_height'] = self.encoder.board_height
    h5file.create_group('model')
    kerasutil.save_model_to_hdf5_group(self.model, h5file['model'])
```

最后，在你序列化了一个模型之后，你还需要知道如何从 HDF5 文件中加载它。

#### 列表 8.6. 从 HDF5 文件反序列化`DeepLearningAgent`

```
def load_prediction_agent(h5file):
    model = kerasutil.load_model_from_hdf5_group(h5file['model'])
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = encoders.get_encoder_by_name(
        encoder_name, (board_width, board_height))
    return DeepLearningAgent(model, encoder)
```

这完成了我们对深度学习代理的定义。作为下一步，你必须确保这个代理能够连接并与环境交互。你通过将`DeepLearningAgent`嵌入到一个人类玩家可以在浏览器中与之对弈的 Web 应用程序中来实现这一点。

### 8.2. 将你的围棋机器人托管到 Web 前端

在第六章和第七章中，你设计和训练了一个神经网络，用于预测人类在围棋游戏中的下一步棋。在第 8.1 节中，你将这个用于*预测*的模型转换成了一个`DeepLearningAgent`，用于*选择*走棋。下一步就是让你的机器人下棋！回到第三章，你构建了一个基本的界面，你可以通过键盘输入走棋，而你那无知的`RandomBot`会将它的回复打印到控制台。现在你已经构建了一个更复杂的机器人，它值得一个更好的前端来与人类玩家进行交互。

在本节中，你将连接`DeepLearningAgent`到一个 Python 网络应用，这样你就可以在浏览器中与之对弈。你将使用轻量级的 Flask 库通过 HTTP 提供这样的代理。在浏览器端，你将使用一个名为 jgoboard 的 JavaScript 库来渲染人类可以使用的围棋盘。代码可以在我们 GitHub 上的仓库中找到，在 dlgo 的 httpfrontend 模块中。我们在这里没有明确讨论这段代码，因为我们不想通过涉及其他语言的 Web 开发技术（如 HTML 或 JavaScript）来分散对构建围棋 AI 这一主要主题的注意力。相反，我们将提供一个关于应用程序做什么以及如何在端到端示例中使用它的概述。图 8.1 提供了你将在本章中构建的应用程序概述。

#### 图 8.1. 为你的围棋机器人构建一个网络前端。httpfrontend 模块启动一个 Flask 网络服务器，它解码 HTTP 请求并将它们传递给一个或多个围棋机器人代理。在浏览器中，基于 jgoboard 库的客户端通过 HTTP 与服务器通信。

![图片 8.1](img/08fig01_alt.jpg)

如果你查看 httpfrontend 的结构，你会找到一个名为 server.py 的文件，它有一个单一、文档良好的方法`get_web_app`，你可以使用它来返回一个要运行的网络应用。以下是如何使用`get_web_app`加载一个随机机器人并为其提供服务的示例。

#### 列表 8.7. 注册一个随机代理并使用它启动网络应用

```
from dlgo.agent.naive import RandomBot
from dlgo.httpfrontend.server import get_web_app

random_agent = RandomBot()
web_app = get_web_app({'random': random_agent})
web_app.run()
```

当你运行这个示例时，一个网络应用将在本地主机（127.0.0.1）上启动，监听端口 5000，这是 Flask 应用中使用的默认端口。你刚刚注册的`RandomBot`作为`random`对应于 httpfrontend 中的静态文件夹中的一个 HTML 文件：play_random_99.html。在这个文件中，一个围棋盘被渲染出来，这也是定义人机游戏规则的地方。人类对手开始时使用黑子；机器人使用白子。每当人类移动后，就会触发 route/select-move/random 来从机器人那里接收下一步移动。收到机器人的移动后，它将被应用到棋盘上，然后轮到人类移动。要与此机器人对弈，请在浏览器中导航到 http://127.0.0.1:5000/static/play_random_99.html。你应该会看到一个可玩演示，如图 8.2 所示。

#### 图 8.2. 在浏览器中运行 Python 网络应用以与围棋机器人对弈

![图片 8.2](img/08fig02.jpg)

在接下来的章节中，你将添加越来越多的机器人，但到目前为止请注意，在 play_predict_19.html 下还有一个可用的前端。这个网络前端与一个名为 predict 的机器人通信，可以用来玩 19×19 的游戏。因此，如果你在围棋数据上训练一个 Keras 神经网络模型并使用一个围棋盘`encoder`，你可以首先创建一个实例`agent = DeepLearningAgent(model, encoder)`，然后将其注册到一个网络应用`web_app = get_web_app({'predict': agent})`中，然后你可以通过`web_app.run()`启动它。

#### 8.2.1. 一个端到端的围棋机器人示例

图 8.3 展示了涵盖整个过程的端到端示例（与我们在第七章开头介绍的过程相同）。您从所需的导入开始，并使用编码器和围棋数据处理器将围棋数据加载到特征*X*和标签*y*中，如列表 8.8 所示。

#### 列表 8.8. 使用处理器从围棋数据中加载特征和标签

```
import h5py

from keras.models import Sequential
from keras.layers import Dense

from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.httpfrontend import get_web_app
from dlgo.networks import large

go_board_rows, go_board_cols = 19, 19
nb_classes = go_board_rows * go_board_cols
encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))
processor = GoDataProcessor(encoder=encoder.name())

X, y = processor.load_go_data(num_samples=100)
```

#### 图 8.3. 深度学习围棋机器人的训练过程

![图片](img/08fig03_alt.jpg)

配备了特征和标签，您可以使用深度卷积神经网络并使用 Adadelta 作为优化器来训练它。这次，您从 dlgo.networks 中选择大型网络。

#### 列表 8.9. 使用 Adadelta 构建和运行大型围棋移动预测模型

```
input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
model = Sequential()
network_layers = large.layers(input_shape)
for layer in network_layers:
    model.add(layer)
model.add(Dense(nb_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta',
     metrics=['accuracy'])

model.fit(X, y, batch_size=128, epochs=20, verbose=1)
```

模型训练完成后，您可以从它创建一个围棋机器人并将其保存为 HDF5 格式。

#### 列表 8.10. 创建并持久化一个`DeepLearningAgent`

```
deep_learning_bot = DeepLearningAgent(model, encoder)
deep_learning_bot.serialize("../agents/deep_bot.h5")
```

最后，您可以从文件中加载机器人并将其托管在 Web 应用程序中。

#### 列表 8.11. 将机器人加载回内存并在 Web 应用程序中托管它

```
model_file = h5py.File("../agents/deep_bot.h5", "r")
bot_from_file = load_prediction_agent(model_file)

web_app = get_web_app({'predict': bot_from_file})
web_app.run()
```

当然，如果您已经训练了一个强大的机器人，您可以跳过除了最后部分之外的所有内容。例如，您可以加载存储在第七章检查点中的一个模型，并通过相应地更改 model_file 来查看它们作为对手的表现。

### 8.3. 在云端训练和部署围棋机器人

到目前为止，所有开发都是在您家里的本地机器上进行的。如果您幸运地能在电脑上使用现代 GPU，那么训练我们在第五章–第七章中开发的深度神经网络对您来说不是问题。如果您没有强大的 GPU 或无法在它上面节省任何计算时间，那么在云端租用 GPU 的计算时间通常是一个不错的选择。

如果您现在不考虑训练，并假设您已经有一个强大的机器人，那么托管这个机器人是云服务提供商可以派上用场的情况之一。在第 8.2 节中，您通过本地主机托管的一个 Web 应用程序运行了一个机器人。如果您想与朋友分享您的机器人或使其公开，这并不完全理想。您既不想确保您的电脑日夜运行，也不想让公众访问您的机器。通过在云端托管您的机器人，您可以分离开发和部署，并可以简单地与任何对玩您的机器人感兴趣的人分享一个 URL。

因为这个话题很重要，但有些特殊，并且仅与机器学习间接相关，所以我们完全将其外包给了 附录 D。阅读并应用附录中的技术是可选的，但推荐。在 附录 D 中，你将学习如何开始使用一个特定的云服务提供商，亚马逊网络服务（AWS）。附录中你将学习以下技能：

+   在 AWS 上创建账户

+   灵活地设置、运行和终止虚拟服务器实例

+   在云 GPU 上以合理的成本创建适合深度学习模型训练的 AWS 实例

+   在一个（几乎）免费的服务器上部署通过 HTTP 服务的 Go 机器人

在学习这些有用的技能的基础上，附录 D 也是部署一个连接到在线围棋服务器的完整 Go 机器人的先决条件，我们将在第 8.6 节中介绍这个话题。

### 8.4. 与其他机器人交谈：Go 文本协议

在 第 8.2 节 中，你看到了如何将你的机器人框架集成到网页前端。为了实现这一点，你处理了机器人和人类玩家之间的通信，使用了超文本传输协议（HTTP），这是运行网络的核心理协议之一。为了避免分心，我们故意省略了所有细节，但有一个 *标准化协议* 在位是必要的。人类和机器人没有共享的语言来交换围棋走法，但协议可以作为一座桥梁。

Go 文本协议（GTP）是全球围棋服务器使用的默认标准，用于连接其平台上的人类和机器人。许多离线围棋程序也基于 GTP。本节通过示例介绍 GTP；你将在 Python 中实现协议的一部分，并使用这个实现让你的机器人与其他围棋程序对弈。

在 附录 C 中，我们解释了如何安装 GNU Go 和 Pachi，这两种常见的围棋程序几乎适用于所有操作系统。我们建议安装这两个程序，所以请确保你的系统上安装了这两个程序。你不需要任何前端，只需要纯命令行工具。如果你已经安装了 GNU Go，你可以通过运行以下命令以 GTP 模式启动它：

```
gnugo --mode gtp
```

使用此模式，你现在可以探索 GTP 的工作原理。GTP 是一种基于文本的协议，因此你可以在你的终端中输入命令并按 Enter 键。例如，要设置一个 9 × 9 的棋盘，你可以输入 `boardsize 9`。这将触发 GNU Go 返回响应并确认命令已正确执行。每个成功的 GTP 命令都会触发一个以符号 `=` 开头的响应，而失败的命令则会导致一个 `?`。要检查当前棋盘状态，你可以发出 `showboard` 命令，这将打印出一个空白的 9 × 9 棋盘，正如预期的那样。

在实际游戏中，两个命令是最重要的：`genmove`和`play`。第一个命令`genmove`用于请求 GTP 机器人生成下一步棋。GTP 机器人通常会将其应用于其游戏状态。这个命令需要的参数只有玩家颜色，即黑或白。例如，要生成白棋并将其放置在 GNU Go 的棋盘上，请输入`genmove white`。这将导致一个响应，如`= C4`，这意味着 GNU Go 接受这个命令（`=`）并在 C4 放置一个白棋。正如你所看到的，GTP 接受标准坐标，如第二章和第三章中介绍的那样。

对于我们来说，与游戏玩法相关的另一个操作是`play`。这个命令用于让一个 GTP 机器人知道它需要在棋盘上走一步棋。例如，你可以通过发出`play black D4`来告诉 GNU Go 你想让它走 D4 的黑棋，这将返回一个`=`来确认这个命令。当两个机器人相互对战时，它们会轮流要求对方生成下一步棋，然后在它们自己的棋盘上`play`这一步棋。这一切都很直接——但我们省略了很多细节。一个完整的 GTP 客户端有很多更多的命令来处理，从处理让子棋到管理时间设置和计分规则。如果你对 GTP 的细节感兴趣，请参阅[`mng.bz/MWNQ`](http://mng.bz/MWNQ)。话虽如此，在基本层面上，`genmove`和`play`将足以让你的深度学习机器人与 GNU Go 和 Pachi 对战。

为了处理 GTP 并包装你的`Agent`概念，以便可以通过使用此协议交换围棋走法，你创建了一个新的 dlgo 模块，名为 gtp。你仍然可以尝试跟随本正文中的实现，但从本章开始，我们建议直接跟随我们在 GitHub 上的实现[`mng.bz/a4Wj`](http://mng.bz/a4Wj)。

首先，让我们正式化 GTP 命令的定义。为此，我们必须注意，在许多围棋服务器上，命令会得到一个序列号以确保我们可以匹配命令和响应。这些序列号是可选的，可以是`None`。对于我们来说，一个 GTP 命令由一个序列号、一个命令以及可能对该命令的多个参数组成。你将这个定义放在`command.py`中的`gtp`模块。

#### 列表 8.12. Python 实现 GTP 命令

```
class Command:

    def __init__(self, sequence, name, args):
        self.sequence = sequence
        self.name = name
        self.args = tuple(args)

    def __eq__(self, other):
        return self.sequence == other.sequence and \
            self.name == other.name and \
            self.args == other.args

    def __repr__(self):
        return 'Command(%r, %r, %r)' % (self.sequence, self.name, self.args)

    def __str__(self):
        return repr(self)
```

接下来，你想要将命令行中的文本输入解析为`Command`。例如，解析“999 play white D4”应该得到`Command(999, 'play', ('white', 'D4'))`。用于此的`parse`函数也包含在`command.py`中。

#### 列表 8.13. 从纯文本解析 GTP `Command`
