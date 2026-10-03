
# 第二章：介绍 TensorFlow.js

> “如果你的行动激励他人梦想更多，学习更多，
> 
> 做更多，成为更多，你就是一个领导者。”
> 
> —约翰·昆西·亚当斯

我们已经稍微谈到了 TensorFlow.js 以及它的功能，但我们还没有真正深入探讨像 TensorFlow.js 这样的机器学习框架到底是什么。在本章中，我们将探讨机器学习框架的概念，然后迅速进入编写代码。我知道编写具有某种实际结果的代码很重要，所以在本章中，你最终将让你的计算机运行 TensorFlow.js 并产生结果。

我们将：

+   看看 TensorFlow.js 的概念
+   设置 TensorFlow.js
+   运行一个 TensorFlow.js 模型包
+   深入了解 AI 的工作原理

让我们从我们将使用的框架开始。

## 你好，TensorFlow.js

考虑到我们之前的章节讨论了古代哲学和机器学习作为一个领域的诞生，你会期望人工智能框架的历史可以追溯到上世纪 60 年代初。然而，人工智能长时间停滞不前，这段时间通常被称为“人工智能寒冬”。人工智能的概念受到怀疑和极端数学计算的困扰，因为当时可用的数据量很小。谁能责怪这些研究人员呢？今天大多数软件开发人员依赖于发布应用程序，而不是从头开始编写支持 GPU 的线性代数和微积分，构建自己的人工智能不应该是例外。幸运的是，由于谷歌 Brain 团队的一些开源贡献，我们有了选择。

当你开始学习机器学习时，会听到很多流行词。TensorFlow、TensorFlow Lite 和 TensorFlow.js 都可能被提到，对于大多数新手来说，这些术语的含义以及为什么会有三个都不清楚。现在，让我们暂时忽略“张量”这个术语，因为你在第一章中已经听过这个词，而且在接下来的章节中你会真正理解它。相反，让我们专注于定义 TensorFlow.js，以便我们可以使用它。

TensorFlow，没有任何额外的“.js”或“Lite”，是谷歌的第一个公开的机器学习框架；谷歌 Brain 团队于 2015 年底发布了它。这个框架专注于用 Python 在云端有效解决谷歌的机器学习问题。谷歌很快意识到将这个流行的框架推广到计算能力有限的物联网和移动设备上会有好处，这就需要对 TensorFlow 进行适应，这就是所谓的 TensorFlow Lite。这一成功的适应为将 TensorFlow 理念推广到其他语言铺平了道路。

你可能可以猜到接下来会发生什么。2018 年初，谷歌宣布了一个由谷歌支持的 JavaScript 导入机器学习框架 TensorFlow 的版本，称为 TensorFlow.js。这一新举措以全新的方式增强了 TensorFlow 的实用性。Daniel Smilkov、Nikhil Thorat 和 Shanqing Cai 是发布 TensorFlow.js 的团队的一部分。在[TensorFlow 开发者峰会](https://youtu.be/YB-kfeNIPCE)上，Smilkov 和 Thorat 使用计算机视觉和网络摄像头训练一个模型来控制*吃豆人*游戏。

正是在这一刻，“仅限 Python”的选项被从流行的人工智能框架选项中移除，神经网络可以有效地穿越 JavaScript 领域。*如果你可以运行 JavaScript，你就可以运行由 TensorFlow.js ML 支持的人工智能。*

这三种实现今天都是活跃的，并随着它们特定目的的增长。通过将 TensorFlow 扩展到 JavaScript 实现，我们现在可以在节点服务器甚至客户端浏览器中实现 AI/ML。在论文“TensorFlow.js: 用于 Web 和更多的机器学习”中[(Daniel Smilkov 等人，2019)](https://oreil.ly/XkIjZ)，他们表示，“TensorFlow.js 使来自庞大 JavaScript 社区的新一代开发人员能够构建和部署机器学习模型，并实现新类的设备上计算。” TensorFlow.js 可以利用广泛的设备平台，同时仍然可以访问 GPU 甚至 Web Assembly。有了 JavaScript，我们的机器学习可以涉足地平线并回来。

值得注意的是，在几项基准测试中，Node 在较低的 CPU 负载下胜过了 Python 3，因此尽管 Python 一直是大多数 AI 的采用语言，JavaScript 作为产品和服务的主要语言平台。

但没有必要删除或推广任何一种语言。TensorFlow 模型基于有向无环图（DAG），这是与语言无关的图，是训练的*输出*。这些图可以由一种语言训练，然后转换并被完全不同的编程语言消耗。本书的目标是为您提供使用 JavaScript 和 TensorFlow.js 的工具，以便充分利用它们。

## 利用 TensorFlow.js

对于很多人来说，“学习”有时可能意味着从基础开始，这意味着从数学开始。对于这些人来说，像 TensorFlow 这样的框架和像 TensorFlow.js 这样的实用分支是一个糟糕的开始。在本书中，我们将构建项目，并涉及 TensorFlow.js 框架的基础知识，我们将很少或根本不花时间在底层数学魔法上。

像 TensorFlow 和 TensorFlow.js 这样的框架帮助我们避免涉及的线性代数的具体细节。您不再需要关注*前向传播*和*反向传播*这样的术语，以及它们的计算和微积分。相反，我们将专注于像*推断*和*模型训练*这样的行业术语。

虽然 TensorFlow.js 可以访问底层 API（如`tfjs-core`）来对经典问题进行一些基本优化，但这些时刻留给了那些无论手头的框架如何都有坚实基础的学者和高级用户。这本书旨在展示 TensorFlow.js 的强大之处，利用框架的辛勤工作和优化是我们将如何做到这一点。我们让 TensorFlow.js 负责配置和优化我们的代码，以适应各种设备约束和 WebGL API。

我们甚至可能走得太远，将机器学习应用于您可以轻松手工编码的算法，但这通常是大多数人真正理解概念的地方。用机器学习解决您理解的简单问题有助于您推断解决您无法手工编码的高级问题的步骤、逻辑和权衡。

另一方面，一些关于神经元、激活函数和模型初始化的基础知识是不能被忽视的，可能需要一些解释。本书的目标是为您提供理论和实用性的健康平衡。

正如您可能已经推测到的那样，TensorFlow.js 的各种平台意味着没有单一的预设设置。我们可以在本书中在客户端或服务器上运行 TensorFlow.js。然而，我们最隐性的交互选项是充分利用浏览器。因此，我们将在浏览器中执行大部分示例。当然，在适当的情况下，我们仍将涵盖托管节点服务器解决方案的关键方面。这两种工具都有各自的优缺点，我们将在探索 TensorFlow.js 的强大之处时提到。

## 让我们准备好 TensorFlow.js

与任何流行工具一样，您可能会注意到 TensorFlow.js 包有几种不同版本，以及几个可以访问代码的位置。本书的大部分内容将专注于 TensorFlow.js 最常用和“准备运行”的版本，即浏览器客户端。优化的框架版本是为服务器端制作的。这些版本与 Python 使用相同的底层 C++核心 API 进行通信，但通过 Node.js，这使您能够利用服务器的图形卡或 CPU 的所有性能。TensorFlow.js AI 模型在各种位置运行，并利用各种环境的各种优化（请参见图 2-1）。

![Tensorflow 选项图表](img/ltjs_0201.png)

###### 图 2-1。TensorFlow.js 的选项

本书中学到的知识可以应用于大多数平台。为了方便起见，我们将覆盖最常见平台的设置过程。如果您不愿意从头开始设置环境，可以直接访问与本书相关的源代码中为您构建的预配置项目，位于[*https://github.com/GantMan/learn-tfjs*](https://github.com/GantMan/learn-tfjs)。

### 在浏览器中设置 TensorFlow.js

让我们来看看运行 TensorFlow.js 的最快、最多功能和最简单的方法。要在浏览器中运行 TensorFlow.js，实际上非常容易。我假设您熟悉 JavaScript 的基础知识，并且以前已经将 JavaScript 库导入到现有代码中。TensorFlow.js 支持多种包含方式，因此任何经验的开发人员都可以访问它。如果您熟悉包含 JavaScript 依赖项，您将熟悉这些常见做法。我们可以以两种方式将 TensorFlow.js 导入到页面中：

+   使用 NPM
+   包含脚本标签

#### 使用 NPM

管理网站依赖项的最流行方式之一是使用包管理器。如果您习惯使用 NPM 或 Yarn 构建项目，您可以通过 NPM 注册表访问代码[*https://oreil.ly/R2lB8*](https://oreil.ly/R2lB8)。只需在命令行安装依赖项：

```js
# Import with npm
$ npm i @tensorflow/tfjs

# Or Yarn
$ yarn add @tensorflow/tfjs
```

导入`tfjs`包后，您可以在 JavaScript 项目中使用以下 ES6 JavaScript 导入代码导入此代码：

```js
import * as tf from '@tensorflow/tfjs';
```

#### 包含脚本标签

如果网站不使用包管理器，您可以简单地向 HTML 文档添加一个脚本标签。这是您可以在项目中包含 TensorFlow.js 的第二种方式。您可以下载并在本地托管 TensorFlow.js，或者利用内容传送网络（CDN）。我们将把脚本标签指向 CDN 托管的脚本源：

```js
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.7.0/dist/tf.min.js">
</script>
```

除了跨网站缓存外，CDN 非常快，因为它们利用边缘位置确保全球快速交付。

###### 注意

您可能已经注意到，我已将此代码锁定到特定版本（2.7.0），我强烈建议您在涉及 CDN 的项目中始终这样做。您不希望在网站出现自动破坏性更改的问题。

### 在 Node 中设置 TensorFlow.js

我们在浏览器中使用的 TensorFlow.js 包与 Node.js 完全兼容，如果您计划仅暂时尝试 Node.js，这是一个很好的解决方案。一个好的规则是，如果您不打算为他人托管实时项目或在大量数据上进行训练，则使用简单的`/tfjs`而不是`/tfjs-node`导入。

如果您的目标是超越实验，并在 TensorFlow.js 中实现有效的 Node.js，您应该花一些时间改进您的 Node.js 设置，使用一些[这些替代包](https://oreil.ly/zREQy)。有两个更好的 TensorFlow.js 分发版本专门为 Node 和速度构建。它们是`tfjs-node`和`tfjs-node-gpu`。请记住，每台开发者机器都是独特的，您的安装和体验可能有所不同。

对于 Node.js，您可能需要在`@tensorflow/tfjs-node`和`@tensorflow/tfjs-node-gpu`之间进行选择。如果您的计算机配置了 NVIDIA GPU 并正确设置了 CUDA 软件，您可以使用后者 GPU 加速的软件包。Compute Unified Device Architecture（CUDA）允许通过并行计算平台直接访问 GPU 加速的 NVIDIA 硬件。虽然 GPU 软件包是 TensorFlow.js 选项中绝对最快的，但由于其硬件和软件约束，它也是最不可能准备好并配置好的软件包。目前，我们的示例将在安装`tfjs-node`上运行，并将可选的 CUDA 配置留给您。

```js
# Import with npm
$ npm i @tensorflow/tfjs-node

# Or Yarn
$ yarn add @tensorflow/tfjs-node
```

###### 警告

通常，如果您的计算机尚未设置为开发高级 C++库，您可能需要安装一些软件来准备好您的计算机。只有当您希望积极使用`tfjs-node`或`tfjs-node-gpu`时，才需要进行这种深入研究。

如果您的 NPM 安装成功，恭喜！您已准备好从此包中导入。如果您已经设置了 Node 来处理 ES6，您可以使用以下代码导入：

```js
import * as tf from '@tensorflow/tfjs-node';
```

如果您尚未配置 Node.js 软件包以处理 ES6 导入，您仍然可以使用经典的 require 访问代码：

```js
const tf = require('@tensorflow/tfjs-node');
```

### 验证 TensorFlow.js 是否正常工作

之前的所有方法都会在您的 JavaScript 代码中提供一个名为`tf`的变量，这使您可以访问 TensorFlow.js。为了确保我们的导入工作正常，让我们记录导入的 TensorFlow.js 库的版本。

将此代码添加到您的 JavaScript 中，如果在控制台中看到一个版本打印出来，您的导入就可以继续进行了！

```js
console.log(tf.version.tfjs);
```

运行页面时，我们可以右键单击页面并检查以访问 JavaScript 控制台日志。在那里，我们将找到我们的日志命令的输出，“3.0.0”或您导入的 TensorFlow.js 版本。对于 Node.js 示例，该值将直接在控制台中打印。

###### 警告

在访问`tf`变量（TensorFlow.js 库）的功能之前，通常需要确保 TensorFlow.js 已经正确加载了后端并准备就绪。上述代码绕过了这个检查，但始终最好运行您的初始代码等待`tf.ready()`的承诺。

#### 下载并运行这些示例

如第一章中所述，您可以访问本书中的代码。为了确保您不必在每个示例中从头开始设置这些项目，请确保您拥有每个项目的源代码，包括之前显示的简单代码。

从书的存储库中以您喜欢的方式下载项目：[*https://github.com/GantMan/learn-tfjs*](https://github.com/GantMan/learn-tfjs)。

转到第二章的目录，并确保您可以在您的计算机上运行代码。

##### 运行简单示例

在`chapter2/simple/simplest-example`中，我们避免使用 NPM，只是从 CDN 中拉取我们的代码。以当前结构化的方式，我们甚至不需要托管网站！我们只需在任何现代浏览器中打开`index.html`，它就会运行！

在某个时候，我们实际上需要托管这些简单示例，因为我们将访问需要完整 URI 的其他资产。我们可以通过使用一个小型的 Web 服务器来托管文件来轻松实现这一点。我知道的最小的 Web 服务器叫做 Web 服务器，有一个有趣的手绘“200 OK！”标志。在五分钟内，我们就可以在本地服务器上正确提供我们的文件。

您可以在[Chrome Web Store 作为扩展](https://oreil.ly/ZOedW)上找到 Chrome 的 Web 服务器。在本书中，我们有时会称此插件为“200 OK！”当您将 Web 服务器指向`index.html`文件时，它将自动为您提供文件，并且所有相邻文件都可以通过其关联的 URL 访问，正如我们将在后续课程中所需的那样。应用程序界面应该看起来像图 2-3。

![200 OK！对话框](img/ltjs_0203.png)

###### 图 2-3。Chrome 的 Web 服务器 200 OK！对话框

如果您想查看其他选项或想要链接到提到的 Chrome 插件，请查看`chapter2/extra/hosting-options.md`，找到适合您的选项。当然，如果您发现一个未列出的绝妙选项，请贡献一个拉取请求。

一旦找到一个以您喜欢的方式运行`simple-example`的服务器，您可以将该服务用于以后的所有简单选项。

##### 运行 NPM web 示例

如果您更熟悉 NPM，此项目的基本 NPM 示例使用 Parcel。Parcel 是最快的应用程序捆绑工具，零配置。它还包括热模块重新加载，以获取实时更新和出色的错误日志记录。

要运行代码，请导航至`chapter2/web/web-example`并执行 NPM 安装（`npm i`）。完成后，在`package.json`中有一个脚本可以启动所有内容。您只需运行启动脚本：

```js
$ npm run start
```

就是这样！我们将使用这种方法来运行本书中所有基于 NPM 的代码。

##### 运行 Node.js 示例

Node.js 示例与 Parcel NPM 示例一样易于运行。虽然 Node.js 通常没有明确的意见，但本书中的 Node.js 示例将包括一些明确的开发依赖项，以便我们可以使我们的 Node.js 示例代码与浏览器示例保持一致。本书中的代码将充分利用 ECMAScript。我们通过一些转译、文件监视和节点魔法来实现这一点。

为了准备这个示例，请导航至`chapter2/node-example`并执行 NPM 安装（`npm i`）。如果遇到任何问题，您可能需要运行`npm i -g ts-node nodemon node-gyp`来确保您拥有所需的库，以确保我们所有的魔法发生。一旦您的节点包正确放置，您可以随时通过运行启动脚本来启动项目：

```js
$ npm run start
```

代码通过 TypeScript 转译，并且使用`nodemon`进行重新加载。如果一切正常运行，您将在运行服务器的控制台/终端中直接看到已安装的 TensorFlow.js 版本。

## 让我们使用一些真实的 TensorFlow.js

现在我们有了 TensorFlow.js，让我们用它来创造一些史诗般的东西！好吧，这有点简化：如果那么容易，这本书就结束了。仍然有很多东西要学习，但这并不妨碍我们乘坐缆车，以获得高层视角。

TensorFlow.js 有大量预先编写的代码和模型可供我们利用。这些预先编写的库帮助我们获得利用 TensorFlow.js 的好处，而无需完全掌握底层概念。
