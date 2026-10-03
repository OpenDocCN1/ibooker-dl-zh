

# 第十章：创建一个编码助手项目：这会早些时候帮到您

## 本章涵盖

+ 将编码模型部署到 API
+ 在本地设置 VectorDB 并使用它作为检索增强生成系统
+ 构建一个 VS Code 扩展来使用我们的 LLM 服务
+ 从项目中获得的见解和经验教训

> 进步不是来自早起的人——进步是由寻找更简单做事方式的懒惰人创造的。——罗伯特·海因莱因

如果您在工作中接触代码，您可能梦想过有一个 AI 助手来帮助您。实际上，也许您已经这样做了。随着市场上出现像 GitHub Copilot 这样的工具，我们已经看到 LLM 将自动完成提升到了新的水平。然而，并不是每个公司都对市场上的产品感到满意，也不是每个爱好者都能负担得起。所以，让我们自己动手吧！

在本章中，我们将构建一个 Visual Studio Code (VS Code) 扩展，使我们能够在代码编辑器中使用我们的 LLM。首选的编辑器将是 VS Code，因为它是一个流行的开源代码编辑器。流行可能是一个低估，因为 Stack Overflow 2023 开发者调查表明它是 81% 开发者的首选编辑器。¹ 它基本上是 Visual Studio 的轻量级版本，Visual Studio 是一个自 1997 年以来一直存在的完整 IDE。

除了选择一个特定的编辑器，我们还将做出一些明智的决定来限制项目的范围，使其更有意义。例如，在上一个项目中，我们专注于构建一个可以部署的出色的 LLM 模型。在这个项目中，我们将从已经训练过编码问题的开源模型开始。为了定制它，我们不会微调，而是围绕它构建一个 RAG 系统，这将使我们更容易保持其更新。此外，由于我们不会训练自己的模型，我们将专注于构建一个擅长 Python 的助手，这是我们在整本书中使用的语言，而不用担心其他所有语言。

现在我们已经清楚地知道了我们要构建什么，并且有一个目标在心中，让我们开始吧！

## 10.1 我们的模型

由于我们只关注 Python，我们决定使用 DeciCoder 作为我们的模型。DeciCoder 是一个只有 1B 参数的商业开源模型。² 尽管它的体积很小，但它确实擅长它所做的事情。它是在 Stack 数据集上训练的，但过滤后只包含 Python、Java 和 JavaScript 代码。它只训练了三种语言，这通常是一个限制，但实际上它是它之所以如此出色的秘密之一。

一些其他需要注意的限制是，它只有一个 2,048 个标记的上下文窗口，对于一个这个规模的模型来说并不差，但当我们考虑到我们计划使用 RAG 系统，并需要给它提供代码示例时，它就显得相对较小了。代码示例通常相当大，这限制了我们可以做的事情和我们可以提供的示例数量。

使用 RAG 与 DeciCoder 的大问题是模型没有进行指令微调。相反，它是为了击败 HumanEval 数据集（[`github.com/openai/human-eval`](https://github.com/openai/human-eval)）而设计的。在这个评估数据集中，模型只给出一个函数名和描述函数应该做什么的文档字符串。仅从这个输入，模型将生成可执行的代码来完成函数。因此，很难知道从 RAG 系统中提供更多上下文是否会帮助模型，但我们将继续尝试找出答案！

最后，它的小巧体积实际上使它成为另一个有趣的选择。因为它如此小巧，我们有可能将模型直接放入我们正在构建的 VS Code 扩展程序中，使用我们在其他章节中讨论的编译方法。这将使我们能够构建一个非常紧凑的应用程序！我们在这本书中不会这样做，主要是因为这将要求我们编写大量的 JavaScript。这是一个问题，因为我们只期望我们的读者熟悉 Python，所以在这里深入解释细节有点过于冒险，但我们将其留作对 JavaScript 高手读者的练习。

我们将要做的是将我们的模型作为 API 提供服务，您可以在本地运行它，并且可以从扩展程序中调用它。在列表 10.1 中，我们创建了一个简单的 FastAPI 服务来提供我们的模型。实际上，您在第六章中已经看到了大部分代码，我们只做了一些小的修改。第一个修改是我们将代码更改为使用 DeciCoder 模型和分词器。第二个修改稍微复杂一些，但我们添加了 `stop` 标记。这些标记会通知模型在遇到它们时停止生成。这是通过创建一个 `StoppingCriteria` 类来实现的。我们选择的标记在我们定义了提示后会有更多的意义，但本质上，我们希望模型一次创建一个函数。

##### 列表 10.1 使用 DeciCoder 的简单 FastAPI 端点

```py
import argparse

from fastapi import FastAPI, Request
from fastapi.responses import Response
import torch
import uvicorn

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

torch.backends.cuda.enable_mem_efficient_sdp(False)    #1
torch.backends.cuda.enable_flash_sdp(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

stop_tokens = ["def", "class", "Instruction", "Output"]     #2
stop_token_ids = [589, 823, 9597, 2301]

class StopOnTokens(StoppingCriteria):
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> bool:
        stop_ids = stop_token_ids
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

tokenizer = AutoTokenizer.from_pretrained("Deci/DeciCoder-1b")    #3
tokenizer.add_special_tokens( #3
    {"additional_special_tokens": stop_tokens}, #3
    replace_additional_special_tokens=False, #3
) #3
model = AutoModelForCausalLM.from_pretrained( #3
    "Deci/DeciCoder-1b", torch_dtype=torch.bfloat16, trust_remote_code=True #3
) #3
model = model.to(device) #3

app = FastAPI()     #4

@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate LLM Response

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")

    # ...      #5

    inputs = tokenizer(prompt, return_tensors="pt").to(device)    #6
    response_tokens = model.generate(
        inputs["input_ids"],
        max_new_tokens=1024,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
    )
    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(
        response_tokens[0][input_length:], skip_special_tokens=True
    )

    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()     #7
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="debug")
```

#1 火炬设置
#2 定义停止行为
#3 加载分词器和模型
#4 运行 FastAPI
#5 RAG 将在这里。
#6 生成响应
#7 启动服务；默认端口为 8000

假设这个列表在一个名为 `server.py` 的 Python 脚本中，您可以通过运行 `$ python server.py` 来启动服务器。一旦它启动并运行，让我们发送一个请求来确保它正确工作。在一个新的终端中，我们可以使用简单的提示向 API 发送一个 `curl` 请求：

```py
$ curl --request POST --header "Content-Type: application/json" --data 
↪ '{"prompt":"def hello_world(name):"}' http://localhost:8000/generate
```

响应应该是一个简单的 Python 函数，用于完成“Hello World”函数。我们从服务器收到的响应是 `return f"Hello {name}!"`。到目前为止，一切顺利！接下来，我们将定制 API 以利用 RAG 系统。

## 10.2 数据为王

现在我们已经决定了一个模型，让我们为我们的 RAG 系统准备一个数据集。RAG 是在不需要微调模型的情况下向模型引入上下文的有效方法；它还允许我们根据我们的数据自定义结果。本质上，如果你想让你的模型了解你组织不断变化的代码库的上下文，RAG 是一个很好的系统。有一个擅长编码的模型很好，但我们希望它擅长 *我们的* 代码。我们希望它使用正确的变量名和导入内部构建的定制依赖项——诸如此类的事情。在本节中，我们将设置一个 VectorDB，上传一个 Python 编码数据集，然后更新我们刚刚构建的 API 以利用它。

### 10.2.1 我们的 VectorDB

在我们真正深入数据集之前，我们首先需要设置我们的基础设施。当然，如果你的数据集足够小，你可以将其加载到内存中，并使用 Faiss 或 USearch 等工具直接在 Python 中运行相似度搜索，但那样有什么乐趣呢？此外，我们还想向你展示 Milvus。

Milvus 是一个出色的开源 VectorDB，与这个领域的大玩家竞争。你可以在本地或大型云集群上运行它，因此它可以很容易地扩展到你的需求。如果你不想处理设置，还有可管理的 Milvus 集群可供选择。我最喜欢的功能是其支持 GPU 的版本，这使得向量搜索变得非常快。

幸运的是，社区也使 Milvus 非常易于接近和设置。事实上，独立版本只需要 Docker 来运行，并附带一个启动脚本，使其更加容易。由于我们将在本项目中本地运行一切，我们将使用独立版本（了解更多信息，请参阅 [`mng.bz/aVE9`](https://mng.bz/aVE9)）。为此，我们需要在终端中运行以下命令：

```py
$ wget https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/
↪ standalone_embed.sh
$ bash standalone_embed.sh start
```

第一条命令将会下载一个 shell 脚本，第二条命令将会运行它。这个脚本实际上只是为了方便，因为 Docker 的 `run` 命令相当长。它还包括两个你应该了解的额外命令。`Stop` 命令，它将停止你的 Milvus Docker 容器，是

```py
$ bash standalone_embed.sh stop
```

`delete` 命令，当你不再希望保留数据时，将会从你的电脑中删除所有数据，它是

```py
$ bash standalone_embed.sh delete
```

你现在不需要运行这些命令，但记住它们，等我们完成后再用。现在我们已经设置了数据库，让我们让它变得有用，并将一些数据加载到其中。

### 10.2.2 我们的数据集

如果这是一个研讨会，我们会向你展示如何编写一个脚本来从 GitHub 拉取你组织的代码，并使用它来增强你的提示。我们甚至可以设置一个 GitHub Actions 流水线，以便每次代码合并到主分支时更新我们的 VectorDB。但由于我们没有访问你的代码，这只是一个书籍，我们将做合理的事情并使用开源数据集。

我们将为我们的项目选择 Alpaca 数据集。Alpaca 数据集是由斯坦福大学在训练同名模型时编译的，使用了蒸馏和 GPT-3 作为导师模型。由于它是合成数据，数据集非常干净，这使得它易于处理。事实上，它如此简单，以至于网上的多个副本已经过滤掉了所有的 Python 代码示例。这个子集包含 18.6K 个 Python 编码挑战，包括一个任务或指令和生成的代码——非常适合我们想要达成的目标。

在列表 10.2 中，我们创建我们的管道来将数据集加载到 Milvus 中。我们创建了一个 `PythonCodeIngestion` 类来处理我们的数据集分块和批量上传的细节。请注意，我们使用了 `krlvi/sentence-t5-base-nlpl-code_search_net` 嵌入模型。这个嵌入模型已经在 CodeSearchNet 数据集（[`github.com/github/CodeSearchNet`](https://github.com/github/CodeSearchNet)）上进行了专门训练，非常适合创建代码的有意义嵌入。

##### 列表 10.2：摄取 Alpaca 的数据管道

