
```py
# utils.py

from dataclasses import dataclass
from typing import Literal, TypeAlias
from utils import count_tokens

SupportedModels: TypeAlias = Literal["gpt-3.5", "gpt-4"]
PriceTable: TypeAlias = dict[SupportedModels, float]
prices: PriceTable = {"gpt-3.5": 0.0030, "gpt-4": 0.0200}

@dataclass ![1](img/1.png)
class Message:
    prompt: str
    response: str | None ![2](img/2.png)
    model: SupportedModels

@dataclass
class MessageCostReport:
    req_costs: float
    res_costs: float
    total_costs: float

# Define count_tokens function as normal
...

def calculate_usage_costs(message: Message) -> MessageCostReport: ![3](img/3.png)
    if message.model not in prices :
        # raise at runtime - in case someone ignores type errors
        raise ValueError(
            f"Cost calculation is not supported for {message.model} model."
        )
    price = prices[message.model]
    req_costs = price * count_tokens(message.prompt) / 1000
    res_costs = price * count_tokens(message.response) / 1000
    total_costs = req_costs + res_costs
    return MessageCostReport(
        req_costs=req_costs, res_costs=res_costs, total_costs=total_costs
    )
```

![1](img/1.png)[#co_implementing_type_safe_ai_services_CO2-1]

使用数据类将`Message`和`MessageCost`类装饰为特殊类，用于存储数据。

![2](img/2.png)[#co_implementing_type_safe_ai_services_CO2-2]

将`response`属性的类型指定为`str`或`None`。这与使用`typing`模块中的`Optional[str]`相似。这种新语法在 Python 3.10 及以后的版本中可用，使用新的联合运算符：`|`。

![3](img/3.png)[#co_implementing_type_safe_ai_services_CO2-3]

将`calculate_usage_costs`函数的签名更改为使用预定义的数据类。此更改简化了函数签名。

当您的代码积累代码异味且难以阅读时，应旨在利用数据类。

在示例 4-4 中使用数据类的首要好处是将相关参数分组以简化函数签名。在其他场景中，您可以使用数据类来：

+   消除代码重复
+   缩小代码膨胀（大类或函数）
+   重构数据块（常用在一起的变量）
+   防止意外数据突变
+   促进数据组织
+   促进封装
+   强制数据验证

它们还可以用于实现许多其他代码增强功能。

数据类是改进应用程序中任何位置的数据组织和交换的出色工具。然而，当构建 API 服务时，它们在几个方面并不原生支持：

**自动数据解析**

在赋值时将 ISO 日期时间格式字符串解析为`datetime`对象

**字段验证**

对字段值赋值执行复杂检查，例如检查字符串是否过长

**序列化和反序列化**

在使用不常见类型时，在 JSON 和 Python 数据结构之间进行转换

**字段过滤**

移除未设置或包含`None`值的对象字段

提到的任何限制都不会迫使你放弃使用数据类。当你需要创建具有最小样板代码的数据中心类时，应使用数据类而不是普通类，因为它们会自动生成特殊方法、类型注解和默认值支持，从而减少潜在的错误。然而，如`pydantic`这样的库支持这些功能，如果你不想实现自己的自定义逻辑（例如，序列化`datetime`对象）。

###### 小贴士

FastAPI 也通过 Pydantic 支持数据类，它实现了具有上述功能的数据类的自定义版本，使你可以迁移大量使用数据类的代码库。

让我们来看看 Pydantic，以及它为什么适合构建 GenAI 服务。

# Pydantic 模型

Pydantic 是最广泛使用的支持自定义验证器和序列化器的数据验证库。Pydantic 的核心逻辑由 Python 中的类型注解控制，并可以以 JSON 格式输出数据，从而允许与其他工具无缝集成。

此外，Pydantic V2 的核心数据验证逻辑已被重写为 Rust，以最大化其速度和性能，使其成为 Python 中最快的几个数据验证库之一。因此，Pydantic 对 FastAPI 以及 Python 生态系统中的 8,000 多个其他包（包括 Hugging Face、Django 和 LangChain）产生了重大影响。它是一个经过实战检验的工具包，由主要科技公司使用，截至撰写本文时，每月有 1.41 亿次下载，使其成为在项目中替代数据类的合适候选。

Pydantic 通过其自己的`BaseModel`实现提供了一套广泛的数据验证和处理工具集。Pydantic 模型与数据类有许多相似之处，但在细微之处有所不同。当你创建 Pydantic 模型时，会调用一系列初始化钩子，这些钩子为模型添加数据验证、序列化和 JSON 模式生成功能，这些是纯数据类所缺乏的。

FastAPI 与 Pydantic 紧密集成，并在底层利用其丰富的功能集进行数据处理。类型检查器和代码编辑器也可以像数据类一样读取 Pydantic 模型，以执行检查并提供自动完成功能。

## 如何使用 Pydantic

您可以使用以下命令将 Pydantic 安装到您的项目中：

```py
$ pip install pydantic
```

Pydantic 在其核心实现了一个`BaseModel`，这是定义模型的主要方法。*模型* 简单地是继承自`BaseModel`的类，并使用类型提示定义字段作为注解属性。任何模型都可以用作模式来验证您的数据。

除了分组数据，^(3) Pydantic 模型还允许您指定服务端点的请求和响应需求，并验证来自外部来源的未经验证的传入数据。您还可以使用 Pydantic 模型（以及您将在不久后了解到的验证器）过滤 LLM 输出。

您可以创建自己的 Pydantic 模型，如 示例 4-5 所示。

##### 示例 4-5\. 创建 Pydantic 模型

```py
from typing import Literal
from pydantic import BaseModel

class TextModelRequest(BaseModel): ![1](img/1.png)
    model: Literal["gpt-3.5-turbo", "gpt-4o"]
    prompt: str
    temperature: float = 0.0 ![2](img/2.png)
```

![1](img/1.png)[#co_implementing_type_safe_ai_services_CO3-1]

定义继承自 Pydantic `BaseModel` 的`TextModelRequest`模型。

![2](img/2.png)[#co_implementing_type_safe_ai_services_CO3-2]

如果未提供显式值，则设置默认值。例如，如果初始化时未提供值，则将`temperature`字段设置为`0.0`。

示例 4-5 也展示了如何将您的数据类转换为 Pydantic 模型以利用其众多功能。

## 复合 Pydantic 模型

使用 Pydantic 模型，您可以声明数据 *模式*，这些模式定义了服务操作中支持的数据结构。此外，您还可以使用继承来构建复合模型，如 示例 4-6 所示。

##### 示例 4-6\. 创建 Pydantic 模型

```py
# schemas.py

from datetime import datetime
from typing import Annotated, Literal
from pydantic import BaseModel

class ModelRequest(BaseModel): ![1](img/1.png)
    prompt: str

class ModelResponse(BaseModel): ![2](img/2.png)
    request_id: str
    ip: str | None
    content: str | None
    created_at: datetime = datetime.now()

class TextModelRequest(ModelRequest):
    model: Literal["gpt-3.5-turbo", "gpt-4o"]
    temperature: float = 0.0

class TextModelResponse(ModelResponse):
    tokens: int

ImageSize = Annotated[tuple[int, int], "Width and height of an image in pixels"]

class ImageModelRequest(ModelRequest): ![3](img/3.png)
    model: Literal["tinysd", "sd1.5"]
    output_size: ImageSize
    num_inference_steps: int = 200

class ImageModelResponse(ModelResponse): ![4](img/4.png)
    size: ImageSize
    url: str
```

![1](img/1.png)[#co_implementing_type_safe_ai_services_CO4-1]

定义继承自 Pydantic `BaseModel` 的`ModelRequest`模型。

![2](img/2.png)[#co_implementing_type_safe_ai_services_CO4-2]

定义`ModelResponse`。如果未提供`ip`可选字段的数据，则使用默认值`None`。`content`字段可以是字节（用于图像图像）或字符串（用于文本模型）。

![3](img/3.png)[#co_implementing_type_safe_ai_services_CO4-3]

通过继承`ModelRequest`来定义`TextModelRequest`和`ImageModelRequest`模型。默认情况下，可选的温度字段设置为`0.0`。`ImageModelRequest`模型的`num_inference_steps`字段是可选的，并设置为`200`。这两个模型现在都需要提供提示字符串字段。

![4](img/4.png)[#co_implementing_type_safe_ai_services_CO4-4]

通过继承`ModelResponse`模型来定义`ImageModelResponse`和`TextModelResponse`模型。对于`TextModelResponse`，提供标记的数量；对于`ImageModelResponse`，提供图像的像素大小以及下载图像的远程 URL。

如 示例 4-6 所示的模型，您拥有了定义文本和图像生成端点需求所需的模式。

## 字段约束和验证器

除了支持标准类型外，Pydantic 还附带了一些受限类型，如`EmailStr`、`PositiveInt`、`UUID4`、`AnyHttpUrl`等，这些类型可以在模型初始化时直接进行数据验证，适用于常见的数据格式。Pydantic 的完整类型列表可在[官方文档](https://oreil.ly/xNbXX)中找到。

###### 注意

一些受限类型，如`EmailStr`，可能需要安装依赖包才能正常工作，但可以非常有助于验证常见的数据格式，例如电子邮件。

要在 Pydantic 限制类型之上定义更多自定义和复杂的字段约束，您可以使用 Pydantic 的`Field`函数与`Annotated`类型一起引入验证约束，例如有效的输入范围。

示例 4-7 使用受限类型和`Field`函数替换了 示例 4-6 中的标准类型提示，以根据模型约束为您的端点实现更严格的数据要求。

##### 示例 4-7\. 使用受限字段

```py
# schemas.py

from datetime import datetime
from typing import Annotated, Literal
from uuid import uuid4
from pydantic import BaseModel, Field, HttpUrl, IPvAnyAddress, PositiveInt

class ModelRequest(BaseModel):
    prompt: Annotated[str, Field(min_length=1, max_length=10000)] ![1](img/1.png)

class ModelResponse(BaseModel):
    request_id: Annotated[str, Field(default_factory=lambda: uuid4().hex)] ![2](img/2.png)
    # no defaults set for ip field
    # raise ValidationError if a valid IP address or None is not provided
    ip: Annotated[str, IPvAnyAddress] | None ![3](img/3.png)
    content: Annotated[str | None, Field(min_length=0, max_length=10000)] ![4](img/4.png)
    created_at: datetime = datetime.now()

class TextModelRequest(ModelRequest):
    model: Literal["gpt-3.5-turbo", "gpt-4o"]
    temperature: Annotated[float, Field(ge=0.0, le=1.0, default=0.0)] ![5](img/5.png)

class TextModelResponse(ModelResponse):
    tokens: Annotated[int, Field(ge=0)]

ImageSize = Annotated ![6
    tuple[PositiveInt, PositiveInt], "Width and height of an image in pixels"
]

class ImageModelRequest(ModelRequest):
    model: Literal["tinysd", "sd1.5"]
    output_size: ImageSize ![6](img/6.png)
    num_inference_steps: Annotated[int, Field(ge=0, le=2000)] = 200 ![7](img/7.png)

class ImageModelResponse(ModelResponse):
    size: ImageSize ![6](img/6.png)
    url: Annotated[str, HttpUrl] | None = None ![8](img/8.png)
```

![1](img/1.png)[#co_implementing_type_safe_ai_services_CO5-1]

将`str`标准类型替换为`Field`和`Annotated`以将字符串长度限制在字符范围内。

![2](img/2.png)[#co_implementing_type_safe_ai_services_CO5-2]

通过传递一个可调用的函数到`default_factory`来生成一个新的请求 UUID，该函数将被调用来生成新的 UUID。

![3](img/3.png)[#co_implementing_type_safe_ai_services_CO5-3]

将可选的`ip`字段限制为任何有效的 IPv4 或 IPv6 地址范围。如果客户端的 IP 地址无法确定，`None`也是一个有效的条目。此可选字段没有默认值，因此如果未提供有效的 IP 或`None`，Pydantic 将引发`ValidationError`。

![4](img/4.png)[#co_implementing_type_safe_ai_services_CO5-4]

将`content`字段限制为 10,000 个字符或字节。

![5](img/5.png)[#co_implementing_type_safe_ai_services_CO5-5]
