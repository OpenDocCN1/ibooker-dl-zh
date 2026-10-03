
将温度限制在 `0.0` 和 `1.0` 之间，默认值为 `0.0`。

![6](img/6.png)[#co_implementing_type_safe_ai_services_CO5-6]

重复使用 `Annotated` 限制 `output_size` 字段为正整数，使用 `PositiveInt` 限制类型。`lte` 和 `gte` 关键字分别表示小于等于和大于等于。

![7](img/7.png)[#co_implementing_type_safe_ai_services_CO5-8]

使用 `Field` 函数将 `num_inference_steps` 字段限制在 `0` 到 `2000` 之间，默认值为 `200`。

![8](img/8.png)[#co_implementing_type_safe_ai_services_CO5-10]

将可选的 `url` 字段限制为任何有效的 HTTP 或 HTTPS URL，其中主机名和顶级域名（TLD）是必需的。

在定义了示例 4-7 中的模型后，你现在可以对传入或传出的数据进行验证，以匹配您所需的数据要求。在这种情况下，FastAPI 将利用 Pydantic 在请求运行时数据验证检查失败时自动返回错误响应，如示例 4-8 所示。

### 示例 4-8. 数据验证失败时的 FastAPI 错误响应

```py
$ curl -X 'POST' \
  'http://127.0.0.1:8000/validation/failure' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
 "prompt": "string",
 "model": "gpt-4o",
 "temperature": 0
}'

{
  "detail": [
    {
      "type": "literal_error",
      "loc": [
        "body",
        "model"
      ],
      "msg": "Input should be 'tinyllama' or 'gemma2b'",
      "input": "gpt-4o",
      "ctx": {
        "expected": "'tinyllama' or 'gemma2b'"
      }
    }
  ]
}
```

## 自定义字段和模型验证器

Pydantic 进行数据验证检查的另一个优秀功能是自定义字段验证器。示例 4-9 展示了如何在 `ImageModelRequest` 上实现这两种类型的自定义验证器。

### 示例 4-9. 为 `ImageModelRequest` 实现自定义字段和模型验证器

```py
# schemas.py

from typing import Annotated, Literal
from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    PositiveInt,
    validate_call,
)

ImageSize = Annotated[
    tuple[PositiveInt, PositiveInt], "Width and height of an image in pixels"
]
SupportedModels = Annotated[
    Literal["tinysd", "sd1.5"], "Supported Image Generation Models"
]

@validate_call ![1](img/1.png)
def is_square_image(value: ImageSize) -> ImageSize: ![2](img/2.png)
    if value[0] / value[1] != 1:
        raise ValueError("Only square images are supported")
    if value[0] not in [512, 1024]:
        raise ValueError(f"Invalid output size: {value} - expected 512 or 1024")
    return value

@validate_call ![1](img/1.png)
def is_valid_inference_step(
    num_inference_steps: int, model: SupportedModels
) -> int:
    if model == "tinysd" and num_inference_steps > 2000: ![3](img/3.png)
        raise ValueError(
            "TinySD model cannot have more than 2000 inference steps"
        )
    return num_inference_steps

OutputSize = Annotated[ImageSize, AfterValidator(is_square_image)] ![4](img/4.png)
InferenceSteps = Annotated ![4
    int,
    AfterValidator(
        lambda v, values: is_valid_inference_step(v, values["model"])
    ),
]

class ModelRequest(BaseModel):
    prompt: Annotated[str, Field(min_length=1, max_length=4000)]

class ImageModelRequest(ModelRequest):
    model: SupportedModels
    output_size: OutputSize ![5](img/5.png)
    num_inference_steps: InferenceSteps = 200 ![6](img/6.png)
```

![1](img/#co_implementing_type_safe_ai_services_CO6-1)

除了静态类型检查外，如果向 `is_square_image` 和 `is_valid_inference_step` 函数传递了不正确的参数，则引发运行时验证错误。

![2](img/#co_implementing_type_safe_ai_services_CO6-2)

`tinysd` 模型只能生成特定尺寸的方形图像。请求非方形图像尺寸（除 `1` 以外的纵横比）应引发 `ValueError`。

![3](img/#co_implementing_type_safe_ai_services_CO6-4)

如果用户请求 `tinysd` 模型的许多推理步骤，则引发 `ValueError`。

![4](img/#co_implementing_type_safe_ai_services_CO6-5)

使用 `OutputSize` 和 `InferenceSteps` 的注解模式创建可重用且更易于阅读的验证器。

![5](img/#co_implementing_type_safe_ai_services_CO6-7)

将 `OutputSize` 字段验证器附加到 `output_size` 字段，以在模型初始化后检查不正确的值。

![6](img/#co_implementing_type_safe_ai_services_CO6-8)

将 `InferenceSteps` 验证器附加到 `ImageModelRequest` 模型，以在模型初始化后对模型字段值进行检查。

如示例 4-9 所示，使用自定义字段验证器，您现在可以确信您的图像生成端点将受到用户提供的错误配置的保护。

### 注意

您还可以使用装饰器模式来验证模型字段。可以通过使用 `@field_validator` 或 `@model_validator` 装饰器将特殊方法与模型字段关联起来，以执行条件数据检查。

虽然 `@field_validator` 通过访问单个字段的值来执行检查，但 `@model_validator` 装饰器允许对涉及多个字段的检查。

使用 `after` 验证器，您可以在 Pydantic 完成其解析和验证后执行额外的检查或修改数据。

## 计算字段

与数据类类似，Pydantic 也允许您实现方法来计算来自其他字段派生的字段。

你可以使用 `@computed_field` 装饰器来实现一个计算字段，用于计算令牌和成本的数量，如下所示示例 4-10。

### 示例 4-10. 使用计算字段自动计算令牌总数

```py
# schemas.py

from typing import Annotated
from pydantic import computed_field, Field
from utils import count_tokens

...

class TextModelResponse(ModelResponse):
    model: SupportedModels
    price: Annotated[float, Field(ge=0, default=0.01)]
    temperature: Annotated[float, Field(ge=0.0, le=1.0, default=0.0)]

    @property
    @computed_field
    def tokens(self) -> int:
        return count_tokens(self.content)

    @property
    @computed_field
    def cost(self) -> float:
        return self.price * self.tokens
```

计算字段对于封装 Pydantic 模型中的任何字段计算逻辑非常有用，以保持代码整洁。请注意，计算字段仅在将 Pydantic 模型转换为字典时通过 `.model_dump()` 或在 FastAPI API 处理程序返回响应时的序列化时才可访问。

## 模型导出和序列化

由于 Pydantic 模型可以序列化为 JSON，因此你在示例 4-7 中定义的模型也可以被导出到（或从）JSON 字符串或 Python 字典中，同时保持任何复合模式，如下所示示例 4-11。

### 示例 4-11. 导出和序列化 `TextModelResponse` 模型

```py
>> response = TextModelResponse(content="FastAPI Generative AI Service", ip=None)
>> response.model_dump(exclude_none=True)
{'content': 'FastAPI Generative AI Service',
 'cost': 0.06,
 'created_at': datetime.datetime(2024, 3, 7, 20, 42, 38, 729410),
 'price': 0.01,
 'request_id': 'a3f18d85dcb442baa887a505ae8d2cd7',
 'tokens': 6}

>> response.model_dump_json(exclude_unset=True)
'{"ip":null,"content":"FastAPI Generative AI Service","tokens":6,"cost":0.06}'
```

## 使用 Pydantic 解析环境变量

除了 `BaseModel`，Pydantic 还实现了一个 `Base` 类，用于从文件中解析设置和秘密。此功能由一个可选的 Pydantic 包 `pydantic-settings` 提供，你可以将其作为依赖项安装：

```py
$ pip install pydantic-settings
```

`BaseSettings` 类提供了可选的 Pydantic 功能，用于从环境变量或秘密文件中加载设置或配置类。使用此功能，设置值可以在代码中设置或由环境变量覆盖。

在生产环境中，这很有用，因为你不想在代码或容器环境中暴露秘密。

当你创建一个继承自 `BaseSettings` 的模型时，模型初始化器将尝试使用提供的默认值设置每个字段的值。如果失败，初始化器将然后从环境变量中读取任何未设置字段的值。

给定一个 dotenv 环境文件 (ENV)：

```py
APP_SECRET=asdlkajdlkajdklaslkldjkasldjkasdjaslk
DATABASE_URL=postgres://sa:password@localhost:5432/cms
CORS_WHITELIST=["https://xyz.azurewebsites.net","http://localhost:3000"]
```

ENV 是一个环境变量文件，可以使用 shell 脚本语法来表示键值对。

示例 4-12 展示了使用 `BaseSettings` 解析环境变量的实际操作。

### 示例 4-12. 使用 Pydantic `BaseSettings` 解析环境变量

```py
# settings.py

from typing import Annotated
from pydantic import Field, HttpUrl, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings): ![1](img/1.png)
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8" ![2](img/2.png)
    )

    port: Annotated[int, Field(default=8000)]
    app_secret: Annotated[str, Field(min_length=32)]
    pg_dsn: Annotated[
        PostgresDsn,
        Field(
            alias="DATABASE_URL",
            default="postgres://user:pass@localhost:5432/database",
        ),
    ] ![3](img/3.png)
    cors_whitelist_domains: Annotated[
        set[HttpUrl],
        Field(alias="CORS_WHITELIST", default=["http://localhost:3000"]),
    ] ![4](img/4.png)

settings = AppSettings()
print(settings.model_dump()) ![5](img/5.png)
"""
{'port': 8000
 'app_secret': 'asdlkajdlkajdklaslkldjkasldjkasdjaslk',
 'pg_dsn': MultiHostUrl('postgres://sa:password@localhost:5432/cms'),
 'cors_whitelist_domains': {Url('http://localhost:3000/'),
                            Url('https://xyz.azurewebsites.net/')},
}
"""
```

![1](img/1.png)[#co_implementing_type_safe_ai_services_CO7-1]

声明从 `pydantic_settings` 包继承的 `AppSettings` 类。

![2](img/2.png)[#co_implementing_type_safe_ai_services_CO7-2]

将 `AppSettings` 配置为以 `UTF-8` 编码从项目根目录的 ENV 文件中读取环境变量。默认情况下，snake_case 字段名称将映射到环境变量名称的大写版本。例如，`app_secret` 变为 `APP_SECRET`。

![3](img/3.png)[#co_implementing_type_safe_ai_services_CO7-3]

验证 `DATABASE_URL` 环境变量是否具有有效的 Postgres 连接字符串格式。如果没有提供，则设置默认值。

![4](img/4.png)[#co_implementing_type_safe_ai_services_CO7-4]

检查 `CORS_WHITELIST` 环境变量是否包含具有主机名和顶级域的唯一 URL 列表。如果没有提供，则将默认值设置为包含单个值 `http://localhost:3000` 的集合。

![5](img/#co_implementing_type_safe_ai_services_CO7-5)

我们可以通过打印模型的转储来检查 `AppSettings` 类是否正常工作。

### 注意

当使用 `_env_file` 参数时，你可以切换环境文件：

```py
test_settings = AppSettings(_env_file="test.env")
```

## FastAPI 中的数据类或 Pydantic 模型
