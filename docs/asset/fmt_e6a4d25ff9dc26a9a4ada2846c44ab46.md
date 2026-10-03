
![计算机屏幕截图  自动生成的描述](img/CH04_F05_Kardell.png)

##### 图 4.5 软件测试类别

我们想确保我们的 API 将是响应的，看到性能瓶颈可能在哪里，并确定系统的容量以供未来规划。为此，我们可以参考一个名为 `Locust` 的框架。`Locust` 允许我们使用我们正在发展的 Python 技能来编写用于负载测试的测试脚本。

我们可以通过告诉 ChatGPT “为 `http://localhost:8000/files` 端点创建一个 `Locust` 测试示例” 来轻松开始，这提供了一些易于遵循的步骤。首先，我们想要使用 `pip install locust` 安装 `Locust`。同时，我们创建一个 `locustfile.py` 文件，如下所示。

##### 列表 4.17 简单的 Locust 文件

```py
from locust import HttpUser, task, between

class FileUser(HttpUser):
    wait_time = between(1, 2.5)

    host = "http://localhost:8000"

    @task
    def get_files(self):
        self.client.get("/files")
```

在 `locustfile.py` 目录下运行命令 `locust`。这将提示你打开浏览器到默认位置 `http://localhost:8089`。在这里，我们将输入用户数量、孵化率、主机以及它应该运行多长时间（我们点击了“高级选项”）以获取（见图 4.6）。

![计算机屏幕截图，描述自动生成](img/CH04_F06_Kardell.png)

##### 图 4.6 使用 Locust 启动新的测试

从那里，我们可以看到测试运行并获得关于我们的 API 性能的指标。显然，在这种情况下，我们正在处理硬编码的值，所以我们将在稍后有机会回来探索我们的 API 的负载测试。然而，我们现在可以继续使用 `Locust` 和负载测试来获得一些经验。我们能否为其他端点添加测试？如果我们需要提醒它们是什么或者它们应该返回什么，不要忘记我们的实时文档在 `http://localhost:8000/docs`。通过添加额外的测试，我们的代码应该看起来像下面的列表。

##### 列表 4.18 Locust 测试

```py
from locust import HttpUser, task, between

class FileUser(HttpUser): #1
    wait_time = between(1, 2.5) #2

    host = "http://localhost:8000" #3

    @task #4
    def get_files(self): #5
        self.client.get("/files")  #6

    @task
    def get_file(self):
        self.client.get("/files/1")

    @task
    def post_file(self):
        self.client.post("/files", json={"file_id": 5})

    @task
    def get_records(self):
        self.client.get("/files/1/records")

    @task
    def get_record(self):
        self.client.get("/files/1/records/1")
```

#1 创建一个继承自 `HttpUser` 的 `FileUser` 类

#2 在请求之间设置一个 1 到 2.5 秒的随机等待时间，以模拟用户在动作之间的暂停

#3 定义 API 请求将发送到的主机

#4 任务装饰器定义了一个用户模拟的动作。

#5 对 `/files` URI

现在，我们可以重新运行 `Locust` 来查看它正在击中各种端点。虽然我们的端点目前是硬编码的，但它们将返回使用的路径参数。所以，为了使事情更有趣，让我们为这些测试中的每一个添加一些代码，使得参数是唯一的，这样我们就可以在 `Locust` 控制台中看到更多的端点被击中。花几分钟时间看看你是否可以用随机数替换硬编码的 `file_ids` 和 `record_ids`，然后重新运行测试。下面的列表显示了我们所提出的示例。

##### 列表 4.19 Locust

```py
import random
from locust import HttpUser, task, between

class FileUser(HttpUser):
    wait_time = between(1, 2.5)
    host = "http://localhost:8000"

    @task
    def get_files(self):
        self.client.get("/files")

    @task
    def get_file(self):
        file_id = random.randrange(1, 10) #1
        self.client.get(f"/files/{file_id}")

    @task
    def post_file(self):
        file_id = random.randrange(1, 10)
        # Sending POST request to the "/files" endpoint
        self.client.post("/files", json={"file_id": file_id})

    @task
    def get_records(self):
        file_id = random.randrange(1, 10)
        self.client.get(f"/files/{file_id}/records")

    @task
    def get_record(self):
        file_id = random.randrange(1, 10)
        record_id = random.randrange(1, 20)
        self.client.get(f"/files/{file_id}/records/{record_id}")
```

#1 在 1 到 9 的范围内生成一个随机的 `file_id`

重新运行我们的测试后，我们可以看到访问端点的更多多样性。在这个时候，我们可能已经注意到了在 `Locust` 控制台中执行测试时出现的问题。如果没有，请查看 `Locust` 的“失败”选项卡。你应该会看到 `/files` 上 `POST` 方法的多个失败，错误信息为 `HTTPError('405 Client Error: Method Not Allowed for url: /files')`。看起来我们在我们的 API 代码中遗漏了一些东西。但在检查它之前，让我们看看 `Locust` 测试的另一个大问题——没有 `assert` 语句。

与我们所有的其他测试一样，我们想要确保我们进行了一些 `assert` 语句。让我们先确保每次都得到一个有效的响应。在继续阅读之前，不妨试一试。

##### 列表 4.20 Locust 测试

```py
…  #1
@task #2
def get_files(self): #3
   response = self.client.get("/files")  #3
   assert response.status_code == 200,  #3
      f"Unexpected status code: {response.status_code}"  #3

@task
def get_file(self):
   response = self.client.get(f"/files/{file_id}")
   assert response.status_code == 200, 
      f"Unexpected status code: {response.status_code}"

@task
def post_file(self):
   response = self.client.post("/files", json={"file_id": file_id})
   assert response.status_code == 200, 
      f"Unexpected status code: {response.status_code}"

@task
def get_records(self):
   file_id = random.randrange(1, 10) #4
   response = self.client.get(f"/files/{file_id}/records")
   assert response.status_code == 200, 
      f"Unexpected status code: {response.status_code}"

@task
def get_record(self):
   file_id = random.randrange(1, 10) #5
   record_id = random.randrange(1, 20) 
   response = self.client.get(f"/files/{file_id}/records/{record_id}")
   assert response.status_code == 200, 
      f"Unexpected status code: {response.status_code}"
```

#1 列表 4.19 中的先前代码设置

#2 使用 `Locust` 任务装饰器定义 `Locust` 应该执行的内容

#3 定义了一个 `get_files` 方法，它对 `/files` URI 执行 `GET` 请求并检查 HTTP 状态码为 200

#4 使用 1 到 9 范围内的随机 `file_id` 发起请求

#5 使用 1 到 9 范围内的随机 `file_id` 和 1 到 19 范围内的随机 `record_id` 发起请求

现在我们有一些 `assert` 语句，我们可以在异常标签页中看到 `Unexpected status code: 405`，除了失败标签页。让我们检查 API 容器的日志，看看有什么。日志显示两个有趣的问题：

```py
"GET /files/3 HTTP/1.1" 200 OK
"POST /files HTTP/1.1" 405 Method Not Allowed
"POST /files HTTP/1.1" 405 Method Not Allowed
"GET /files/6/records HTTP/1.1" 307 Temporary Redirect
"GET /files HTTP/1.1" 200 OK
```

正如我们所见，`POST` 请求得到了 `405 Method Not Allowed` 的响应，我们还看到了一个 `307 Temporary Redirect`。当我们回顾定义端点的代码时，我们发现有些端点后面有斜杠，而有些则没有：

```py
@router.get("/files")
@router.get("/files/{file_id}")
@router.post("/files/", status_code=status.HTTP_201_CREATED)
@router.get("/files/{file_id}/records/")
@router.get("/files/{file_id}/records/{record_id}")
```

所有的 `Locust` 请求都没有在请求中包含尾部斜杠，所以让我们确保服务器和客户端都是一致的。在我们的例子中，我们将从服务器部分移除尾部斜杠。重新构建我们的容器并重新运行测试，给我们带来了更一致的经历，没有错误日志或 `Locust` 中的失败。我们确实看到 `Locust` 中仍然有异常。这是因为我们编写了所有的 `assert` 语句，期望返回 `200` 状态码。然而，当我们参考文档（或查看代码）时，我们被提醒我们的 `POST` 请求响应的是 `201` 而不是 `200`。处理完这个问题后，我们看到测试正常运行，有审查失败、异常以及如图 4.7 所示的相关分析。

![一个图表的截图  自动生成的描述](img/CH04_F07_Kardell.png)

##### 图 4.7 连续的 Locust 运行视图

## 4.9 改进我们的代码

现在我们有了单元测试和负载测试，你应该可以在代码中工作以进行改进感到舒适。尽管代码很简单，我们仍然可以利用 `FastAPI` 的一些好处。本节展示了版本控制、使用前缀、文档和验证的好处。

### 4.9.1 使用前缀和版本控制

你可能已经注意到，我们路由器中的每个端点都包括 `/files`。虽然这没问题，但我们还想包括一个版本号，并将端点放在 URL 中的 `api` 端点下。我们希望最终得到类似 `/api/v1/files` 的初始标准。使用 `/api` 有助于将我们的 API 组织在一个区域。这可以帮助进行 Web 服务器配置、防火墙规则、监控等。`v1` 表示 API 版本，我们可以用它来支持多个 API 版本。记住，一旦最终用户开始使用 API，就不应该对 API 进行破坏性更改，这会迫使他们升级他们的应用程序。在这些情况下，你应该在 `/v2` 下提供新的 API，并在可能的情况下逐步淘汰旧版本。记住，`/api/v1` 只是一个建议。你可以命名端点，并根据自己的需求使用任何版本控制方案。

我们通过使用 `APIRouter` 中可用的 `prefix` 参数重构我们的代码，为端点添加额外的 `/api/v1`。我们更新了 `files.py`：

```py
router = APIRouter(prefix="/api/v1/files")
```

然后，我们从我们定义的每个端点中移除 `/files`。表 4.3 显示了差异。

##### 表 4.3 使用 `prefix` 参数对路由器的影响

| 之前 | 之后 |
| --- | --- |
| `@router.get("/files")` | `@router.get("")` |
| `@router.get("/files /{file_id}")` | `@router.get("/{file_id}")` |
| `@router.post("/files ", status_code=status.HTTP_201_CREATED)` | `@router.post("", status_code=status.HTTP_201_CREATED)` |
| `@router.get("/files /{file_id}/records")` | `@router.get("/{file_id}/records")` |
| `@router.get("/files /{file_id}/records/{record_id}")` | `@router.get("/{file_id}/records/{record_id}")` |

我们可以看到，通过使用前缀，我们现在在每个端点中需要输入的字符更少了。我们还需要重新运行我们的单元测试，看看是否仍然正常工作，正如我们可能预料的那样，我们的测试都失败了，显示 `404` 错误，因为它们不再指向正确的 URL。我们需要继续更新我们的单元测试，包括 `/api/v1`，并重新运行它们。

### 4.9.2 添加 Pydantic

`Pydantic` 是一个广泛使用的数据验证库，它捆绑在 `FastAPI` 中。我们将在这里使用它来摆脱我们的硬编码字符串（尽管我们仍然会返回硬编码的值）。通过添加 `Pydantic`，我们使我们的代码更容易在未来扩展，并能够利用其一些功能进行验证和文档化。

之后，我们将致力于构建模拟实际 ACH 文件和记录的对象。目前，让我们看看 `Pydantic` 将如何增强我们当前的 API。我们的第一个端点在 `/api/v1/files/{file_id}` 上执行 `GET` 操作，我们之前返回的是一个 JSON 字符串。它看起来像这样（带有新定义的前缀）。

##### 列表 4.21 我们之前的硬编码函数

```py
@router.get("/{file_id}")
async def read_file(file_id: str):
    return {"file_id": file_id}
```

现在，我们创建一个 `AchFile` 类，它将模拟我们在上一个列表中返回的硬编码值。

##### 列表 4.22 我们第一个 `Pydantic` 类

```py
from pydantic import BaseModel, Field

class AchFile(BaseModel):
    file_id: str
```

在这个简单的类设置到位后，我们可以更新路由器以返回该类的实例。

##### 列表 4.23 更新我们的路由以使用我们新创建的 `Pydantic` 类

```py
…
from ach.AchFile import AchFile
…
@router.get("/{file_id}")
async def read_file(file_id: str) -> AchFile:
    return AchFile(file_id=file_id)
```

我们的单元测试仍然应该通过，所以我们没有破坏任何东西。如果我们开始在 Docker 中启动，我们会遇到一个 `ModuleNotFoundError: No module named 'ach'` 的错误。这是因为 Python 不知道在哪里找到我们的类，甚至有可能我们忘记复制它了！在启动之前，请向您的 Dockerfile 中添加另一个 `COPY` 语句并设置 `PYTHONPATH`。这应该能让我们的代码正常运行。

##### 列表 4.24 更新我们的 Dockerfile
