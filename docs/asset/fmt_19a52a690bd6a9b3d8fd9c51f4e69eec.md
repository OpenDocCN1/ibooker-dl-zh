
在表就绪后，我们可以创建编写这些审计记录所需的代码。

### 9.3.2 为审计进行的 TDD 测试

我们首先为 API 创建单元测试，然后添加必要的代码以使测试通过。我们也应该考虑 BDD 风格的测试吗？这个答案取决于业务如何选择使用审计日志。如果日志主要作为开发者使用的东西，那么业务可能不需要与我们合作来开发测试。相反，如果业务计划使用审计日志来拉取活动以帮助确定指标，我们可能需要与他们合作来开发测试。我们可以很容易地看到业务提出那种类型的要求，并需要更新我们的测试以读取类似`WHEN I REQUEST A LIST OF FILES THEN I SHOULD SEE AN AUDIT LOG RECORD INDICATING FILES WERE SHOWN`的内容。现在，我们将坚持我们的标准单元测试。单元测试定义了一个记录然后检索它，以便我们可以比较字段。我们目前有一个`user_id`字段，但由于我们目前没有跟踪用户，我们将留空。

##### 列表 9.21 审计日志的单元测试

```py
class TestAuditLog:
    @pytest.fixture(autouse=True) #1
    def setup_teardown_method(self):  #1
        SqlUtils.truncate_all()  #1
        yield #1

    def test_insert_record(self): 
        log_record = AuditLogRecord( #2
            ip_address="192.168.1.0/24",  #2
            user_agent="Mozilla/5.0",  #2
            http_request="GET",  #2
            http_response=200,  #2
            url="/api/v1/files",  #2
            message="Test audit log insert",  #2
        )  #2
        logger = AuditLog() #3
        audit_record_id = logger.log_record(log_record)  #3
        retrieved_record =   #3
➥logger.get_log_record(audit_record_id)  #3
        excluded_fields = {"audit_log_id", "created_at"} #4
        assert SqlUtils.get_row_count_of_1("audit_log") is True
➥, "Expected 1 record" 
        assert retrieved_record.dict(exclude=excluded_fields) 
➥ == #D
     log_record.dict( 
            exclude=excluded_fields
        ), f"Expected {log_record}, 
➥ but got {retrieved_record}" 
```

#1 每次清除数据库

#2 创建一个虚拟记录

#3 使用`AuditLog`类创建和检索记录

#4 从字典中排除指定的字段，因为这些字段由数据库填充且不影响比较；然后断言记录匹配

单元测试需要一个`AuditLogRecord`对象，这是一个标准的 Pydantic 模型，用于存储我们的数据。在这个模型中，我们使用了两种新的数据类型——`IPvAnyNetwork`和`AnyUrl`（两者都可作为从 Pydantic 导入的导入项）。`IPvAnyNetwork`对于表示我们在表中定义的`INET`字段效果很好，并且对于 IP 地址提供了预期的验证，无论它们是单个地址（例如，`127.0.0.1`）还是一系列地址（例如，`192.168.1.0/24`）。当然，我们不会期望在这个字段中看到地址范围。`AnyUrl`字段类似地提供了对 URL 的验证，使我们免于编写正则表达式来验证字段。列表 9.22 显示了我们的`AuditLogRecord`类，为了简洁起见，删除了一些不太有趣的字段。

##### 列表 9.22 `AuditLogRecord`类

```py
class AuditLogRecord(BaseModel):
…
    ip_address: Optional[IPvAnyNetwork] = Field( #1
        None,
        title="IP Address",
        description="IP address from which the request originated.",
    )
…
    url: Optional[AnyUrl] = Field( #2
        None, 
        title="URL", 
        description="URL accessed by the request.", max_length=255
    )
…
    @field_validator("url", mode="before") #3
    @classmethod #3
    def convert_url_to_str(cls, v: AnyUrl) -> str:  #3
        return str(v) 
```

#1 将`ip_address`定义为`IPvAnyNetwork`类型，这允许 Pydantic 验证 IP 地址和网络

#2 将 URL 定义为`AnyUrl`类型，这允许 Pydantic 验证 URL 是否格式正确

#3 定义了一个字段验证器，以便将 URL 转换为字符串以存储在数据库中

如列表 9.22 所示，我们也使用字段验证器将 URL 字段作为字符串返回。这是因为当我们尝试将其插入数据库时，我们会收到错误，因为 URL 数据类型与字符串不兼容。我们可以在负责插入记录的审计日志类中明确编码它，但这种方法可以很好地封装一切，其他开发者不必担心它，这有助于减少认知负荷。

能够写入记录并使这个单元测试通过的最后一步是生成一个实际要插入数据库的类。在以下列表中，我们创建了一个包含`log_record`和`get_log_record`方法的`AuditLog`类。对于其他数据库记录，我们特别使用了`Sql`作为名称的一部分。在这种情况下，我们并不一定想将其绑定到数据库。

##### 列表 9.23 `AuditLog`类

```py
class AuditLog:
    @staticmethod
    def log_record(log_record: AuditLogRecord):
        with get_db_connection(row_factory=dict_row) as conn: #1
            log_record_dict = log_record.dict()  #1
            log_record_dict["url"] = str(log_record_dict["url"]) 
            result = conn.execute( #2
                """  #2
           INSERT INTO audit_log (user_id, ip_address,  #2
                       user_agent, http_request, http_response,  #2
  url, message)  #2
           VALUES (%(user_id)s,  #2
                %(ip_address)s,  #2
                %(user_agent)s,  #2
                %(http_request)s,  #2
                %(http_response)s,  #2
                %(url)s,  #2
                %(message)s #2
                )  #2
           RETURNING audit_log_id #2
            """,  #2
                log_record_dict,  #2
            )  #2
 #2
        return result.fetchone()["audit_log_id"] #3

    @staticmethod
    def get_log_record(audit_log_id: str):
        with get_db_connection(row_factory=
➥class_row(AuditLogRecord)) as conn:
            result = conn.execute(
                """
                SELECT *
                FROM audit_log
                WHERE audit_log_id = %(audit_log_id)s
                """,
                {"audit_log_id": audit_log_id},
            )

        return result.fetchone()
```

#1 定义了一个要使用的字典

#2 定义了要插入的字段；我们将返回从数据库中获得的`audit_log_id`。

#3 因为使用了`dict_row`作为`row_factory`，我们可以直接引用`audit_log_id`。

在此基础上，我们的单元测试通过了，我们可以在任何想要的地方添加审计记录。现在，让我们看看如何将我们所学的内容与 FastAPI 框架集成。

### 9.3.3 审计逻辑

向我们的应用程序添加审计的最明显方式是通过调用`log_record`方法并传递一个`AuditLogRecord`，正如我们在单元测试中所看到的。我们当然可以取我们的现有端点之一并添加调用，最终得到以下类似列表的内容，这允许我们写入消息，但并没有捕获我们为定义的字段所定义的一些其他可能有用的信息。

##### 列表 9.24 添加我们的 API 记录的一种方式

```py
async def read_files() -> list[AchFilesResponse]:
    AuditLog().log_record( #1
        AuditLogRecord(  #1
            message="User retrieved ACH files -test"  #1
        )  #1
    )  #1
    return AchFileSql().get_files_response()
```

#1 在 API 执行时手动调用`log_record`方法，但我们遗漏了许多有趣的字段

我们可以通过包括请求字段来增强前面的列表。然后我们可以从请求对象中提取额外的信息。

##### 列表 9.25 在我们的 API 中包含请求信息

```py
async def read_files(request: Request) -> #1
➥ list[AchFilesResponse]: 
    AuditLog().log_record(
        AuditLogRecord(
            ip_address=request.client.host, #2
            message="User retrieved ACH files -test"
        )
    )
    return AchFileSql().get_files_response()
```

#1 更新方法以包括请求对象

#2 我们现在可以访问一些有趣的字段

虽然这样可行，但它要求开发者添加比我们更愿意的更多样板逻辑。由于我们主要对审计 API 的使用感兴趣，我们可以采取另一种方法。

## 9.4 使用中间件进行日志记录

为了记录 API 请求，我们可以考虑添加一个具有唯一日志记录请求目的的中间件函数。我们可以使用请求和响应，这使我们能够捕获响应代码。虽然我们也可以考虑配置我们的 Nginx 日志以捕获生产环境中的信息，但了解如何在 FastAPI 中实现这一点也是有帮助的，因为它可以广泛应用于其他需求。我们使用`@app.middleware`注解来编写我们的日志记录。

##### 列表 9.26 日志记录的中间件层

```py
@app.middleware("http") #1
async def log_requests(request: Request, call_next): #2
    response = await call_next(request) #3
    log_message = getattr(request.state, 'log_message', #4
                  "Default audit log message") 
    log_record = AuditLogRecord( #5
        ip_address=request.client.host,  #5
        user_agent=request.headers.get #5
➥('user-agent', 'unknown'),  #5
        http_request=request.method,  #5
 http_response=response.status_code, #5
 url=str(request.url), #5
 message=log_message, #5
  )
    logger = AuditLog() #6
    logger.log_record(log_record) 
    return response #7
```

#1 定义了一个中间件组件，该组件为每个 HTTP 请求执行

#2 定义一个日志请求的方法

#3 将请求传递给下一层

#4 获取在`request.state`中定义的`log_message`，如果不存在则默认

#5 使用我们期望的请求字段构建一个`AuditLogRecord`

#6 编写日志消息

#7 返回响应

插入该中间件层并使其执行日志记录相对容易。然后我们可以记录所有传入的调用，这消除了样板代码的需求。此外，它使我们的代码更简洁，让开发者专注于他们的代码。唯一的缺点是添加消息需要更多的努力。当然，消息将反映`REST API`调用正在做什么。因此，从某种意义上说，记录消息是多余的（并且是空间的浪费）。例如，当日志显示他们在`/api/v1/files`上执行`GET`操作时，我们是否需要知道有人查看 ACH 文件？当然，我们可能需要出于业务原因这样做，因为当被审查时，对于不熟悉数据的人来说，解释消息更容易。与其浪费存储空间，我们可能决定有一个实用方法，可以在提取期间执行该解释并插入适当的消息。用例可能各不相同，我们只想说明一些替代方案。在这种情况下，我们确实想在数据库表中存储文本，我们将使用装饰器来完成这项工作。以下列表显示了如何使用我们定义的`log_message`注解在请求状态中包含消息。

##### 列表 9.27 `log_message`注解

```py
def log_message(message: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request: Request = kwargs.get("request", None)
            if request: #1
                request.state.log_message = message
            return await func(*args, **kwargs)

        return wrapper

    return decorator
```

#1 如果有请求，它将保存传递的消息。

最后，我们只需要对 API 进行一些小的更改。通过使用我们新的注解，我们可以自定义写入数据库的消息。以下列表显示了我们对 API 调用的最终更改。

##### 列表 9.28 日志的最终更改

```py
@log_message("Retrieving ACH files") #1
async def read_files(request: Request) -> #2
➥ list[AchFilesResponse]: 
    return AchFileSql().get_files_response()
```

#1 为我们的每个 API 调用添加一个独特的消息

#2 确保请求对象是方法的一部分

添加这部分逻辑就完成了我们的 API 消息记录。现在，任何我们用`@log_message`注解的 API 都会写入日志消息。这只是追踪和审计应用程序的开始。通过探索诸如 OpenTelemetry、OpenTracing 或 Prometheus 等项目，您可以极大地扩展在应用程序中收集数据的能力，仅举几个例子。

## 9.5 查看审计日志

现在我们有了应用程序的日志 API 请求，让我们添加在应用程序中查看日志的能力。这个页面将像第八章中的异常页面一样工作。我们需要检索审计记录并将它们放入 DataGrid 中以便查看。当然，我们可以根据需要扩展功能，但到目前为止，仅仅返回数据就是我们需要完成的任务。

### 9.5.1 创建页面
