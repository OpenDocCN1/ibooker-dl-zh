
# 第八章. 认证和授权

到目前为止，您已经构建了可以与数据库交互、流式传输模型响应和处理并发用户的 GenAI 服务。

您的服务现在已启动并运行，但由于它们没有受到攻击者或恶意用户的保护，部署到生产环境可能会遇到问题。

在本章中，您将学习如何通过认证层来保护您的服务，并实现授权守卫以保护敏感资源免受非特权用户的侵害。

为了实现这一点，我们将探索各种认证和授权模式，然后实现`JWT`和基于身份的认证以及基于角色的访问控制。

## 认证和授权

在讨论认证方法之前，让我们简要澄清认证和授权是两个不同的概念，它们经常被错误地互换使用。

根据`OWASP`的定义：^(1)

> *认证*是通过确定一个或多个认证者（如密码、指纹或安全令牌）的有效性来验证个人、实体或网站是否是其声明的身份的过程。

另一方面，美国国家标准与技术研究院（`NIST`）将授权定义为：

> 一种验证请求的操作或服务是否被特定实体批准的过程。

虽然认证是关于验证身份，但授权侧重于验证身份访问或修改资源的权限。

###### 小贴士

一个可能有助于阐明这种区别的类比是在机场通过护照控制。认证就像在移民局出示护照，而授权就像拥有进入一个国家的正确签证，一旦进入，它将指定您的停留时间和允许的活动。

在本章后面讨论授权之前，让我们更详细地讨论认证方法。

## 认证方法

您可以在您的 GenAI 服务中实施几种认证机制，通过身份验证来确保它们的安全。

根据您的安全要求、应用环境、预算和项目时间表，您可能决定采用以下一个或多个认证机制：

- **基本认证**：要求使用用户名和密码等凭证来验证身份。
- **JSON Web Tokens (JWT)**：要求使用*访问令牌*来验证身份。您可以将访问令牌想象成电影院的门票，它规定了您是否可以进入放映厅以及您将访问哪个放映厅以及您将坐在哪里。
- **OAuth**：使用*OAuth2*标准通过身份提供者验证身份。
- **基于密钥**：使用一对私钥和公钥来验证身份。与令牌不同，授权服务器向客户端颁发一个公钥，并存储一个链接的私钥副本，它可以在以后用于验证.^(2)

图 8-1 更详细地显示了上述认证方法的数据流。

![bgai 0801](img/bgai_0801.png)

###### 图 8-1. 认证方法

虽然了解认证机制，但在解决您的安全需求时，选择采用的方法仍然可能具有挑战性。为了帮助选择任务，表 8-1 比较了上述认证方法。

表 8-1. 认证方法比较

| 类型 | 优点 | 缺点 | 用例 |
| --- | --- | --- | --- |
| 基本认证 | + 简单性<br>+ 快速实现<br>+ 易于理解 | 以明文发送凭证 | + 原型<br>+ 内部或非生产环境 |
| 令牌 | + 可扩展性<br>+ 解耦有助于实现微服务架构<br>+ 令牌可以签名和加密以提高安全性<br>+ 高度可定制<br>+ 自包含减少数据库往返次数<br>+ 可以通过 HTTP 头传递 | + 需要不断重新生成短期令牌<br>+ 客户端令牌存储的复杂性<br>+ 令牌可能变得很大，消耗过多的带宽<br>+ 无状态令牌可能使多步骤应用难以实现<br>+ 客户端配置错误可能使令牌受到损害 | + 单页和移动应用<br>+ 需要自定义认证流程的应用<br>+ REST API |
| OAuth | + 将认证委托给外部提供者<br>+ 基于标准（OAuth2）并在企业场景中得到实战检验<br>+ 代表用户访问外部资源 | + 难以理解和实现<br>+ 每个身份提供者可能对 OAuth 流程的实现略有不同 | + 需要从 GitHub、Google 或 Microsoft 等外部身份提供者获取用户数据的应用<br>+ 需要与自己的身份提供者（们）进行 SSO 的企业应用 |
| 基于密钥 | + 与安全壳（SSH）访问类似的认证机制 | + 管理和保持私钥安全可能很复杂<br>+ 被泄露的密钥可能造成安全风险<br>+ 可扩展性问题 | + 小型应用<br>+ 内部环境中的应用 |

您现在应该对选择适当的认证机制充满信心。在下一节中，您将为您的人工智能实现基本、JWT 和 OAuth 认证，以全面了解其底层组件及其交互。

### 基本认证

在基本认证中，客户端在从服务器请求访问资源时提供用户名和密码。这是最简单的技术，因为它不需要实现 cookies、会话标识符或任何登录表单。由于其简单性，基本认证非常适合沙盒环境和原型设计。然而，避免在生产环境中使用它，因为它在每次请求中都以明文形式传输用户名和密码，使其非常容易受到拦截攻击。

要通过基本身份验证执行认证请求，您必须添加一个值为`Basic <credentials>`的`Authorization`头，以便服务器能够成功认证它。`<credentials>`值必须是用户名和密码通过单个冒号连接的*Base64*编码（即`base64.encode(ali:secretpassword)`）。

在 FastAPI 中，您可以使用基本身份验证保护端点，如示例 8-1 所示。

##### 示例 8-1. 在 FastAPI 中实现基本身份验证

```py
import secrets
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

app = FastAPI()
security = HTTPBasic() ![1](img/1.png)
username_bytes = b"ali"
password_bytes = b"secretpassword"

def authenticate_user(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)]
) -> str:
    is_correct_username = secrets.compare_digest(
        credentials.username.encode("UTF-8"), username_bytes ![2](img/2.png)
    )
    is_correct_password = secrets.compare_digest(
        credentials.password.encode("UTF-8"), password_bytes ![2](img/2.png)
    )
    if not (is_correct_username and is_correct_password):
        raise HTTPException( ![3](img/3.png)
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

AuthenticatedUserDep = Annotated[str, Depends(authenticate_user)] ![4](img/4.png)

@app.get("/users/me")
def get_current_user_controller(username: AuthenticatedUserDep): ![4](img/4.png)
    return {"message": f"Current user is {username}"}
```

![1](img/#co_authentication_and_authorization_CO1-1)

FastAPI 实现了多个 HTTP 安全机制，包括可以利用 FastAPI 依赖注入系统的`HTTPBasic`。

![2](img/#co_authentication_and_authorization_CO1-2)

使用内置的`secrets`库比较提供的用户名和密码与服务器上的值。使用`secrets.compare_digest()`确保检查操作的持续时间保持一致，无论输入是什么，以避免*时间攻击*。^(3)

注意，`secrets.compare_digest()`只能接受包含 ASCII 字符（即仅英文字符）的字节或字符串输入。要处理其他字符，您需要先将输入编码为`UTF-8`字节，然后再执行凭证检查。

![3](img/#co_authentication_and_authorization_CO1-4)

返回符合安全标准的标准化授权`HTTPException`，以便浏览器能够再次向用户显示登录提示。异常消息必须是通用的，以避免向攻击者泄露任何敏感信息，例如用户账户的存在。

![4](img/#co_authentication_and_authorization_CO1-5)

使用`HTTPBasic`与`Depends()`一起返回包含提供的用户名的`HTTPBasicCredentials`对象。

将安全依赖注入到任何 FastAPI 端点将使用实现的身份验证来保护它。你现在可以通过访问`/docs`页面并向`/users/me`端点发送请求来亲自体验这一点。

端点在其前面将显示一个*锁*图标，当您发起请求时，您应该看到一个登录警报，要求您提供凭证，正如您在图 8-2 中看到的那样。

![bgai 0802](img/bgai_0802.png)

###### 图 8-2. FastAPI 中的基本身份验证

干得好！在 25 行代码中，你成功实现了基本身份验证来保护端点。现在你可以在自己的原型和开发服务器中使用基本身份验证。

请记住，在生产级别的 GenAI 服务中，您应该避免采用基本的认证机制。对于面向公众的服务，*JWT 认证*是一个更好且更安全的替代方案。它通过在令牌中存储所有认证细节来消除服务器端会话的需求。它还维护数据完整性，并支持跨不同域的广泛接受的标准。

### JSON Web Tokens (JWT) 认证

既然您对认证的基本概念更加熟悉，让我们为您的 FastAPI 服务实现一个更复杂但更安全的 JWT 认证层。作为此过程的一部分，您还需要重构现有的端点，将它们组合在一个单独的资源 API 路由器下，以便一次性对多个端点进行分组、命名、标记和保护。

#### 什么是 JWT？

JWT 是一种通过令牌在应用程序之间断言声明的 URL 安全且紧凑的方式。

这些令牌由三个部分组成：

- **头部**：除了日期时间和发行机构外，还需要指定令牌类型和签名算法。
- **有效载荷**：指定表示资源上声明的令牌主体以及附加元数据的内容。
- **签名**：创建令牌的函数将使用*编码后的有效载荷*、*编码后的头部*、一个*密钥*和签名算法来对令牌进行签名。

###### 小贴士

`base64`编码算法常用于对数据进行编码和解码，以实现紧凑性和 URL 安全性。

图 8-3 展示了一个典型的 JWT 的样子。

![bgai 0803](img/bgai_0803.png)

###### 图 8-3. JWT 组件（来源：[jwt.io](https://jwt.io)）

JWT 是安全的、紧凑的，并且方便，因为它们可以包含执行用户认证所需的所有信息，避免了多次数据库往返的需求。此外，由于它们的紧凑性，您可以通过 HTTP `POST`主体、头部或 URL 参数在网络中传输它们。

#### 开始使用 JWT 认证

要在 FastAPI 中实现 JWT 认证机制，您需要安装`passlib`和`python-jose`依赖项：

```py
$ pip install passlib python-jose
```

在安装了依赖项之后，您还需要在数据库中创建表来存储生成的用户和相关令牌数据。为了数据持久性，让我们将数据库迁移以创建`users`和`tokens`表，如图 8-4 所示。

![bgai 0804](img/bgai_0804.png)

###### 图 8-4. `users`和`tokens`表的实体关系图

如果您查看图 8-4，您将注意到`tokens`表与`users`表之间存在一对一的关系。您可以使用令牌记录来跟踪每个用户的成功登录尝试，并在需要时撤销访问权限。

接下来，让我们定义所需的 SQLAlchemy 模型和 Pydantic 架构，用于数据库查询和数据验证，如示例 8-2 和 8-3 所示。

##### 示例 8-2. 声明用户 SQLAlchemy ORM 模型

```py
# entities.py

import uuid
from datetime import UTC, datetime
from sqlalchemy import Index, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(length=255), unique=True)
    hashed_password: Mapped[str] = mapped_column(String(length=255))
    is_active: Mapped[bool] = mapped_column(default=True)
    role: Mapped[str] = mapped_column(default="USER")
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.now(UTC), onupdate=datetime.now(UTC)
    )

    __table_args__ = (Index("ix_users_email", "email"),)
```

在数据访问层使用 ORM 模型，而在端点层，Pydantic 架构将验证传入和传出的认证数据。

##### 示例 8-3. 声明包含用户名和密码字段验证器的用户 Pydantic 架构
