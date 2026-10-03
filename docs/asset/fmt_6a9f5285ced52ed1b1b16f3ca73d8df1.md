
# 认证与授权服务实现

## 密码服务

```py
# services/auth.py

from fastapi.security import HTTPBearer
from passlib.context import CryptContext

class PasswordService:
    security = HTTPBearer()
    pwd_context = CryptContext(schemes=["bcrypt"]) ![1](img/1.png)

    async def verify_password(
        self, password: str, hashed_password: str
    ) -> bool:
        return self.pwd_context.verify(password, hashed_password) ![2](img/2.png)

    async def get_password_hash(self, password: str) -> str:
        return self.pwd_context.hash(password) ![2](img/2.png)
```

![1](img/1.png)[#co_authentication_and_authorization_CO4-1]

创建一个带有秘密和由 `bcrypt` 库管理的密码上下文的 `AuthService`，该服务将处理所有用户密码的哈希和验证。

![2](img/2.png)[#co_authentication_and_authorization_CO4-2]

使用 `bcrypt` 的密码学算法和应用程序密钥来哈希和验证密码。

`bcrypt` 密码学库为 `PasswordService` 提供了哈希和验证密码的核心功能。使用此服务，现在可以认证请求。

如果请求无法认证，你还需要抛出与授权相关的异常，如示例 8-7 所示。

### 示例 8-7：创建认证异常

```py
# exceptions.py

from fastapi import HTTPException, status

UnauthorizedException = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)

AlreadyRegisteredException = HTTPException(
    status_code=status.HTTP_400_BAD_REQUEST,
    detail="Username already registered",
)
```

你最常遇到的两个授权 HTTP 异常与未经授权访问或由于使用已使用的用户名导致的错误请求有关。

一旦通过凭证验证了用户的身份，你需要为他们颁发一个*访问令牌*。这些令牌应该是短暂的，以减少攻击者如果令牌被盗，可以使用令牌访问资源的时间窗口。

为了减少令牌的大小并防止*令牌伪造*，令牌服务将使用（一个秘密）对令牌有效载荷进行签名并使用如 Base64 之类的编码进行编码。有效载荷通常包含用户的详细信息，例如他们的 ID、角色、发行系统和过期日期。

令牌服务还可以在认证过程中解码接收到的令牌并检查其有效性。

最后，令牌服务还需要数据库访问来存储和检索令牌以执行其功能。因此，它应该继承`TokenRepository`，如示例 8-8 所示。

### 示例 8-8：实现令牌仓库

```py
# repositories.py

from entities import Token
from repositories.interfaces import Repository
from schemas import TokenCreate, TokenUpdate
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

class TokenRepository(Repository):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list(self, skip: int, take: int) -> list[Token]:
        async with self.session.begin():
            result = await self.session.execute(
                select(Token).offset(skip).limit(take)
            )
        return [r for r in result.scalars().all()]

    async def get(self, token_id: int) -> Token | None:
        async with self.session.begin():
            result = await self.session.execute(
                select(Token).where(Token.id == token_id)
            )
        return result.scalars().first()

    async def create(self, token: TokenCreate) -> Token:
        new_token = Token(**token.dict())
        async with self.session.begin():
            self.session.add(new_token)
            await self.session.commit()
            await self.session.refresh(new_token)
        return new_token

    async def update(
        self, token_id: int, updated_token: TokenUpdate
    ) -> Token | None:
        token = await self.get(token_id)
        if not token:
            return None
        for key, value in updated_token.dict(exclude_unset=True).items():
            setattr(token, key, value)
        async with self.session.begin():
            await self.session.commit()
            await self.session.refresh(token)
        return token

    async def delete(self, token_id: int) -> None:
        token = await self.get(token_id)
        if not token:
            return
        async with self.session.begin():
            await self.session.delete(token)
            await self.session.commit()
```

在实现了`TokenRepository`之后，您现在可以开发`TokenService`，如示例 8-9 所示。

### 示例 8-9：通过继承令牌仓库实现令牌服务

```py
# services/auth.py

from datetime import UTC, datetime, timedelta
from exceptions import UnauthorizedException
from jose import JWTError, jwt
from pydantic import UUID4
from repositories import TokenRepository
from schemas import TokenCreate, TokenUpdate

class TokenService(TokenRepository):
    secret_key = "your_secret_key"
    algorithm = "HS256"
    expires_in_minutes = 60 ![1](img/1.png)

async def create_access_token(
    self, data: dict, expires_delta: timedelta | None = None ![2](img/2.png)
) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=self.expires_in_minutes)
    token_id = await self.create(TokenCreate(expires_at=expire)) ![3](img/3.png)
    to_encode.update(
        {"exp": expire, "iss": "your_service_name", "sub": token_id} ![4](img/4.png)
    )
    encoded_jwt = jwt.encode(
        to_encode, self.secret_key, algorithm=self.algorithm ![5](img/5.png)
    )
    return encoded_jwt

async def deactivate(self, token_id: UUID4) -> None:
    await self.update(TokenUpdate(id=token_id, is_active=False))

def decode(self, encoded_token: str) -> dict:
    try:
        return jwt.decode(
            encoded_token, self.secret_key, algorithms=[self.algorithm]
        )
    except JWTError:
        raise UnauthorizedException

async def validate(self, token_id: UUID4) -> bool:
    return (token := await self.get(token_id)) is not None and token.is_active
```

![1](img/1.png)[#co_authentication_and_authorization_CO5-1]

实现一个`TokenService`以发行和检查身份验证令牌。配置在服务的所有实例之间共享。

![2](img/2.png)[#co_authentication_and_authorization_CO5-2]

根据提供给令牌服务的提供的数据生成带有过期日期的访问令牌。

![3](img/3.png)[#co_authentication_and_authorization_CO5-3]

在数据库中创建一个令牌记录并获取一个唯一的标识符。

![4](img/4.png)[#co_authentication_and_authorization_CO5-4]

访问令牌必须在小时内过期，因此将使用计算字段`exp`来检查令牌的有效性。

![5](img/5.png)[#co_authentication_and_authorization_CO5-5]

使用`base64`算法将生成的令牌编码为编码字符串。

现在您已经有了`PasswordService`和`TokenService`，您可以使用专门的更高层次的`AuthService`来完成核心 JWT 身份验证机制。

示例 8-10 显示了包含用于注册用户、发行访问令牌和保护 API 路由的几个依赖函数的`AuthService`类的实现。

### 示例 8-10：实现一个处理高级身份验证逻辑的认证服务
