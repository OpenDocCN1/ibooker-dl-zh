
# 使用 SPIFFE/SPIRE 的双向 TLS

以密码学方式绑定的工作负载身份和短期证书，可以在没有可窃取令牌的情况下实现零信任身份验证。

让我们从最直接的方法开始：将用户的令牌传递整个调用链。

## 智能体冒充（令牌透传）

为了传播用户身份，智能体在任何 MCP 交互中都代表或*冒充*用户。冒充的优势在于，它无需修改即可保留你现有的 RBAC 基础设施。你的审计日志自然会记录哪个最终用户访问了哪些数据，从而一举满足合规要求。你还可以执行用户级配额和速率限制，防止单个用户耗尽共享资源。

在智能体冒充模式中，MCP 服务器从智能体运行时接收最终用户的凭证，并在调用上游 API 时直接使用它们。上游服务会认为请求来自用户，而不是来自智能体。这在概念上类似于 OAuth2 令牌透传模式：智能体运行时将用户的访问令牌传递给 MCP 服务器，MCP 服务器在调用上游 API 时将其包含在 `Authorization` 请求头中。

以护士 Alice 为例，她通过一个医疗助理智能体查询患者记录。护士通过 OpenID Connect 向智能体运行时进行身份验证，并获得一个访问令牌。当护士问：“显示患者 4711 的化验结果”时，智能体运行时将护士的令牌与工具请求一起转发给 MCP 服务器。随后，MCP 服务器在 `Authorization` 请求头中携带 Alice 的令牌，调用医院的患者记录 API。患者记录 API 会执行其现有的用户级权限——检查这位护士是否有权读取患者 4711 的记录——审计日志会显示护士 Alice 访问了患者 4711 的化验结果，而不仅仅是“智能体”访问了它们。

这种模式确实会带来令牌生命周期方面的运维复杂性。用户访问令牌通常会在几分钟到几小时后过期；如果你的智能体任务运行时间超过了令牌的生命周期，除非你实现刷新逻辑，否则调用将会失败。

你还会面临作用域爆炸（scope explosion）问题。用户的令牌必须对智能体可能调用的每个上游 API 都有效，这通常意味着要授予用户宽泛的 OAuth 作用域，而这些作用域违反了最小权限原则。如果你的患者助理代理可能会调用实验室 API、药房 API 和排班 API，那么护士的令牌就需要这三个系统的作用域，即使这次特定查询只涉及其中一个系统。

还存在凭证窃取的风险。如果 MCP 服务器遭到入侵，攻击者可以窃取并重放用户令牌，从而访问用户可访问的任何资源。防御措施要求令牌生存期短、服务之间采用强 mTLS，并且 Pod 内部具备运行时安全。

在 Kubernetes 上，这种模式通常涉及一个入口控制器，用于认证用户并将访问令牌注入请求头。你可以使用 Traefik、带 `oauth2-proxy` 的 NGINX，或带 `RequestAuthentication` 的 Istio，在边缘处理访问令牌传递。

图 9-3 展示了完整流程，并突出显示了验证步骤。

![](img/3a93443ca88ca3c8ef163f48ebd612bc_344_0.png)

*图 9-3. 代理模拟流程，展示用户令牌的传播与验证*

即使你传递用户令牌进行授权，仍应使用双向 TLS（mTLS），无论是直接使用还是通过服务网格（如 Istio 或 Linkerd）来加密 MCP 服务器与上游 API 之间的流量，并验证流量是否来自经过授权的工作负载。

## 服务账户委托

模拟方法在需要用户级权限且身份基础设施支持的情况下效果很好，但当代理和上游服务都运行在同一个 Kubernetes 集群中，且代理级归属足够时，存在一种更简单的替代方案。

你依靠 Kubernetes 原生的工作负载身份，而不是外部令牌服务器，这意味着更少的组件和更低的操作开销。Kubernetes 中的每个 Pod 都已经有一个服务账户（`ServiceAccount`），可以通过标准 RBAC 携带权限。该模式利用这些内置原语，在代理运行时、MCP 服务器和上游 API 之间建立信任，而无需单独的身份提供方。

### 服务账户作为工作负载身份

Kubernetes 中的服务账户是 Pod 的命名空间级身份。当你创建 Pod 时，Kubernetes 会为其分配一个服务账户——要么是你明确指定的账户，要么是 Pod 所在命名空间中的默认服务账户。此身份不绑定到人类用户，而是绑定到工作负载，因此非常适合服务到服务的认证。

每个服务账户都有一个关联的令牌，Kubernetes 会将其自动挂载到 Pod 的 `/var/run/secrets/kubernetes.io/serviceaccount/token` 路径。该令牌是一个签名的 JSON Web 令牌（JWT），包含标识服务账户的声明，包括其名称、命名空间和唯一标识符。Kubernetes API 服务器使用自己的私钥为这些令牌签名，任何信任 API 服务器的组件都可以验证它们。

一旦存在该服务账户，你可以通过在 Pod 的 `spec` 中设置 `serviceAccountName` 字段，将其分配给 Pod。当该 Pod 启动时，Kubernetes 会将服务账户令牌以文件形式注入，并自动保持刷新。

令牌刷新至关重要。服务账户令牌不是静态的；Kubernetes 会定期轮换它们以保障安全。任何读取令牌的代码都应在每次使用时读取，而不是在内存中缓存。每次从文件系统读取可以确保你始终使用的是当前有效的令牌。

服务账户令牌可以在两种环境中使用：集群内部，以及额外配置后的集群外部。在集群内部，服务账户令牌是 Kubernetes API 服务器原生理解的一等公民。当 Pod 调用 Kubernetes API 并在 `Authorization` 头中包含其服务账户令牌时，API 服务器会验证签名、提取身份，并检查 RBAC 策略以决定是否允许该请求。

如果 API 服务器暴露了 OIDC 发现端点，那么服务账户令牌也可以在集群外部得到验证。大多数托管 Kubernetes 服务（如 GKE、EKS 和 AKS）默认启用此功能。在这种设置下，服务账户令牌是有效的 JWT，任何能够访问集群 OIDC 公钥的服务都可以验证它。

这样做的代价是增加了复杂性：你必须配置外部服务信任集群的 OIDC 签发者，获取签名密钥，并处理令牌验证逻辑。我们在第 315 页的“通过 OIDC/JWT 进行外部验证”中更详细地介绍了外部验证。

### 服务器身份与代理身份

服务账户委托分为两种流程，取决于上游 API 看到的是谁的身份。这两种流程都使用服务账户令牌，但区别在于哪个服务账户令牌会到达上游 API。

在 *服务器身份* 流程中，MCP 服务器在调用上游 API 时使用自己的服务账户令牌。代理运行时的身份不会传播；上游 API 基于 MCP 服务器的身份强制执行权限。这种方法更简单，并且当使用给定 MCP 服务器的所有代理运行时对上游资源具有统一访问权限时，效果很好。

在 *代理身份* 流程中，代理运行时将自己的服务账户令牌发送给 MCP 服务器，MCP 服务器再将该令牌转发给上游 API。上游 API 基于代理运行时的身份强制执行权限，从而允许不同的代理运行时在调用同一个 MCP 服务器时拥有不同的访问级别。

图 9-4 并排展示了这两种流程。

![](img/3a93443ca88ca3c8ef163f48ebd612bc_346_0.png)

*图 9-4. 服务器身份与代理身份流程*

关键决策在于粒度。如果每个代理运行时对给定 MCP 服务器的工具都应具有相同权限，请使用服务器身份。如果不同的代理运行时需要不同的权限级别，请使用代理身份。

我们将详细介绍每种方法，首先从创建服务账户及访问其令牌的机制开始。

### 服务账户的使用

要为服务账户授予合适的权限，你必须定义 RBAC 规则。为了代理安全，你应该使用针对具体用例的自定义 API 组和资源，而不是标准的 Kubernetes 资源。

这里有一个关键区别：保护对 Kubernetes `Service` 资源的访问 *不会* 保护对该服务端点的访问。对某个服务拥有 `get` 权限的服务账户只能读取服务元数据，而不能调用实际服务。

相反，应定义代表应用级权限的应用特定资源。示例 9-1 展示了一个基本的 RBAC 设置，涉及 `Role`、`ServiceAccount` 和 `RoleBinding` 声明。有关如何在 Kubernetes 中设置 RBAC 的更多信息，请参阅 *Kubernetes 模式* 中的 *访问控制* 模式。

*示例 9-1. 具有自定义资源 RBAC 的服务账户*

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: customer-support-mcp
  namespace: agents
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: data-platform
  name: customer-data-reader
rules:
- apiGroups: ["agents.example.com"]
  resources: ["customer-queries"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: customer-support-mcp-binding
  namespace: data-platform
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: customer-data-reader
subjects:
- kind: ServiceAccount
  name: customer-support-mcp
  namespace: agents
```

为你的应用领域使用代理特定的 API 组。

## MCP 安全

定义代表应用级权限的自定义资源名称，而不是 Kubernetes 资源。

将 `agents` 命名空间中的 `ServiceAccount` 绑定到位于 `data-platform` 命名空间中的这个角色。

应用特定的资源（如示例 9-1 中的 `customer-queries` 资源，或类似的资源如 `medical-records`、`support-tickets` 等）不需要在 Kubernetes API 服务器上注册为自定义资源定义（CRD）。它们只存在于 RBAC 规则中，纯粹用于通过 `SubjectAccessReview` 进行授权检查。这使您能够在不管理 CRD 开销的情况下获得细粒度、应用特定的权限。

Kubernetes 将 `ServiceAccount` 令牌挂载到每个 Pod 中的已知路径。读取它很简单，但必须正确操作，以避免使用过期的令牌。

示例 9-2 中的函数每次被调用时都会读取令牌，确保你始终使用最新版本。

*示例 9-2：正确读取 ServiceAccount 令牌*

```python
from pathlib import Path

def get_serviceaccount_token() -> str:
    """Read the current ServiceAccount token from the filesystem."""
    token_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
    return token_path.read_text().strip()
```

每次调用时都读取令牌以确保其最新——不要缓存在内存中，因为 Kubernetes 会在文件系统上自动刷新此令牌。

### 发起已认证请求

在服务器身份模式下，MCP 服务器在调用上游 API 时使用自己的 `ServiceAccount` 令牌。对于代理身份模式，代理会发送其 `ServiceAccount` 令牌，验证后，MCP 在向任何上游 API 发出请求时，会将该代理令牌复制到 `Authorization:` HTTP 头中。

示例 9-3 展示了在使用服务级身份时，如何在 HTTP 请求中发送令牌。

*示例 9-3：MCP 服务器使用自己的令牌调用上游*

```python
import httpx
from pathlib import Path

async def call_upstream_with_service_token(
    endpoint: str,
    payload: dict,
    user_id: str | None = None
) -> dict:
    """Call upstream API with the MCP server's ServiceAccount token."""
    sa_token = get_serviceaccount_token()

    headers = {
        "Authorization": f"Bearer {sa_token}",
        "Content-Type": "application/json"
    }

    if user_id:
        payload["_audit_user_id"] = user_id

    async with httpx.AsyncClient() as client:
        response = await client.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
```

使用示例 9-2 中定义的函数，从文件系统重新读取 `ServiceAccount` 令牌。

将令牌作为 Bearer 令牌包含在 `Authorization` 头中。

可选地在 payload 中包含最终用户 ID 以用于审计目的。

上游 API 会看到 MCP 服务器的身份并据此执行权限。如果需要跟踪哪个最终用户触发了请求以用于审计目的，可以在请求 payload 或自定义头（如 `X-User-ID`）中包含该信息。

这种方法简单且开销最小，但使用此 MCP 服务器的所有代理运行时都会获得相同级别的访问权限。

### 通过令牌验证进行身份认证

当 MCP 服务器在确定代理身份时收到来自代理运行时的 `ServiceAccount` 令牌，它必须先验证该令牌，然后再信任它。任何接收来自 MCP 服务器请求的上游 API 服务也是如此。

Kubernetes 正是为此提供了 `TokenReview` API。

`TokenReview` API 以令牌作为输入，返回该令牌是否有效及其所代表的身份。示例 9-4 展示了 MCP 服务器如何与 Kubernetes API 交互以验证令牌。

*示例 9-4：使用 TokenReview 验证代理令牌*

```python
import httpx
from kubernetes import client, config

config.load_incluster_config()
auth_v1 = client.AuthenticationV1Api()

async def validate_agent_runtime_token(token: str) -> dict:
    """Validate agent runtime token using Kubernetes TokenReview API."""
    token_review = client.V1TokenReview(
        spec=client.V1TokenReviewSpec(token=token)
    )
    result = auth_v1.create_token_review(token_review)

    if not result.status.authenticated:
        raise ValueError("Token validation failed: not authenticated")

    username = result.status.user.username
    if not username.startswith("system:serviceaccount:agents:"):
        raise ValueError(f"Token from unauthorized namespace: {username}")

    return {
        "username": username,
        "uid": result.status.user.uid,
        "groups": result.status.user.groups
    }
```

- 从集群内服务账户加载 Kubernetes 配置。
- 创建包含待验证令牌的 `TokenReview` 对象。
- 将 `TokenReview` 提交到 Kubernetes API 服务器。这是一个同步调用，会填充 `TokenReview` 资源的 `status` 部分。
- 检查令牌是否已通过身份认证（签名有效且未过期）。
- 提取格式为 `system:serviceaccount:namespace:name` 的 `ServiceAccount` 用户名。
- 执行允许列表策略：仅接受来自 `agents` 命名空间的令牌。

此验证步骤对安全性至关重要。通过调用 `TokenReview`，MCP 服务器确认 Kubernetes API 服务器签发并签署了此令牌。
