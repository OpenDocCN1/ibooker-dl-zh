
# 允许列表检查

允许列表检查是一个简单的基于命名空间的过滤器，用于初始访问控制。它限制了哪些 ServiceAccount 可以使用此 MCP 服务器，防止来自无关命名空间的 Pod 调用你的工具。对于基于 RBAC 策略的细粒度授权，请使用下一节所述的 SubjectAccessReview。

令牌验证会带来一些小的延迟。你可以使用令牌的哈希作为键，以较短的生存时间（TTL）缓存验证结果，以减少开销，但必须确保缓存能正确处理令牌过期。

## 使用 SubjectAccessReview 进行授权

验证令牌可以证明身份，但并不能告诉你该身份是否有权执行特定操作。Kubernetes 为此提供了 SubjectAccessReview API 用于授权检查。

SubjectAccessReview 向 Kubernetes API 服务器询问：“该 ServiceAccount 能否对此资源执行此操作？”它遵循所有 RBAC 策略，因此你可以根据集群的当前状态得到明确答案。示例 9-5 演示了如何向 API 服务器查询此信息。

**示例 9-5：使用 SubjectAccessReview 检查权限**

```python
from kubernetes import client

authz_v1 = client.AuthorizationV1Api()

async def check_agent_permission(
    username: str,
    namespace: str,
    api_group: str,
    resource: str,
    verb: str
) -> bool:
    """Check if a ServiceAccount has permission to perform an action."""
    sar = client.V1SubjectAccessReview(
        spec=client.V1SubjectAccessReviewSpec(
            user=username,
            resource_attributes=client.V1ResourceAttributes(
                namespace=namespace,
                group=api_group,
                resource=resource,
                verb=verb,
            )
        )
    )
```

```python
result = authz_v1.create_subject_access_review(sar)
return result.status.allowed
```

- TokenReview 中的 ServiceAccount 用户名，例如 `system:serviceaccount:agents:agent-runtime`。
- 自定义资源的 API 组，例如 `agents.example.com`。
- 资源类型，例如 `customer-queries`。
- 正在执行的操作：`get`、`list`、`create`、`update`、`delete` 等。
- 将 SubjectAccessReview 提交到 API 服务器。

这种方法让您无需构建独立的授权系统，即可利用示例 9-1 中定义的 Kubernetes RBAC 来实现应用级权限。您所检查的自定义资源（`customer-queries`、`medical-records` 等）不需要作为 CRD 存在——它们是仅用于授权决策的虚拟资源。

## 通过 OIDC/JWT 进行外部验证

虽然 ServiceAccount 委派的常见场景是在集群内使用，但有些情况下您需要在集群外验证 ServiceAccount 令牌。例如，您可能调用支持 OIDC 联合的云提供商 API，或者您的混合架构中某些服务在 Kubernetes 之外运行，但仍需要信任集群身份。

Kubernetes 可以将 ServiceAccount 令牌公开为符合 OIDC 标准的 JWT，任何支持 OIDC 的服务都可以验证这些令牌。这要求集群的 API 服务器配置 OIDC 颁发者 URL，大多数托管 Kubernetes 服务默认启用此功能。示例 9-6 给出了如何以编程方式实现对 ServiceAccount 令牌验证的示例。

集群的 API 服务器在 `<cluster-url>/.well-known/openid-configuration` 地址公开了 OIDC 发现端点。该端点发布集群的 OIDC 颁发者 URL 以及用于签名令牌的 JSON Web 密钥集（JWKS）的位置。外部服务获取 JWKS，使用这些密钥验证令牌的签名，并验证标准的 JWT 声明，例如过期时间和受众。

**示例 9-6. 通过 OIDC 在外部验证 ServiceAccount 令牌**

```python
import jwt
import httpx

async def validate_sa_token_externally(
    token: str,
    cluster_issuer: str,
    expected_audience: str
) -> dict:
    """Validate a Kubernetes ServiceAccount token using OIDC discovery."""
    discovery_url = f"{cluster_issuer}/.well-known/openid-configuration"
    async with httpx.AsyncClient() as client:
        discovery_resp = await client.get(discovery_url)
        discovery_resp.raise_for_status()
        discovery = discovery_resp.json()
        jwks_uri = discovery["jwks_uri"]
        jwks_resp = await client.get(jwks_uri)
        jwks_resp.raise_for_status()
        jwks = jwks_resp.json()
        signing_key = jwt.PyJWKClient(jwks_uri).get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=expected_audience,
            issuer=cluster_issuer
        )
        return claims
```

- 从集群的颁发者 URL 发现 OIDC 配置端点。
- 获取包含用于签名令牌的公钥的 JSON Web 密钥集。
- 根据令牌的密钥 ID 标头提取正确的签名密钥。
- 验证令牌的受众声明，以确保它是为您的服务签发的。
- 返回验证后的声明，包括 ServiceAccount 身份。

要使此功能生效，集群必须配置为在其创建的 ServiceAccount 令牌中包含受众声明。默认情况下，Kubernetes 集群为其签发的所有服务账户令牌使用自己的颁发者 URL（通过 `--service-account-issuer` Kubernetes API 服务器启动选项指定）作为受众。

您可以使用 `--api-audiences` 选项，以逗号分隔的受众 URL 列表覆盖默认值。

示例 9-7 展示了如何为单个 Pod 的 ServiceAccount 指定受众。

**示例 9-7. 为 Pod 的 ServiceAccount 声明受众**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: demo
spec:
  serviceAccountName: my-sa
  containers:
    - name: app
      image: ghcr.io/example/app:latest
      volumeMounts:
        - name: oidc
          mountPath: /var/run/my-audience
          readOnly: true
  volumes:
    - name: oidc
      projected:
        sources:
          - serviceAccountToken:
              path: token
              audience: "https://my.service.example"
              expirationSeconds: 3600
```

- 所附加的 ServiceAccount。
- 挂载服务账户令牌的目录。
- 要挂载的服务账户令牌列表，可为多个受众提供不同条目。
- 保存令牌的文件名。
- 添加到令牌 JWT 的 `aud:` 声明中的受众。

如果您需要为调用不同上游服务而使用多个受众，可以在示例 9-7 中指定多个 `serviceAccountToken` 条目，每个条目挂载到不同的文件中，或者利用 TokenRequest API 生成针对多个受众的令牌。

ServiceAccount 委派非常适合集群边界内的工作负载到工作负载的认证，但它从根本上说是基于工作负载的，而非基于用户的。

当您需要在系统边界之间将操作归因于单个用户时，OAuth2 提供了委派访问的主流标准。OAuth2 允许用户授予应用程序代表其行动的权限，而无需共享凭据，这正是代理代表用户调用上游 API 时所需要的。

虽然本书并未全面介绍 OAuth2，但理解令牌交换——OAuth2 的一个关键扩展——对于代理型安全模式至关重要。令牌交换允许您用一种凭据换取另一种具有不同范围和受众的凭据，从而实现保留用户和代理身份的细粒度委派。

## OAuth2 与模型上下文协议

模型上下文协议规范在 MCP 服务器需要经过身份验证的访问时使用 OAuth 2.1 进行授权。MCP 服务器充当 OAuth 资源服务器，使用标准 OAuth2 机制保护其工具和资源。

MCP 实现遵循既定的 OAuth2 规范。MCP 客户端*必须*实现 OAuth 2.0 授权服务器元数据（RFC 8414）以发现授权端点。实现*应该*支持动态客户端注册（RFC 7591）以简化设置。所有客户端*必须*使用用于授权码流程的代码交换证明密钥（PKCE）。

对于需要委派语义的多用户代理型系统，RFC 8693（令牌交换）提供了同时保留用户和代理身份的机制；我们将在下一节详细探讨这种模式。

有关 OAuth2 安全模式和委派流程的全面介绍，请参阅 Gary Archer 等人所著的 *使用 OAuth 的云原生数据安全*（O’Reilly 出版）。

## 通过 OAuth2 令牌交换实现委派身份

模拟模式保留了用户上下文，但引入了令牌生命周期复杂性和范围爆炸问题。服务账户模式简化了操作，但失去了按用户归属的能力。OAuth2 令牌交换（RFC 8693）两者兼顾：既提供了一种保留用户身份的标准方式，又使调用服务的身份对上游系统可见。

令牌交换实现了*委派语义*，使原始用户与代表其执行操作的服务仍然可以区分开来。交换后的令牌在其声明中同时携带两种身份：`sub` 标识为其执行工作的用户（例如 `alice@example.com`），而 `act` 标识当前执行操作的角色（例如 `customer-support-agent` 或 `customer-data-mcp-server`）。这种双重身份令牌允许服务执行复合策略，例如“如果用户具有权限*并且*服务已获授权进行此操作，则允许”。

令牌交换可以在MCP工作流的两个节点发生。首先，代理运行时可以将用户的令牌交换为面向MCP服务器的令牌，该令牌同时标识用户和代理。其次，MCP服务器可以将其令牌交换为面向上游API的令牌，该令牌标识用户和MCP服务器。两种场景下的交换机制完全相同——只有 `act` 声明中的参与者身份会根据当前执行工作的服务而变化。

本节重点介绍第一种场景：代理运行时将用户令牌交换为面向MCP服务器的令牌。该模式同样适用于MCP服务器为访问上游API而进行的令牌交换。

令牌交换流程涉及一个令牌服务——通常是您的身份提供商或专用的安全令牌服务——如图9-5所示。用户进行身份验证，并从身份提供商处获取访问令牌。代理运行时调用令牌交换端点，将用户令牌作为 `subject_token`，将MCP服务器作为 `audience`。令牌服务验证用户令牌并返回一个委派令牌。代理运行时将这个委派令牌发送给MCP服务器，MCP服务器可以直接使用它，或者再次交换以获取上游API调用的权限。

![](img/3a93443ca88ca3c8ef163f48ebd612bc_356_0.png)

*图9-5. OAuth2 令牌交换*

回顾第306页“代理模拟（令牌直通）”中的医疗保健示例。当护士爱丽丝查询患者记录时，代理运行时会将她的令牌交换为同时标识医疗助理代理和护士爱丽丝的委派令牌。上游患者记录API强制执行复合策略：“仅当代理是医疗助理 *且* 爱丽丝有权访问该患者时才允许访问。”两个身份都保存在一个经过加密签名的单一令牌中。

当合规性要求同时了解谁（用户）和什么（代理）访问了数据，且您的身份提供商支持令牌交换时，此模式是理想之选。Keycloak、Auth0和Azure AD等现代身份平台均支持RFC 8693。如果您有多个具有不同权限范围的代理，令牌交换可以实现细粒度的权限范围控制，而无需为每个代理创建单独的用户账户。

在Kubernetes上，实施令牌交换需要将您的身份提供商配置为支持RFC 8693——对于Keycloak，这是内置功能。代理运行时在调用MCP服务器之前执行交换，如示例9-8所示。

**示例9-8. 代理运行时通过RFC 8693执行令牌交换**

```python
import httpx

async def exchange_token(
    user_token: str,
    agent_token: str,
    upstream_audience: str,
    token_endpoint: str
) -> str:
    """Exchange user token for delegated token via RFC 8693."""
    payload = {
        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
        "subject_token": user_token,
        "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
        "audience": upstream_audience,
        "actor_token": agent_token,
        "actor_token_type": "urn:ietf:params:oauth:token-type:access_token",
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(token_endpoint, data=payload)
        response.raise_for_status()
        token_data = response.json()
        return token_data["access_token"]
```

- 用于令牌交换请求的RFC 8693授权类型。
- 用户的访问令牌（代理代表其执行操作的委托主体）。
- MCP服务器或上游API作为受众，将令牌范围限定到特定服务。
- 代表执行交换的参与者的代理访问令牌。
- 根据RFC 8693，当存在 `actor_token` 时，必须提供 `actor_token_type`。
- 委派令牌包含两个身份：用户位于 `sub` 中，代理位于 `act` 中。

MCP服务器从代理运行时接收交换后的令牌。它可以直接将此双身份令牌转发给上游API，或者执行自己的令牌交换，成为 `act` 声明中的新参与者，同时在 `sub` 中保留用户身份。

权衡之处在于额外的复杂性。您必须运营一个令牌交换端点，处理令牌交换错误，并缓存交换后的令牌，以避免每次代理调用都重复进行交换而增加延迟。

为了在不同上下文中隔离令牌，您可以基于元组 `(user_subject, agent_identity, audience)` 构造缓存键。当您收到交换后的令牌时，解码JWT并提取 `exp` 声明（Unix秒内的过期时间戳）。将您的缓存生存时间（TTL）设置为 `exp - current_time - safety_margin`，其中安全裕度用于解决时钟偏差和网络延迟——通常为30-60秒。切勿将令牌缓存超过其实际生存期；过期的缓存令牌会被上游API拒绝，迫使您无论如何都要重新交换，这既浪费了缓存存储空间，又浪费了失败的API调用。
