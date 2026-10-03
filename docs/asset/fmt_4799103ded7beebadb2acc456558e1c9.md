
| 本地NVMe存储 | RWO（节点本地） | 微秒级延迟，多GB/s吞吐量 | 高（需要数据暂存，将检查点复制到持久化存储，在Pod重新调度时会丢失） | 数据暂存以获得最大I/O性能，以及能够容忍重新暂存并具有健壮远程检查点策略的作业 |

生产级存储架构通常组合多种解决方案：使用对象存储配合流式API处理大型不可变数据集，使用分布式文件系统或云托管存储来存储需要在分布式工作节点之间进行ReadWriteMany（RWX）访问的共享检查点，并可选地使用本地NVMe进行数据暂存以最大化GPU利用率。

关键要求是具备RWX能力的存储用于检查点和模型工件，使不同节点上的多个工作Pod能够在分布式训练期间访问共享状态，并支持抢占后的作业恢复。

在为训练工作负载配置存储时，平台管理员必须将存储容量规划为不仅容纳最终模型工件，还要容纳训练过程中生成的所有中间检查点。存储容量规划的一个实用经验法则是 `2 × base_model_size + checkpoint_overhead`，即考虑基础模型、中间检查点（训练框架通常每*N*步保存一次）和最终输出。例如，使用LoRA适配器训练Llama 3.1 8B模型并进行频繁检查点保存，通常需要100 GB的存储空间；而全面微调70B模型可能需要500 GB或更多，具体取决于检查点频率和保留策略。

在配置存储基础设施以支持检查点管理和恢复之后，平台管理员还必须解决分布式训练框架以性能为中心的设计所带来的安全问题，特别是在不同团队共享同一GPU集群的多租户环境中。

## 训练作业安全

像Ray和PyTorch Distributed模块这样的分布式训练框架引入了独特的安全挑战，超出了传统Kubernetes工作负载安全考虑的范畴。理解这些框架的架构以及

对于部署多租户训练环境的平台管理员来说，实施适当的默认安全配置至关重要。这主要是因为设计决策将性能置于内置安全隔离之上。

这些框架假设它们在可信网络环境中运行，系统已经对参与者完成了身份验证，而不是使用应用级安全机制。这种设计理念适用于研究或单租户环境的原始用例，但并不代表传统的生产级多租户 Kubernetes 集群。

Ray 和 PyTorch 这两个框架都只提供有限的内置身份验证和授权能力，尤其是在分布式组件之间的通信方面。

任何能够与 Ray 集群或 PyTorch 训练任务建立网络连接的进程，都可以以完整的应用权限执行任意代码。这些框架默认以未加密方式发送消息、接受来自任何网络源的连接，并在不进行安全检查的情况下执行工作负载，将网络可访问性视为隐式授权。

默认配置缺乏通信加密，再加上基于 cloudpickle 的序列化机制（该机制因可执行任意 Python 代码而众所周知地不安全），进一步扩大了攻击面。

平台管理员必须认识到，这些安全限制是优先考虑训练性能而非隔离保证的基本设计选择。这些框架明确记载，分布式功能“仅用于内部通信”，并且“并非为在不可信环境中使用而构建”。试图强行添加强大的安全边界将带来性能开销，从而抵消其核心价值主张。

因此，保护分布式训练部署必须包含基础设施级别的控制。通过 Kubernetes 网络策略（NetworkPolicies）实现的网络隔离成为主要的安全机制，它创建了可信隔离区，在同一训练任务内只有经过授权的 Pod 才能相互通信。可选的加密可以通过 Ray 的 TLS 或 PyTorch 的加密 CNI 插件分层叠加，在接受性能开销的同时提供纵深防御。分布式训练框架必须被视为本质上不安全的组件，只有通过精心设计的基础设施控制进行封装，并在网络和命名空间级别强制执行安全边界，它们才会变得安全。

## Ray 安全指南

Ray 是一个分布式计算框架，通常用于强化学习、超参数调优和分布式训练工作负载。Ray 的安全模型体现了其性能优先的设计理念：它期望在可信网络环境中运行可信代码，不提供内置的访问控制或代码隔离机制。

这一设计原则应用于整个生态系统，使得任何能够通过网络访问 Ray 服务的客户端，都可以通过 Ray 作业（Ray jobs）、Ray 客户端 API（Ray Client APIs）和 Dashboard REST 端点执行任意代码。

首先要考虑的操作是按照 Ray 专属指南为 gRPC 通道启用 TLS 身份验证，该指南描述了如何在 RayCluster 自定义资源中将 TLS 配置为 `rayStartParams`，并将 TLS 证书挂载为 Kubernetes 密钥。这有助于缩小攻击面，但不能替代网络隔离的需求。

平台管理员应在专用的 Kubernetes 命名空间中部署每个 Ray 集群，并实施基础设施级别的安全控制，以便在生产环境中安全地部署 Ray。主要的安全机制是*通过 Kubernetes 网络策略（NetworkPolicies）实现网络隔离*，为 Ray 集群建立严格边界。

配置得当的网络策略应默认拒绝发往 Ray 头节点（head）和工作节点（worker）Pod 的所有入站流量，然后显式地仅允许必要的通信：在 Ray 内部端口上进行的工作节点到头节点的通信（6379 端口用于全局控制服务 GCS，8265 端口用于 Dashboard，10001 端口用于 Ray 客户端服务器）、同一 Ray 集群内用于对象存储访问的 Pod 间通信，以及通过身份验证代理对 Ray Client 或 Jobs API 的严格受控访问（参见示例 7-13）。

*示例 7-13. 用于 Ray 集群隔离的 NetworkPolicy 配置*

```yaml
# Deny all ingress traffic by default for the Ray namespace
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ray-default-deny
  namespace: ray-cluster-team-a
spec:
  # Empty selector applies to all pods in the namespace,
  # denying all ingress by default
  podSelector: {}
  policyTypes:
  - Ingress
---
# Allow worker-to-head communication on Ray internal ports
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ray-worker-to-head
  namespace: ray-cluster-team-a
spec:
  podSelector:
    matchLabels:
      # Select Ray head node pods to receive traffic from workers
      ray.io/node-type: head
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          # Allow ingress only from pods with the same cluster label,
          # ensuring isolation between Ray clusters
          ray.io/cluster: ray-cluster-team-a
      ports:
      # Port 6379 for Ray GCS server
      - protocol: TCP
        port: 6379
      # Port 8265 for Ray dashboard
      - protocol: TCP
        port: 8265
      # Port 10001 for Ray client server
      - protocol: TCP
        port: 10001
---
# Allow pod-to-pod communication within same Ray cluster
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ray-intra-cluster
  namespace: ray-cluster-team-a
spec:
  podSelector:
    matchLabels:
      ray.io/cluster: ray-cluster-team-a
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          # Object store requires cluster-internal access
          ray.io/cluster: ray-cluster-team-a
---
# Allow controlled access to Ray Client/Jobs API through auth proxy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ray-api-access
  namespace: ray-cluster-team-a
spec:
  podSelector:
    matchLabels:
      ray.io/node-type: head
  policyTypes:
  - Ingress
  ingress:
  - from:
    # Only allow access from authentication proxy in a separate namespace
    - namespaceSelector:
        matchLabels:
          name: auth-proxy-namespace
      podSelector:
        matchLabels:
          app: oauth2-proxy
      ports:
      # Dashboard access through authentication proxy only
      - protocol: TCP
        port: 8265
```

## PyTorch 安全指南

PyTorch Distributed 是分布式深度学习中使用最广泛的框架，为全球大规模训练工作负载提供支持。PyTorch 的分布式数据并行（DDP）在多个进程间复制模型，并使用集合通信后端同步梯度。NCCL 用于 GPU 到 GPU 的通信，Gloo 用于基于 CPU 的通信。

PyTorch 的分布式功能与 Ray 存在相同的严重安全限制：安全策略明确声明，分布式功能“仅用于内部通信”，并且“并非为在不可信环境中使用而构建”。该框架不提供内置授权，默认以未加密方式发送消息，并接受来自任何网络源的连接。任何拥有网络访问权限的人都可以以完整权限执行任意代码。与 Ray 不同，PyTorch 分布式不提供内置的 TLS 加密，使得网络级隔离成为唯一有效的安全控制手段。

在 Kubernetes 中保护 PyTorch 分布式训练，需要实施与框架安全假设相一致的 NetworkPolicy，以创建网络隔离边界。平台管理员应利用标签选择器来限定 NetworkPolicy 的作用范围，例如，使用 `pytorch-job-name=my-training-job` 之类的标签来选择任务的 Pod，从而允许任务内部通信，同时阻止外部流量。设计良好的策略应遵循为 Ray 提供的示例的相同模式：默认拒绝所有入站流量，然后显式地仅允许必要的通信：同一 PyTorchJob 内的 Pod 间通信（NCCL 使用动态端口范围，需要任务 Pod 之间的所有 TCP 流量），以及从 Kubeflow Trainer 命名空间到 rank-0 工作节点的入站流量。

## 训练任务的可观测性

Kubernetes 上分布式训练任务的可观测性带来了超越传统应用监控的独特挑战，要求平台管理员对系统进行监控埋点，以跟踪数十或数百个执行协调工作负载的临时 Pod 上的训练进度、资源利用率和任务健康状况。

### 训练作业的可观测性

与无状态微服务不同，在无状态微服务中监控单个实例即可获得足够的可见性，而分布式训练作业则要求跨所有 worker Pod 进行关联可观测性。单个 worker 变慢可能会拖慢整个作业的进度，一个节点上的 GPU 利用率可能不理想，而其他节点却运行高效，或者梯度同步瓶颈可能只有在检查整个 worker 集合的通信模式时才会显现。

此外，训练作业会长时间运行（几小时到几周），因此必须同时捕获用于即时检测问题的实时运行指标，以及用于分析收敛行为、调试失败实验和跨多次训练运行优化超参数的历史训练指标。

平台管理员必须实现覆盖三个维度的全面可观测性：跟踪模型性能和收敛的应用级训练指标、监控资源利用率（即 GPU）和作业健康的基础设施指标，以及捕获通信模式和协调开销的分布式系统指标。

#### 分布式训练的指标收集

我们在第 164 页的“GPU 使用情况监控”中已经介绍了如何跟踪 GPU 指标，因此本节重点介绍其他组件：

- **训练指标**：训练指标捕获实际的模型性能和学习进度，帮助了解训练运行是收敛到目标精度，还是因超参数配置错误而发散。现代训练框架与实验跟踪系统集成，这些系统记录训练损失、验证损失、准确率、学习率计划，以及数据科学家定义的自定义应用特定指标。TensorBoard 是 TensorFlow 生态系统的一部分，但也已成为 PyTorch 中可视化训练指标的事实标准。它被集成到训练代码中，并使用 `torch.utils.tensorboard.SummaryWriter` 或 `tf.summary` 记录指标。TensorBoard 服务器实例读取这些日志，提供基于 Web 的仪表板，显示训练步骤和 epoch 上的指标趋势。在 Kubernetes 部署中，TensorBoard 通常作为独立的 Deployment 或 Pod 运行，挂载训练作业写入日志所使用的那同一个持久卷，从而在作业跨分布式 worker Pod 执行时，让数据科学家能够实时监控训练进度。

  TensorBoard 适用于单次运行，而跨多次训练运行比较指标则需要使用 MLflow 或 Weights & Biases 等其他工具。

- **作业级指标**：使用 Kubeflow Trainer Operator 可以简化整个作业的跟踪，监控 worker Pod 的状态、副本数量，并更新状态条件（如 Created、Running、Succeeded 或 Failed）。每个作业由多个 Pod 组成，这些 Pod 会生成可导出到 Prometheus 的常规指标，训练作业也会产生与作业生命周期相关的 Kubernetes 事件。

#### 跨分布式 Worker 的日志记录

分布式训练作业的日志记录会带来复杂性，因为有意义的日志分析需要关联多个并行执行的 worker Pod 的日志，而这些 Pod 通常会在略有不同的时间戳上生成相同的日志消息。

最直接的方法是采用*集中式日志记录*基础设施，将所有 Pod 日志收集到可搜索的日志聚合系统中，例如 Elasticsearch、Loki 或 CloudWatch Logs，并按作业名称、worker rank 和 Pod 名称标记日志，以便进行过滤和关联。

在 PyTorch 分布式训练中，标准做法是仅让 rank-0 输出详细训练日志，显示 epoch 进度、损失值和检查点操作，而其他 worker 要么完全抑制输出，要么只记录错误条件，从而减少日志量并避免重复信息。这种方法通常足以提供对作业内部状态的良好可见性，但调试分布式训练故障时，常常需要检查所有 worker 的日志，以确定具体哪个 rank 遇到错误。因此，一个好的实践是始终收集所有 worker 的日志，并默认使用工具或 UI 过滤器仅显示 rank-0 的日志。

在规模化场景下，日志量可能成为主要挑战，因为对多个作业的日志进行通用全文搜索并不是一种可管理的解决方案。这使得结构化日志记录变得至关重要：训练代码应以结构化（JSON）格式输出日志，并包含一组通用字段，如 `job_name`、`worker_rank`、`step_number`、`epoch`、`loss_value` 等。可以将 Kubernetes 日志框架（如 Fluent Bit 或 Fluentd）配置为解析日志，并使用来自 Pod 标签和注解的额外元数据丰富日志，自动添加命名空间、节点名称和 GPU 设备 ID 字段，以创建可查询的完整日志记录。

#### 追踪分布式训练操作

为了突出显示与协调或通信模式相关的瓶颈，分布式追踪可能是必要的。

PyTorch 通过 PyTorch Profiler（`torch.profiler`）提供内置的性能分析能力，它可以对训练代码进行插桩，以捕获详细的性能跟踪信息，包括 CPU 操作、GPU 内核执行、内存分配，以及对分布式训练至关重要的集合通信操作，如 all-reduce 和 all-gather。

Profiler 结果可以通过 TensorBoard 的性能分析插件进行可视化，该插件提供时间线视图（展示 GPU 利用率随时间的变化）、堆栈跟踪（用于识别训练代码中的性能瓶颈）以及分布式视图（展示跨 rank 的通信模式）。这可以帮助数据科学家优化批次大小、调整梯度累积策略，或识别制约分布式训练吞吐量的网络瓶颈。

还可以使用 NVIDIA Nsight Systems 在 CUDA 内核级别对 GPU 工作负载本身进行更底层的性能分析。然而，这种级别的分析通常保留给特定的性能优化工作，而非日常监控，因为长时间运行的分析会话生成的跟踪文件大小可能增长到 GB 级，并且分析开销本身也会影响训练性能。

#### 经验总结

在本章中，我们探讨了在 Kubernetes 上运行生产级 AI 训练工作负载所需的运维基础，从调度、网络到存储和安全。
