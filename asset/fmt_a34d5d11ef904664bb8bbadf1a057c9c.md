
```yaml
# PyTorchJob that will use topology-aware flavor selection
# (for TrainJob, apply this label to the TrainJob metadata)
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: llm-training
  labels:
    # This label triggers the Kueue admission controller and
    # binds the job to a specific queue
    kueue.x-k8s.io/queue-name: team-queue
spec:
  ...
```

## NVIDIA KAI 调度器

KAI Scheduler 提供了最先进的 GPU 拓扑感知能力，原生理解 NVIDIA GPU 互联技术，包括 NVLink、NVSwitch、NVLink 桥接器，以及它们与 InfiniBand 等网络架构之间的关系。

KAI Scheduler 与 NVIDIA GPU Operator（参见第 95 页的“NVIDIA GPU Operator”）紧密集成，以访问详细的 GPU 信息并做出拓扑感知调度决策，从而减少平台管理员手动打标签的需求。

该调度器在放置分布式训练作业时会分析 GPU 拓扑，自动将 worker 共同放置在具有最佳网络配置的节点上。例如，对于需要 8 个 GPU 的训练作业，最好将其放置在一个单一节点上，每个 GPU 都通过 NVSwitch 与其他 GPU 相连，而不是通过 InfiniBand 将 GPU 分散到多个节点上。Kueue 通过 ResourceFlavors 在准入层管理拓扑感知（决定使用哪一层硬件），而 KAI Scheduler 则将拓扑优化直接集成到调度决策中，并与组调度和公平共享能力一起，确保拓扑优化的放置不会违反配额策略或造成资源碎片化。

这种深度的 GPU 拓扑集成和对 NVIDIA 特有硬件的感知，使 KAI Scheduler 成为大规模 NVIDIA GPU 集群的自然之选；在这些集群中，训练性能在很大程度上取决于 GPU 互联的选择，尤其是与 NVIDIA 特有功能（如第 98 页“子 GPU 分配”一节中描述的多实例 GPU（MIG）分区和 GPU 分数分配）结合使用时。

## Volcano

Volcano 调度器包含拓扑感知调度插件，可根据节点标签提供的硬件拓扑信息优化 Pod 放置。与 Kueue 的 ResourceFlavor 方法类似，平台管理员必须为节点打上拓扑特征标签（例如 `gpu-interconnect: nvlink`、`network-fabric: infiniband`、`rack-id: rack-1`），Volcano 的拓扑插件会使用这些标签，在节点为作业的 GPU 需求提供最佳互联带宽时对其给出更高评分。

然而，Kueue 将 ResourceFlavors 用作准入阶段的抽象，将实际放置委托给底层调度器；Volcano 则将拓扑评分直接集成到其调度决策中，自动将分布式训练 worker 共同放置在同一个机架内或具有直接 NVLink 连接的节点上，而无需显式的亲和性规则。

### 如何做出正确的选择

对于需要高级拓扑感知的组织，将不同解决方案分层组合，可以构建全面的拓扑感知组调度。一种常见的架构是将 Kueue 的准入控制与 Volcano 或 KAI Scheduler 等拓扑感知调度器相结合，从而 Kueue 通过 ResourceFlavor 拓扑选择在作业级别管理准入，而底层调度器负责详细的拓扑优化 Pod 放置。这种关注点分离使平台管理员能够在准入层实施拓扑策略（请求高级互连的作业必须具备相应的配额），同时将放置优化委托给对拓扑有深入理解的专用调度器。

平台管理员通过相应地为节点打标签（机架 ID、网络架构类型、GPU 互联能力），并在所选调度器中创建拓扑感知配置，来配置拓扑信息。不同的 GPU 厂商都提供了简化这一打标签过程的工具；尤其是 NVIDIA 的工具套件先进且全面。

虽然前面讨论的调度器侧重于根据 GPU 互联特性在节点间放置 Pod 的拓扑感知，但在单个节点层面，CPU 和设备局部性也会影响性能，因此拓扑考量同样重要。Kubernetes 已通过 Topology Manager 组件解决了这一挑战（参见下面的补充说明）。

#### Kubernetes 拓扑管理器

Kubernetes Topology Manager 是一个运行在每个节点上的 Kubelet 组件，它通过协调资源分配来补充调度器级别的拓扑策略，从而确保单个 Pod 获得最佳硬件局部性。

在现代多插槽服务器上实现最佳硬件使用时，需要考虑的一个主要方面是非统一内存访问（NUMA）内存架构：每个 CPU 插槽都有自己的本地内存，访问其他插槽上的内存会产生更高的延迟。对于 GPU 密集型工作负载来说，这是一个关键考量，因为跨 NUMA 内存访问可能会显著降低性能。

在 Topology Manager 之前，Kubernetes 中的 CPU Manager 和 Device Manager 会独立做出资源分配决策，可能会将来自一个 NUMA 节点的 CPU 和来自另一个 NUMA 节点的 GPU 分配给同一个 Pod，从而迫使产生跨 NUMA 内存流量（图 7-3）。

Topology Manager 通过为单个 Pod 启用 CPU 和设备（包括 GPU）的拓扑感知来解决此问题，确保分配给 Pod 的 CPU 和 GPU 位于同一个 NUMA 节点，以最大限度地减少内存访问延迟。

平台管理员可以使用不同的策略配置 Topology Manager：`best-effort` 会尝试 NUMA 对齐但不强制执行；`restricted` 仅在资源能够正确对齐时才允许 Pod 准入；`single-numa-node` 要求 Pod 的所有资源都来自单个 NUMA 节点。这是最严格的策略，可保证最佳局部性，但在资源受限的集群中可能会降低调度灵活性。

![](img/3a93443ca88ca3c8ef163f48ebd612bc_265_0.png)

*图 7-3：单个多插槽服务器内的 NUMA 架构：本地内存访问与跨 NUMA 内存访问的延迟对比*

虽然拓扑感知调度会针对性能优化单个作业的放置，但它并未解决在共享集群中多个团队竞争时公平分配稀缺 GPU 资源这一根本性挑战。

## 配额管理与多租户：GPU 即服务

虽然 Gang 调度确保分布式作业中的所有 Pod 被一起调度，拓扑感知调度则优化它们在具有最佳互连的硬件上的放置，但这些机制只解决单个作业如何执行的问题。它们并没有解决当多个团队竞争集群容量时，谁能获得稀缺 GPU 资源的问题。

将共享 GPU 集群作为多租户平台运营，需要精细的配额管理，以确保跨团队的公平资源分配，同时最大化整体 GPU 利用率。对于 AI/ML 工作负载，这是标准 Kubernetes ResourceQuotas 无法充分应对的挑战。

传统的 Kubernetes 配额在命名空间级别使用硬限制，配额耗尽时会阻止工作负载启动，但这种方法没有解决固定且稀缺的资源问题，在实践中导致 GPU 利用率低下。某个团队的数据科学家没有运行实验时，该团队可能有未使用的配额，而另一个团队的作业尽管有紧迫的训练截止日期，却无限期排队；昂贵的 GPU 资源因为配额边界阻止了动态共享而闲置不用。

*GPU 即服务（GPUaaS）*架构通过实现包含借用、抢占和公平性策略的分层配额管理解决了这一问题。这些策略使团队能够在集群容量可用时突破其保证配额进行突发使用，同时确保在需求超过供应时，没有任何团队能够垄断资源。其主要目标是以机会式方法最大化 GPU 使用率，同时在需要时保证既定配额。

## 对比配额管理与多租户解决方案

多种调度解决方案都提供了 GPU 资源的配额管理能力，每种方案在资源分配、公平性策略和多租户支持方面各有不同。表 7-4 对主要选项进行了比较。了解每种解决方案如何处理配额执行和资源借用，有助于平台管理员选择符合其组织跨团队公平 GPU 共享需求的技术。

*表 7-4. 配额管理与多租户解决方案*

| 解决方案 | 配额管理能力 | 多租户方法 | 实现方式 | 最适合场景 |
|---|---|---|---|---|
| Kueue | 支持借用和抢占的分层配额 | 命名空间范围的 LocalQueue 映射到集群范围的 ClusterQueue，基于队列组的跨团队共享，用于工作负载优先级的优先级类别 | 使用 ClusterQueue/LocalQueue 模型的准入控制器、基于队列组的资源借用、基于优先级的抢占 | 需要灵活配额共享、团队间资源借用以及不替换调度器的准入控制的多租户环境 |
| NVIDIA KAI Scheduler | 基于项目的 GPU 配额与公平性算法 | Project CRD 通过专用 GPU 配额隔离团队，层级队列结构支持部门与团队组织，公平份额防止垄断 | 集成调度器，使用 Project CRD 进行配额分配、公平份额调度和 GPU 特定优化 | GPU 密集型集群，需要配额策略与 GPU 感知调度决策紧密集成（分片 GPU、MIG 支持） |
| Volcano | 基于队列的配额与比例分配 | 多个具有独立资源限制的队列、队列级优先级、命名空间到队列的映射以实现团队隔离 | Queue CRD，包含资源限制、用于比例共享的权重，以及用于抢占的回收策略 | 需要在一个组件中集成队列管理和调度，并实现比例资源划分的批处理工作负载 |

### Kueue

Kueue 提供了一个专为训练作业等批处理工作负载设计的综合配额管理系统，实现了将集群级资源治理与团队级队列管理分开的两层架构。

在集群层面，ClusterQueue 定义资源池，包含 CPU、内存和 GPU 的配额限制，以及多个团队竞争有限容量时资源如何分配的策略。平台管理员可以创建代表不同资源层级的 ClusterQueue，例如，一个拥有 32 块 NVIDIA H100 GPU、用于生产训练工作负载的“gpu-training” ClusterQueue；一个拥有 8 块 GPU、用于实验的“gpu-development” ClusterQueue；以及一个使用云 Spot 实例、用于成本敏感工作负载的“gpu-spot” ClusterQueue。每个 ClusterQueue 都指定了*名义配额*（工作负载可用的保证资源）和*借用限制*（其他队列空闲时可临时借用的最大资源量），从而实现灵活的能力共享，在不牺牲公平性保证的前提下提高利用率。LocalQueue 为提交工作负载提供了面向团队的接口，映射到特定的 ClusterQueue，并支持命名空间范围的作业提交和自动配额执行。

队列组（cohort）管理功能通过将不同的 ClusterQueue 分组来实现资源借用，从而支持复杂的多租户共享策略。每个 ClusterQueue 将其保留配额贡献给队列组的总配额，因此总容量是所有这些配额之和。当团队 A 有等待中的工作负载但其 10 个 GPU 的配额已耗尽时，Kueue 允许该团队在队列组限制内借用团队 B 和团队 C 配额中的未使用容量。Kueue 会自动扩展团队 A 的有效分配以利用空闲 GPU，但当团队 B 或团队 C 提交需要其保证配额的新作业时，会立即抢占这些借用的资源。与硬配额边界相比，这种动态共享显著提高了集群利用率。只要有作业在等待，GPU 就不会闲置，同时还能维持公平性保证，确保每个团队始终能够获得其名义配额分配。此外，Kueue 支持能够模拟复杂组织结构的分层队列，允许平台管理员映射部门层级、项目团队和资源池，以反映真实的组织边界。

示例 7-9 包含了所有这些特性：队列组（cohort）、名义配额、借用限制和优先级类别。

*示例 7-9. 使用队列组和优先级类别的 Kueue 配额管理*

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: team-a-queue
spec:
  cohort: shared-gpu-cohort
  resourceGroups:
  - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
    flavors:
      # Using a built-in ResourceFlavor that matches all nodes.
      - name: default-flavor
        resources:
        - name: nvidia.com/gpu
          nominalQuota: 10
          borrowingLimit: 22
  queueingStrategy: BestEffortFIFO
  preemption:
    reclaimWithinCohort: Any
    withinClusterQueue: LowerPriority
---
# High-priority class for production workloads
apiVersion: kueue.x-k8s.io/v1beta1
kind: WorkloadPriorityClass
metadata:
  name: production-priority
spec:
  # Specifying a high value makes this class higher priority
  value: 10000
  description: "High priority for production training jobs"
---
# Production PyTorchJob from Team A (high priority)
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: production-llm-training
  namespace: team-a
  labels:
    # The definition of the LocalQueue is skipped here for brevity
    kueue.x-k8s.io/queue-name: team-a-training-queue
    kueue.x-k8s.io/priority-class: production-priority
  ...
```

![](img/3a93443ca88ca3c8ef163f48ebd612bc_269_0.png)

多个 ClusterQueue 属于同一个队列组 `shared-gpu-cohort`，从而支持跨团队资源借用。总容量是该队列组中所有 ClusterQueue 配额之和。

`BestEffortFIFO` 排队策略按先进先出顺序处理工作负载，同时尽力高效地打包资源。当配额可用时，作业按照提交顺序被接纳。

`reclaimWithinCohort: Any` 策略允许任何团队在需要收回其名义配额时抢占借用的资源。

`withinClusterQueue: LowerPriority` 策略允许同一 ClusterQueue 内的高优先级工作负载在配额耗尽时抢占低优先级工作负载。

该生产作业使用 `production-priority`，这意味着 Kueue 会在实验作业之前接纳它，并在必要时抢占这些实验作业。

Kueue 的公平份额算法会考虑每个团队的历史资源消耗情况和当前队列深度，优先考虑最近消耗资源较少或等待接纳时间较长的团队。它实现了一种在公平性（确保所有团队都能获得资源）与效率（优先接纳能够有效利用资源的大型作业）之间取得平衡的排队策略。

这些优先级类别使紧急的生产训练作业能够抢占低优先级的实验工作负载，从而使组织能够实施服务级别协议：在需求高峰期让特定团队或作业类型获得优先访问权，同时在其它时段仍可共享容量。

## NVIDIA KAI 调度器

NVIDIA KAI Scheduler 实现了针对 GPU 的配额管理，采用为大规模 GPU 集群设计的层级队列结构。它在各团队之间提供按公平份额（fair-share）的调度，并将 GPU 拓扑感知集成到配额分配决策中。

与 Kueue（作为可与任何底层调度器配合工作的准入控制层）不同，KAI Scheduler 是一个完整的调度器替代方案，在单一组件中同时处理配额强制执行和 Pod 放置决策。这使得配额策略与 GPU 特定调度优化（例如 GPU 分数分配和 Multi-Instance GPU (MIG) 支持）之间能够实现更紧密的集成。

然而，这种紧密耦合意味着采用 KAI Scheduler 需要完全替换默认的 Kubernetes 调度器。Kueue 可以与默认的 kube-scheduler 协同工作，也可以分层叠加在 KAI Scheduler 这类专用调度器之上，从而构建出由 Kueue 管理配额和准入控制、而 KAI Scheduler 优化 GPU 放置的架构。

## Volcano

Volcano 调度器通过 Queue CRD 实现了自己的队列抽象和配额管理，这些 CRD 定义了资源限制和优先级，为 Kueue 的 ClusterQueue/LocalQueue 模型提供了另一种选择。Volcano 队列包含回收（reclaim）策略，当更高优先级的工作负载到达时，可以抢占低优先级作业；还包含按配置权重在队列间分配集群容量的比例资源分配。

与 Kueue（将准入控制与调度机制分离，并作为调度器之上的一个层工作）不同，Volcano 在单个调度器组件中同时实现了队列管理和组调度（gang scheduling）。这提供了一种更单体化的方法，需要完全替换 kube-scheduler，但可以在配额策略与调度决策之间提供更紧密的集成。

## 做出正确选择
