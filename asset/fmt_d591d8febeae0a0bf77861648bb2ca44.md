
# 使用 GB200 超级芯片的旗舰配置

使用 GB200 超级芯片的旗舰配置就是 NVIDIA 所称的 NVL72 系统，它本质上是一台封装在单个机架中的 AI 超级计算机。NVL72 代表一个拥有 72 个 Blackwell GPU——以及 36 个 Grace CPU——全部通过 NVLink 互连的系统。

GB200 NVL72 由 18 个计算节点（每个均为 1U 尺寸）构建而成，每个节点包含两个 GB200 超级芯片，因此每个计算节点共有 4 个 Blackwell GPU + 2 个 Grace CPU，如图 2-5 所示。

![一张计算机芯片特写，AI 生成的内容可能不正确。](img/ch02_figure_5_1744914895521591.png)

**图 2-5.** GB200 NVL72 机架内的一个 1U 计算托盘，带有两个 Grace-Blackwell 超级芯片，每个超级芯片模块包含一个 Grace CPU 和两个 Blackwell GPU 芯片。NVL72 有 18 个这样的托盘连接在一起（来源：[developer.nvidia.com](https://developer.nvidia.com/blog/nvidia-gb200-nvl72-delivers-trillion-parameter-llm-training-and-real-time-inference/#:~:text=The%20GB200%20compute%20tray%20is,7%20TB%20of%20fast%20memory)）。

通过将 18 个计算节点连接在一起，GB200 NVL72 将 72 个 Blackwell GPU（18 个节点 × 4 个 GPU）和 36 个 Grace CPU（18 个节点 × 2 个 CPU）互联，形成一个强大、统一的 CPU-GPU 集群。NVL72 的非凡之处在于，每个 GPU 都能以极高的速度与任何其他 GPU 通信，仿佛全部 72 个 GPU 都位于同一块主板上。NVIDIA 通过 GPU 上的 NVLink 5 连接与名为 NVSwitch 的专用交换芯片相结合实现了这一目标。

## NVLink 与 NVSwitch

每个 Blackwell GPU 拥有 18 个 NVLink 5 端口，每个端口支持 100 GB/s 的双向数据传输，或单向 50 GB/s。综合来看，单个 GPU 通过 NVLink 与对等 GPU 之间可以传输高达 1.8 TB/s（18 个 NVLink 端口 × 100 GB/s）的数据。这是上一代每 GPU NVLink 带宽的两倍，因为 Hopper H100 使用的是 NVLink 4，其双向 900 GB/s 的速度仅为 NVLink 5 的一半。

GPU 通过 NVSwitch 芯片以网络方式互联布线。NVSwitch 本质上是一种与网络交换机类似的交换芯片，但它是专为 NVLink 打造的。这意味着任何 GPU 都可以通过某个 NVSwitch 芯片以全带宽单跳访问任何其他 GPU。图 2-6 展示了 NVL72 中使用的 NVLink Switch 托盘。

![一张设备特写，AI 生成的内容可能不正确。](img/ch02_figure_6_1744914895521612.png)

**图 2-6.** 来自 NVL72 的一个 NVLink Switch 托盘（来源：[developer.nvidia.com](https://developer.nvidia.com/blog/nvidia-gb200-nvl72-delivers-trillion-parameter-llm-training-and-real-time-inference/#:~:text=Fifth,System)）。

每个交换托盘包含两颗 NVSwitch 芯片（图中可见的大芯片）和多个高速端口（蓝色线缆代表 NVLink 连接）。在 NVL72 机架中，如图 2-7 所示，9 个这样的交换托盘提供了完全连接 72 个 Blackwell GPU 的交换结构。

![一个带有黑色背景的灰色矩形物体，AI 生成的内容可能不正确。](img/ch02_figure_7_1744914895521633.png)

**图 2-7.** NVL72 机架内 9 个托盘的 NVSwitch 系统

每个交换托盘包含两颗 NVSwitch 芯片，因此整个系统共有 18 颗 NVSwitch 芯片。该网络采用全交叉开关（full crossbar）布局，每个 GPU 都与每个 NVSwitch 相连，每个 NVSwitch 也与每个 GPU 相连，从而在任意一对 GPU 之间提供高带宽路径。

具体来说，每个 GPU 使用其 18 个 NVLink 端口连接到 18 颗 NVSwitch 芯片（每个交换机一条链路）。这意味着任意 GPU 最多经过两次跳转（GPU → NVSwitch → GPU）即可到达另一个 GPU，沿途带宽极为可观。

整个 72-GPU 网络的带宽，即聚合对分带宽（aggregate bisection bandwidth），在机架内约为 130 TB/s。打个比方，这比类似规模的顶级 InfiniBand 集群还要高出许多倍。这种设计基本上将整个机架变成了一台巨大的共享内存机器。

## 多 GPU 编程

从编程模型的角度来看，一个 GPU 可以直接通过 NVLink 从另一个 GPU 的 HBM 内存执行内存加载，甚至可以从另一个节点的 Grace CPU 加载，并在微秒级时间内获取数据。换句话说，这些 GPU 可以在整个机架的全局共享内存空间中运行，非常类似于单台服务器内核心共享内存的方式。这就是为什么 NVL72 常被描述为“一个大 GPU”或“机架中的 AI 超级计算机”。

RDMA（远程直接内存访问）等软件库通过提供简洁的编程接口（如跨 GPU 的原子操作），屏蔽了全局共享内存空间的复杂性。分布式训练和推理工作负载需要在多个 GPU 之间频繁同步和交换信息。传统上，GPU 位于不同的计算节点和机架中，因此同步通过相对较慢的网络链路（如 InfiniBand 和 Ethernet）进行。在扩展到多个 GPU 以支持大型 AI 模型时，这往往是瓶颈。

借助 GB200 NVL72 系统，这些数据交换通过 NVLink 和 NVSwitch 以极快的速度完成。这意味着你可以将训练任务或推理集群扩展到 72 个 GPU，且通信开销极小。由于 GPU 花在等待彼此数据上的时间大大减少，整体吞吐量在扩展到 72 个 GPU 时接近线性增长。相比之下，考虑在由 9 台独立计算服务器（每台配备 8 个 Hopper H100 GPU）组成的相似规模 72-GPU H100 集群上运行相同的任务。这种配置需要 InfiniBand，这将产生网络瓶颈，从而大幅降低集群的扩展效率。

让我们用具体数字来分析和比较 GB200 NVL72 与 72-GPU H100 集群。在单个 NVL72 机架内，GPU 到 GPU 的带宽约为 100 GB/s 以上，小消息的延迟约为 1–2 微秒。在传统 InfiniBand 网络中，每个 GPU 的带宽可能约为 20–80 GB/s（取决于网卡数量及其速率），延迟可能为 5–10 微秒甚至更高。就节点间 GPU 通信而言，NVL72 网络比最好的 InfiniBand 网络具有更高的吞吐量（每个 GPU 提升 2 倍或更多）和更低的延迟（提升 3–5 倍）。在实际场景中，跨 GPU 聚合梯度的 `all-reduce` 集合操作在基于 InfiniBand 的 H100 集群上可能占用迭代时间的 20–30%，而在基于 NVLink 的 NVL72 集群上仅占用 2–3%。

NVIDIA 的一项真实测试证明了这一点，该测试显示，基于 NVL72 的单个系统比类似的基于 H100 的集群快约 4 倍，这主要归功于通信效率。关键要点是，在单个 NVL72 机架内，通信速度非常快，通信瓶颈几乎被完全消除，因此不再是优先考虑的问题。而在传统 InfiniBand 和 Ethernet 集群中，通信往往是主要瓶颈，需要在软件层面进行精心优化和调优。

总之，设计和实现软件时应充分利用 NVL72 的配置，尽可能将工作负载的通信保持在机架内部（“机架内”），以发挥高速 NVLink 和 NVSwitch 硬件的优势。只有在绝对必要时，才使用基于 InfiniBand 或 Ethernet 的较慢机架间（“机架间”）通信，以扩展超出 NVL72 的计算和内存资源。

## 基于 NVIDIA SHARP 的网络内聚合

另一个硬件加速优化是 [NVIDIA SHARP](https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing/)，全称是 Scalable Hierarchical Aggregation and Reduction Protocol（可扩展分层聚合与归约协议）。SHARP 直接集成在 NVSwitch 硬件中，能够在网络硬件本身直接执行聚合操作（例如 `all-reduce`、`all-gather` 和 `broadcast`）。NVSwitch 交换结构直接合并部分结果，无需将数据经由 GPU 转发。通过将集合通信操作从 GPU 和 CPU 卸载到交换机硬件本身，SHARP 大幅提高了效率、降低了延迟，并减少了网络中传输的数据量。

SHARP 的效率提升意味着在分布式训练期间，聚合梯度或同步参数的重任由 NVSwitch 专用的 SHARP 引擎承担。结果是在机架内和机架间配置下都能实现更高效的扩展。即使 GPU 数量增加，SHARP 也能实现接近线性的性能提升。这种网络内计算能力对于训练超大规模模型尤其关键，因为在集合操作上节省的每一微秒都能转化为可观的整体加速。

## 多机架与存储通信

接下来，我们来讨论一个 NVL72 机架如何与另一个 NVL72 机架或外部存储系统（如共享文件系统）通信。正如我们前面所述，在 NVL72 机架内部，NVLink 覆盖了所有 GPU 到 GPU 的流量。但在机架外部，则依赖更传统的网络硬件。

NVL72 中的每个计算节点都配备了高速网络接口卡（NIC）和数据处理器（DPU）。在 GB200 NVL72 参考设计中，每个节点有 4 个 ConnectX InfiniBand 网卡和 1 个 BlueField-3 DPU。这 4 个 InfiniBand 网卡每个运行速率为 400 Gb/s。合计起来，这些网卡向外部发送和接收数据的速率约为 1.6 Tbit/s（1600 Gbit/s），如图 2-8 所示。

![图示展示了带有 InfiniBand 网络系统的 NVIDIA HGX Grace Hopper Superchip。每个 Grace Hopper Superchip 内部具有硬件一致性。每个 Superchip 通过 PCIe 连接到一个 BlueField 3 DPU，然后这些 DPU 以总计 100 GB/s 的带宽连接到 NVIDIA Quantum-2 InfiniBand NDR400 交换机。](img/ch02_figure_8_1744914895521653.png)

**图 2-8.** InfiniBand 与 BlueField DPU 网卡

在单个机架的 18 个计算节点中，若充分利用，机架外部总吞吐量接近 30 Tbit/s（1.6 Tbit/s × 18 个计算节点）的聚合网络容量。关键在于，NVIDIA 预见了被称为“AI 工厂”的多机架部署，因此他们确保 NVL72 可以通过每个节点上的这 4 个网卡接入更大的网络结构。

**提示**

NVIDIA 还提供基于 Ethernet 的解决方案，使用 Spectrum 交换机并采用融合以太网上的 RDMA（RoCE）作为互连替代方案，称为 Spectrum-X。

每个节点中的 BlueField-3 DPU 有助于卸载诸如 RDMA、TCP/IP 和 NVMe 存储访问等网络任务。这确保了 Grace CPU 不会因管理网络中断而陷入困境。DPU 本质上充当智能网络控制器，利用 NVIDIA 名为 GPUDirect 的 RDMA 软件，在网卡和 GPU 内存之间直接移动数据，无需 CPU 参与。这在从存储服务器流式传输大型数据集时尤其有用，因为 DPU 可以处理数据传输并直接将数据存入 GPU 内存，而 CPU 则专注于数据预处理等其他任务。

当扩展到多个 NVL72 机架时，NVIDIA 使用 Quantum 系列 InfiniBand 交换机。多个 NVL72 机架可通过这些 InfiniBand 交换机互连，形成大型 NVL72 机架集群，如图 2-9 所示。

![图示显示了 NVIDIA HGX Grace Hopper 与 NVLink Switch System。每个 Grace Hopper Superchip 内具有硬件一致性。在一个最多包含 256 个 Grace Hopper Superchip 的集群中，每个 Grace Hopper Superchip 通过 NVLink Switch System 相互连接。每个 Superchip 还通过 PCIe 连接 BlueField 3 DPU，然后以 100 GB/s 总带宽连接到 NVIDIA Quantum-2 InfiniBand NDR400 交换机。](img/ch02_figure_9_1744914895521673.png)

**图 2-9.** 使用 InfiniBand 扩展 NVL72 集群

例如，一个包含 8 个机架、总计 576 个 GPU 的 NVL72 集群可以通过第二层 InfiniBand 交换机构建，尽管跨机架的 InfiniBand 通信性能将低于单个机架内的 NVLink/NVSwitch 通信。

## 预集成机架设备

由于 NVL72 是一个复杂的系统，NVIDIA 将其作为预集成机架“设备”在单个机柜中交付。它组装好了所有 18 个计算节点、全部 9 个 NVSwitch 单元、内部 NVLink 线缆、电源分配和冷却系统。其理念是，组织可以将其作为一个整体订购，到达后即可投入使用。只需将机架连接到设施电源，接好水冷接口，将 InfiniBand 线缆连接到网络，然后开机即可！无需单独为 72 个 GPU 布线 NVLink，因为 NVIDIA 已在机架内为您完成。即使是液冷配置也是自包含的，我们稍后将讨论。

这种设备化方法加速了部署，并确保系统构建正确且经过 NVIDIA 验证。该机架还包含 NVIDIA Base Command Manager 集群管理软件，以及用于集群作业调度和编排的 SLURM 和 Kubernetes。换句话说，NVL72 机架的设计目标是直接集成到您的环境中。

## 共封装光学：网络硬件的未来

随着网络数据吞吐率攀升至 800 Gbit/s 甚至更高，传统的可插拔光模块方法开始触及功率和信号极限。这种方法在交换机处将电信号转换为光信号时会消耗大量功率。NVIDIA 正通过在其网络设备中集成共封装光学（CPO）来解决这一问题。

采用共封装光学，光发射器直接集成在交换机芯片旁边。这大幅缩短了电路路径，使得机架间能够实现更高带宽的链接，降低功耗，并提高整体通信效率。NVIDIA 正将 CPO 集成到计划于 2025 年底或 2026 年推出的 Quantum-3 InfiniBand 交换机中。

实际上，像 CPO 这样的技术正为将成百上千个机架（“AI 工厂”）连接成一个统一网络铺平道路，使机架间带宽不再成为瓶颈。这些光网络技术进步对于实现高性能的机架间带宽至关重要，以确保网络在超大规模下能够跟上 GPU 的发展。

总而言之，在 NVL72 机架内部，NVIDIA 使用 NVLink 和 NVSwitch 在 72 个 GPU 之间构建了一个极速的全互连网络。这些互连速度如此之快且均匀，使得 GPU 实际上像一个整体一样工作。在机架之外，高速 NIC（例如 InfiniBand 或以太网）将机架连接到其他机架或存储，并由 DPU 高效管理数据移动。

NVL72 是一个极其强大的独立系统，也是更大规模 AI 超级计算机的基本构建模块。NVIDIA 与 CoreWeave、Lambda Labs、AWS、Google Cloud、Azure 和 Equinix 等云服务提供商合作，提供基于 NVL72 的机架，企业可以快速租用或部署。

AI 工厂——由多个此类机架组成的大规模 AI 数据中心——的概念正在成为现实。NVIDIA 的硬件和网络路线图明确瞄准实现 AI 工厂愿景。简而言之，NVL72 展示了协同设计能走多远：GPU、网络和物理机架硬件携手构建，以尽可能无缝高效地扩展到成千上万个 GPU。

## 计算密度与功率需求

NVL72 机架在计算密度方面非常惊人，这意味着单个机架的功耗非常高。满载的 NVL72 在最大负载下可消耗高达约 120 kW 的功率。而 NVIDIA 上一代 AI 机架的功耗约为 50–60 kW。将 72 个尖端 GPU 以及所有支持硬件装入一个机架，已触及数据中心基础设施可处理的极限。

要为 NVL72 机架提供 120 kW 电力，不能只使用单一标准电源线路。数据中心通常会配置多条大容量电路来提供这种功率。例如，可以为机架输入两路独立电源以实现冗余，每路电源可承载 60 kW。正常运行时，负载在两路电源之间均衡分配。如果一路电源发生故障，系统可以削减部分负载或限制 GPU 性能，以保持在剩余电源的容量范围内。这种冗余对于防止电路烧断导致您长达数月的训练任务中断非常重要。

在机架内部，电力被分配到每个 1U 计算节点的电源。电力从 AC 转换为 DC，供本地电子设备使用。NVL72 中的每个计算节点包含 2 个 Grace-Blackwell 超级芯片，它们合计消耗约 5-6 kW 功率。18 个计算节点的总功耗约为 100 kW。NVSwitch 托盘、网络交换机和冷却泵约占 20 kW，因此整个 NVL72 机架总功耗为 120 kW。

在典型的数据中心电压（例如 415 V 三相交流电）下，电流巨大，因此所有设备都针对高电流进行了工程设计。运维人员必须仔细规划才能容纳这样的机架，这通常需要专用的电源分配单元（PDU）和仔细监控。电源瞬变也是一个考虑因素，因为 72 个 GPU 从空闲状态升至满功率时，可能在短短几毫秒内迅速消耗数十千瓦的功率。良好的设计应包含电容器或时序控制，以避免较大的电压跌落。
