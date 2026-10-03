
NVIDIA 的 Blackwell GPU 架构为大模型带来了前所未有的推理速度。在早期的 MLPerf 测试中，搭载 Blackwell GPU 的服务器打破了[记录](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-massive-performance-leaps-in-mlperf-inference-v5-0/#:~:text=Additionally%2C%20in%20this%20round%2C%20Blackwell,FP8%E2%80%93while%20meeting%20benchmark%20accuracy%20requirements)。NVIDIA 随后将其基于 ARM 的 Grace CPU 与 Blackwell GPU 结合，创造了 Grace-Blackwell 超级芯片。

接下来，NVIDIA 将 36 个 Grace-Blackwell 超级芯片（每个包含 1 个 Grace CPU 和 2 个 Blackwell GPU）连接到一个名为 GB200 NVL72 的超密集机架部署中。这些超级芯片通过第 5 代 NVLink 交换机连接，以实现快速数据共享。借助该系统，NVIDIA 在 GB200 NVL72 上运行 4050 亿参数的 Llama 3.1 模型时，[实现](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-massive-performance-leaps-in-mlperf-inference-v5-0/#:~:text=Additionally%2C%20in%20this%20round%2C%20Blackwell,FP8%E2%80%93while%20meeting%20benchmark%20accuracy%20requirements)了每个 GPU 3.4 倍的吞吐量提升，相比上一代基于 Hopper 的系统。

Blackwell 的改进，如第二代 Transformer Engine 和新的 FP4 精度，是关键。通过利用 FP4 数据类型进行量化推理，基于 Blackwell 的 GB200 NVL72 系统每个 GPU 的数学吞吐量相对于 FP8 精度翻了一番。

关键在于，基准测试的准确性保持在目标范围内，因此这种性能与精度之间的权衡基本上是中性的。在系统层面，收益不断累积。如图 5-6 所示，GB200 NVL72 机架在 Llama 405 推理任务上提供的吞吐量比类似的基于 Hopper 的集群高出 30 倍。

![带有绿色柱状图的图形，AI 生成的内容可能不正确。](img/ch10_figure_6_1744914898457752.png)

**图 5-6.** NVIDIA 的 Grace-Blackwell "GB200 NVL72"超级集群（深绿色）提供的 LLM 推理吞吐量比上一代（浅绿色）高出 30 倍，这得益于 9 倍的 GPU 数量和显著的每 GPU 加速（来源：[NVIDIA Blackwell 在 MLPerf 推理 v5.0 中带来巨大性能飞跃 | NVIDIA 技术博客](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-massive-performance-leaps-in-mlperf-inference-v5-0/)）。

这归因于多个因素，包括单个 GPU 的速度提升、更快的 NVLink/NVSwitch 互连，以及 72 个 GPU 在机架内近距离协同工作的庞大集群规模。当时，GB200 NVL72 为大模型的延迟和大规模吞吐量树立了新的行业标准，"the 405"即是对 Llama 4050 亿参数模型的一种称呼。

**提示：** Meta 首席执行官马克·扎克伯格将 Llama 4050 亿参数模型随口称为"the 405"，这是借鉴了加利福尼亚州居民[称呼](https://www.pbssocal.org/shows/lost-la/the-5-the-101-the-405-why-southern-californians-love-saying-the-before-freeway-numbers)高速公路的方式。具体来说，[加利福尼亚人](https://www.youtube.com/watch?v=dCer2e0t8t8)会在高速公路编号前加上"the"（例如"the 405"高速公路）。

此外，值得注意的是，即便是较小的 8-GPU Blackwell 服务器（称为 DGX B200），在 Llama-70B 聊天模型上的吞吐量也比类似的 8-GPU Hopper 服务器高出 3.1 倍。这证明了 Blackwell 的架构优势——即使在 GPU 数量相同的情况下。

AI 系统性能工程师学到的经验是：诸如 FP4 Tensor Core 等新硬件特性——以及 GPU 之间更高的 NVLink 带宽——与 GB200 NVL72 等设计所实现的更高程度模型并行相结合时，可以带来数量级的性能提升。

从 Hopper 升级到 Blackwell 立即实现了投资回报：基于 Hopper 的 GPU 推理响应需要 6 秒，而 Blackwell GPU 只需 2 秒。因此，采用相同硬件配置的 Blackwell GPU 集群可以服务更多用户。投资 Grace-Blackwell 平台带来了渐进式改进，以及在服务能力上的显著阶梯式增长。对于任何应对不断增长的模型规模和用户需求的 AI 提供商而言，这都是一个竞争优势。

## 借助 Grace-Blackwell 集群加速万亿参数大语言模型的实验与洞察

NVIDIA 训练万亿参数模型的参考设计是 GB200 NVL72 Grace-Blackwell 机架。这是一个功能强大的服务器机架，在一个内存一致性域内整合了 72 个 Blackwell GPU 和 36 个 Grace CPU。

[基准测试](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-massive-performance-leaps-in-mlperf-inference-v5-0/#:~:text=The%20Blackwell%20platform%20is%20available,connected%20using%20NVLink%20and%20NVSwitch) 表明，对于拥有 4050 亿参数的巨大模型，GB200 NVL72 的训练速度比前几代产品更快。Adrian Cockcroft 是一位以在云计算和系统架构领域的变革性工作而闻名的杰出技术专家，他的独立分析[认为](https://adrianco.medium.com/deep-dive-into-nvidia-blackwell-benchmarks-where-does-the-4x-training-and-30x-inference-0209f1971e71#:~:text=There%20are%20three%20generations%20of,been%20as%20impressive%2C%20and%20users)，将配置齐全的 Grace-Blackwell 与基于 Hopper 的高端集群相比，实现 4× 加速是可能的。

这一性能提升的关键因素包括：Blackwell 更高的单 GPU FLOPs、Grace 增加的 CPU 内存容量、更快的 NVLink-C2C 互连，以及单机架系统的庞大规模。具体来说，如果一个模型以前在 Hopper GPU 集群上训练需要 10 天，那么新的 Blackwell 集群大约 2–3 天就能完成。这大大加速了获得洞察的时间，并提高了实验的频率。

AI 系统性能工程师学到的经验是：扩展的成效与投入的硬件数量并不总是线性相关。要实现高效扩展，必须消除各个层面的瓶颈。Grace-Blackwell 设计同时解决了多个瓶颈，包括更快的 GPU、专用 Tensor Core 上更快的矩阵运算、更大的单 GPU 内存以支持更大的批次规模，以及改进的 NVLink 和 NVSwitch 架构，以低延迟连接数十个 GPU。所有这些都带来了更好的扩展效率。过去难以充分利用 72 个 H100 GPU 的训练任务和推理服务器，如今在 NVL72 配置中的 72 个 Grace-Blackwell GPU 上实现了近乎线性的扩展效率，因为通信开销大大降低。

使用 Grace-Blackwell 进行训练工作负载的投资回报（ROI）体现在更快的训练运行和更快速的实验上。例如，假设一家企业使用自己的数据集训练一个 5000 亿参数的大语言模型。使用 GB200 NVL72 系统，可以将训练时间从 4 周缩短到 1 周。这可能会带来变革性的影响，因为模型能更快投入生产，并且每次获得洞察所消耗的能源也更少。

因此，尽管 GB200 集群的前期成本高昂，但从每个训练模型和每次获得洞察的角度来衡量，组织能以加速研发和可能更低的成本获得价值。总之，这个案例证明了采用下一代硬件进行大规模大语言模型训练的 ROI，包括更快的实验结果和更好的每美元吞吐量。这使得组织和研究实验室能够在不大幅增加成本的情况下推进 AI 前沿。

## HPE 面向万亿参数时代的 Grace-Blackwell 超级计算机

2025 年初，HPE（慧与）[出货](https://www.hpe.com/us/en/newsroom/press-release/2025/02/hpe-announces-shipment-of-its-first-nvidia-grace-blackwell-system.html#:~:text=HOUSTON%20%E2%80%93%C2%A0%20FEBRUARY%2013%2C%202025%C2%A0,to%20optimize%20efficiency%20and%20performance)了其首款 NVIDIA Grace-Blackwell 系统——HPE Cray XD。这是他们对 GB200 NVL72 机架规格的实现。这标志着该技术成为 NVIDIA 自有实验室之外的首批实际部署之一。HPE Cray XD 机架级系统专为需要使用单一统一内存空间训练和服务 1 万亿参数规模模型的组织而构建。

HPE 强调，这类系统为超大规模模型提供了更低的每 token 训练成本和一流的吞吐量（[HPE 宣布出货其首款 NVIDIA Grace Blackwell 系统 | HPE](https://www.hpe.com/us/en/newsroom/press-release/2025/02/hpe-announces-shipment-of-its-first-nvidia-grace-blackwell-system.html#:~:text=%E2%80%9CAI%20service%20providers%20and%20large,leading%20services%20expertise.%E2%80%9D)）。这意味着，尽管其机架的绝对成本很高，但当模型达到万亿规模时，以每投资一美元每秒处理的 token 数衡量的效率要高得多。目标用户是云 AI 服务提供商和大型企业研究团队。

首批使用 HPE Cray XD 的 HPE 工程师学到的经验是，将所有 72 个 GPU 放在一个域中，并行训练任务的调试和优化比传统的每节点 8 个 GPU 集群更容易，因为网络通信模式中的可变部分更少。然而，他们也了解到了 NVL72 系统特有的故障模式，例如 NVLink/NVSwitch 结构中的故障。采用 NVL72 机架设计后，这类故障可能会同时影响许多 GPU。而以前，一条故障的 InfiniBand 链路只会影响一个节点。

对于这些系统的先驱客户来说，采用 NVL72 机架设计的 ROI 是一种竞争优势。HPE 及其客户现在可以将大规模模型装入内存，并且训练速度远快于竞争对手。此外，通过将 Cray XD 作为单一集成解决方案交付，客户的价值实现时间大大缩短。组织现在可以在数天内启动一台 1 exaFLOP 的 AI 超级计算机并投入工作，而不是数月或数年。快速的部署时间使 HPE 的客户能够为大型 AI 模型购买交钥匙解决方案。这降低了他们的投资风险，加快了实现 ROI 的速度，并带来了巨大的 AI 系统性能突破。

## 训练并服务 100 万亿参数模型

设想一个（并不遥远的）未来，某个 AI 研究实验室尝试一项非同寻常的任务，比如训练一个稠密的 100 万亿参数 Transformer。这大约是目前以数百亿或数千亿参数计的大模型的 100 倍。如何才能借助 Grace-Blackwell（GB200）超级集群实现这一目标？

在这一假设情景中，实验室组建一个由多个 GB200 NVL72 单元组成的训练集群——例如 8 个机架，总计 576 个 Blackwell GPU 和 288 个 Grace CPU 通过网络连接在一起。即使使用 Blackwell 每个 GPU 高达 192 GB 的 HBM 内存，一个 100 万亿参数的模型也几乎无法装进这 8 机架系统的 GPU HBM 内存总和之中——即使采用降低的 8 位精度也不行。

例如，在 8 位精度下，100 万亿参数将需要 100 TB 或 100,000 GB 的存储空间。100,000 GB / 每 GPU 192 GB HBM = 520 个 GPU。我们在一个 8 机架系统中只有 576 个 GPU，因此模型可以加载，但完全在 GPU HBM RAM 中训练是不可能的，因为训练还需要额外的梯度、激活和优化器状态，这些会带来巨大的额外开销——超出了单个机架的容量。

由于 100 万亿参数的模型无法装入 8-NVL72 机架超级集群，因此团队需要利用 Grace CPU 内存池，根据需要将 GPU 内存溢出到其中。这要求系统通过机架内的 NVLink 5 互联或机架间的 InfiniBand 在 CPU RAM 中流式加载和卸载权重。当然，你会希望尽可能避免跨机架通信。

人们几乎肯定会采用张量并行和流水线并行来将庞大的网络划分到多个 GPU 上。另一种选择是将模型拆分到……例如，采用流水线并行时，每个 GB200 机架负责 1/8 的层。然而，这需要层间进行跨机架通信——从而降低推理和训练的速度。

请记住，在机架内部，72 个 GPU 全部以完整的 NVLink 带宽互连，并以高速协同处理模型各自的部分。然而，机架之间由较慢的 InfiniBand 链路连接各个 NVLink 域。工程师们深知这种层级结构，会设计高效、流水线化且并行的计算/通信调度。例如，当一个机架完成给定批次的各层计算后，它会将中间结果（激活值）传递给下一个机架的层进行处理。与此同时，第一个机架开始计算下一批次——如此循环。

通过结合数据并行、流水线并行和张量并行（3D 并行），可以确保尽管存在跨机架通信延迟，所有 GPU 都能保持忙碌。为了进一步减轻压力，可以在传输过程中将激活值量化到`FP8`，并应用高效的 all-reduce 梯度更新算法，从而在每一步最小化跨机架同步。这样可以避免分布式训练中可怕的“停止世界”同步屏障。

还可以考虑采用混合专家（MoE）方法，将 100 万亿参数的模型拆分为 100 个独立的 10 万亿参数专家模型。这意味着在处理任意一个 token 时，只有一部分专家（机架）处于活跃状态。这将显著减少每一步的计算量和通信量。

然而，使用 MoE 还会带来额外的挑战，包括网络的门控、将 token 路由到正确的专家，以及确保专家间的负载均衡。可以使用动态专家扩展等技术，让所有机架都得到均匀利用，避免网络热点——即使在不同专家被激活时也是如此。

在软件方面，需要像 NVIDIA `Dynamo`这样的编排层，与`TensorRT-LLM`或`vLLM`服务引擎配合，来协调系统中大量 GPU 的工作。它可以为不同的流水线阶段分配微批次，通过重新映射 GPU 来处理 GPU 故障，并优化集群中大规模 KV 缓存的复用。

为 100 万亿参数的模型提供推理服务将是一项浩大的工程。团队可能会利用`Dynamo`的解耦功能，将模型权重放在 CPU 内存中，并仅针对特定查询将必要的专家/层加载到 GPU 内存中。在这种情况下，系统需要使用不同机架上计算出的部分结果来聚合生成响应。

团队还会大力依赖 4 位权重量化进行推理，利用 Blackwell 的`FP4`支持，将活动模型占用空间缩小到约 50 TB。这仍然很大，但在多机架系统的能力范围之内。

这个思想实验的教训凸显了需要哪些超大规模性能技巧，包括层次化并行、激进的量化与稀疏化，以及精细的调度。否则，这将是一个棘手的问题。模型的内部神经网络架构和物理集群网络必须协同设计，使得每一个延迟和带宽的考量都成为首要关注点。

成功训练一个 100 万亿参数模型的假想投资回报率将是开创性的。这样的模型可能展现出无与伦比的能力，在准确性上实现巨大飞跃，或者处理极其复杂的人工通用智能（AGI）或人工超级智能（ASI）任务。

然而，成本和风险也非常高。训练可能会在 GPU 计算时间上花费数千万甚至数亿美元。工程团队必须确保 576 个 GPU 中的每一个在整个过程中都得到最大程度的利用。任何低效都可能浪费计算时数和资金。在相对较小规模上得到验证的技术——例如 DeepSeek 在性能受限的 NVIDIA H800 训练集群上训练 V3 模型时使用的技术——在这个 100 万亿参数规模下将是必不可少的。

如果做得好，回报将是一个首创的 AI 系统和史册留名的地位。如果做得不好，可能会烧掉巨额财富而收效甚微。因此，这个假想案例凸显了性能工程师面临的现实——在极端规模下，规划、优化、性能剖析和迭代是让不可能变为可能的唯一途径。其理念是将超大规模的硬件投资高效地转化为前所未有的 AI 能力。

## 关键要点

### 协同设计的硬件与软件优化

大语言模型的性能提升真正来源于硬件与软件紧密集成、协同设计的创新突破。

### 数值精度进步

使用更新的数值精度模式，如 Hopper 上的 8 位和 Blackwell 上的 4 位，可以在不牺牲准确性的情况下显著提升训练速度。这表明，即使数字表示方式的微小改变也能带来巨大的性能收益。

### 计算与数据传输的重叠

多个案例研究强调了将计算与数据传输重叠的算法如何帮助抵消较低内存带宽等硬件限制。

### 高级调度与批处理
