
```yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: ClusterTrainingRuntime
metadata:
  name: my-torch-distributed-runtime
  labels:
    trainer.kubeflow.org/framework: torch
spec:
  mlPolicy:
    numNodes: 1
    torch:
      numProcPerNode: auto
  template:
    spec:
      replicatedJobs:
      - name: node
        template:
          metadata:
            labels:
              trainer.kubeflow.org/trainjob-ancestor-step: trainer
          spec:
            template:
              spec:
                containers:
                - name: node
                  image: pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
```

将其替换为`TrainingRuntime`即可创建命名空间级的训练运行时。

数据科学家使用此名称为其作业选择所需的运行时。

SDK 使用该标签来指导`TrainJob`的配置。

`spec`可以为大部分值定义默认值；例如，这里就表示作业只能使用一个节点。

管理员可能希望通过用自定义镜像替换该值，来控制集群中使用的镜像。该镜像是特定于 GPU 的，因此在本例中适用于 NVIDIA CUDA。

集群配置完成且`TrainingRuntime`可用后，平台管理员的工作就完成了。数据科学家现在可以专注于创建训练作业（示例 6-5）。

**示例 6-5. 使用 Hugging Face TRL 作为 CustomTrainer 的 Trainer 函数**

```python
def my_custom_trainer(**kwargs):
    from datasets import load_dataset
    from transformers import AutoTokenizer, set_seed
    from trl import SFTTrainer
    # It is not mandatory to set a fixed seed but it is useful for reproducibility
    set_seed(kwargs["seed"])
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        ...,  # kwargs[...]
        use_fast=True
    )
    # Load datasets
    train_dataset = load_dataset(
        ...,  # kwargs[...]
    )
    # Initialize Trainer
    trainer = SFTTrainer(
        model=...,
        args=...,
        train_dataset=train_dataset,
        eval_dataset=...,
```

## 第 6 章：模型定制

### 在 Kubernetes 上运行微调任务

![](img/3a93443ca88ca3c8ef163f48ebd612bc_233_0.png)

```python
    peft_config=...,
    processing_class=tokenizer,
)
trainer.train()
trainer.save_model(
    ..., # kwargs[...]
)
```

自定义训练器函数必须是自包含的，因此`import`必须是函数体的一部分。此示例基于 Hugging Face 库：`datasets`用于训练数据集（也可用于评估数据集），`transformers`用于分词器，`trl`用于实际的训练器类。请注意，这里没有 Kubeflow Trainer 特有的代码；该函数是一个普通的 Python 训练函数，可以直接调用。

分词器与正在微调的模型相关。使用支持并发工作的快速分词器非常重要，以避免训练过程变慢。

数据集包含模型在微调期间应学习的新知识。它可以是公开可用的数据集，但更可能是自定义数据集。

`SFTTrainer`的初始化与之前的示例相同。在这里，你可以选择模型、指定数据集，并配置 PEFT 技术（例如`LoraConfig`）。

`train()`方法初始化训练过程。硬件配置（如 GPU 和 worker 的数量）不在这里指定；而是在创建作业时定义（参见示例 6-6）。

当你调用`client.train(func=my_custom_trainer)`时，SDK 会对你的 Python 函数进行序列化，并将其嵌入到`TrainJob`自定义资源中。`TrainingRuntime`的基础容器镜像（预装了 PyTorch、Transformers、PEFT）会在运行时对你的函数进行反序列化并执行。这与传统的 Kubernetes 工作流不同：你无需构建或推送自定义镜像，只需在修改函数后重新运行 SDK 命令即可。

其权衡之处在于，基础镜像必须已经包含你的所有依赖项，而且你的函数必须是可序列化的（`import`必须引用已安装的包，不能有复杂的闭包）。

在训练器函数中定义了训练逻辑和配置后，你现在可以使用 Kubeflow Python SDK 创建`TrainJob`。

**示例 6-6：通过 Kubeflow SDK 创建 TrainJob**

```python
from kubeflow.trainer import CustomTrainer, TrainerClient

client = TrainerClient()
torch_runtime = client.get_runtime("my-torch-distributed-runtime")
job_name = client.train(
    trainer=CustomTrainer(
        # The custom trainer function is injected here with its parameters
        func=my_custom_trainer,
        func_args=..., # load_args()
        num_nodes=8,
        resources_per_node={
            "cpu": 4,
            "memory": "64Gi",
            "nvidia.com/gpu": 1,
        },
    ),
    runtime=torch_runtime,
)
client.wait_for_job_status(name=job_name, status={"Running"})
_ = client.get_job_logs(job_name, follow=True)
# It is possible to get all the steps and the status for each of them
# steps = client.get_job(name=job_name).steps
# client.delete_job(job_name)
```

在此示例中，使用上一个示例中定义的自定义函数创建了`CustomTrainer`，并由`TrainerClient`提交`TrainJob`。

在提交作业时指定硬件要求。硬件要求与模型大小以及你所执行的微调类型直接相关。在此示例中，该值已用于使用 PEFT LoRA 定制 Meta-Llama-3.1-8B-Instruct 模型。

客户端可以等待特定的作业状态。这是一个阻塞调用；在示例中，代码使用它来等待作业进入运行状态。你还可以获取日志或配置 TensorBoard 的使用。TensorBoard 是一个可视化工具包，最初来自 TensorFlow 项目，现在兼容多个库，包括 PyTorch。

虽然你可以随时删除作业，甚至在作业运行期间也可以删除，但此操作也会从 Kubernetes 中移除`TrainJob`对象及其关联的元数据。如果你没有使用外部实验跟踪，请考虑保留已完成的作业，以维护训练运行的记录。

### 一些 YAML 魔法

像示例 6-6 中的训练作业需要许多参数（超过 10 个），如果作业创建代码在 Jupyter Notebook 中运行（可能使用 Kubeflow Notebooks 组件），则可以使用`yamlmagic`库轻松配置所有参数。

这个项目是一个 Python 模块，可以通过`pip install yamlmagic`在 notebook 中安装，并通过`%load_ext yamlmagic`加载；之后，可以使用以`%%yaml my_params`开头的代码块来初始化一个变量（如`my_params`）。该代码块第一行之后的每一行都会被解析为 YAML，`my_params`就变成一个可供使用的 Python 字典。示例 6-7 展示了如何在 Jupyter Notebook 中使用`yamlmagic`配置训练参数。

**示例 6-7：在 Jupyter Notebook 中使用 `yamlmagic` 进行训练配置**

```python
# In a Jupyter Notebook cell
%load_ext yamlmagic
%%yaml training_config
model_name: meta-llama/Llama-3.2-3B
dataset: openai/gsm8k
num_epochs: 3
learning_rate: 2.0e-4
output_dir: /mnt/models/llama-gsm8k
# Now use the config with Kubeflow SDK
from kubeflow.trainer import TrainingClient

client = TrainingClient()
client.train(
    name="llama-math-tuning",
    model=training_config["model_name"],
    dataset=training_config["dataset"],
    num_epochs=training_config["num_epochs"],
    learning_rate=training_config["learning_rate"],
    output_dir=training_config["output_dir"]
)
```

在 LoRA 示例中，训练过程不会产生一个全新的完整模型。相反，每个保存的检查点都是一个 LoRA 适配器，可以在运行时与基础模型动态组合。这支持高效地服务多个微调模型，如图 6-5 所述。虽然这对于高效服务来说不是最佳方法，但为了测试目的，可以将 LoRA 适配器与基础模型合并，创建一个新的独立模型。该场景在示例 6-8 中介绍。

**示例 6-8：将 LoRA 适配器与基础模型合并**

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained(
    ...,
    device_map="cuda"
)
finetuned_path = "/opt/app-root/Meta-Llama-3.1-8B-Instruct/checkpoint-100/"
model = PeftModel.from_pretrained(base_model, finetuned_path)
merged_model = model.merge_and_unload()
merged_model.save(...)
```

必须首先加载基础模型。`device_map`参数使模型直接加载到 GPU 上。

必须知道 LoRA 微调模型的存储路径。每个训练轮次后都会创建一个新的检查点（也称为模型候选），在此示例中选择了第 100 个检查点。

在基础模型和微调层一起加载后，可以使用`merge_and_unload()`方法进行合并，从而获得新模型。

示例 6-8 从管理员和数据科学家的角度涵盖了 Kubeflow Trainer 项目的使用，展示了他们如何在一个分布式模型定制项目上进行协作。所显示的检查点路径（如`/opt/app-root/Meta-Llama-3.1-8B-Instruct/checkpoint-100/`）需要持久存储基础设施，才能超越临时训练作业的生命周期而存留下来。有关训练工作负载存储解决方案的全面介绍，包括持久卷、对象存储和分布式文件系统，请参见第 249 页的“训练存储”。然而，这只是第一步。

从平台的角度来看，调度这些长时间运行、资源密集型的工作负载需要额外的优化，以确保集群的公平使用，并防止硬件（尤其是 GPU）利用率不足。这里未涉及的一个重大挑战是组调度。分布式工作负载通常要求系统

![](img/3a93443ca88ca3c8ef163f48ebd612bc_237_0.png)

需要同时部署其所有 Pod 才能正确运行。这是一种“全有或全无”的语义。第 7 章将完全聚焦于这些平台优化，其中包含一个专门讨论 gang scheduling 的章节。

对于数据科学家来说，体验更加简单，因为 Kubeflow 生态系统让他们能够专注于模型定制生命周期，而对底层 Kubernetes 平台的感知有限。

Kubeflow 项目包含众多组件，以支持完整的 MLOps 或 LLMOps 生命周期。Kubeflow 模型注册表已在第 54 页的“Kubeflow Model Registry”中介绍，重点讨论了模型元数据管理，而本章将介绍 Kubeflow Trainer，以实现分布式训练任务。

数据科学家可以通过利用另外两个 Kubeflow 组件来开发和管理微调示例中包含的 Python 代码：Kubeflow Notebooks 和 Kubeflow Pipelines。

Kubeflow Notebooks 管理 Jupyter 等基于 Web 的 IDE 的基础设施，使数据科学家能够轻松自助配置环境，并试验 Kubeflow Trainer SDK。

在试验并定义训练任务之后，数据科学家可以使用 Kubeflow Pipelines 将笔记本转换为可复现的流水线。这使得该逻辑可以被多次执行以重新训练模型，既可以将代码提取为不同的步骤，也可以直接将笔记本纳入流水线中。

### 其他框架

虽然 Kubeflow Trainer 为大多数用例提供了全面的解决方案，但生态系统中还有几个值得考虑的替代方案。

Kubeflow Trainer 项目采用 Kubernetes 原生的方法来管理分布式训练任务的生命周期，使平台管理员和数据科学家都能使用他们偏好的工具。

虽然 Kubeflow Trainer 和 Hugging Face 的 TRL 为 Kubernetes 上的分布式训练提供了强大且以平台为中心的解决方案，但其他几个项目和库（如 DeepSpeed 和 Unsloth）提供了专门的工具来优化微调过程，尤其侧重于 LLM 的效率、速度和资源管理。此外，带有 KubeRay 的 Ray 提供了一种在 Kubernetes 上编排分布式训练工作负载的替代方法。

#### DeepSpeed

DeepSpeed 是一个深度学习优化库，它封装了 PyTorch 以简化训练任务的管理。将 DeepSpeed 与 Kubeflow Trainer 结合使用与之前的示例非常相似。你只需选择一个兼容 DeepSpeed 的`TrainingRuntime`（例如默认的 DeepSpeed 分布式运行时），并更新自定义的训练器逻辑（示例 6-9）。

**示例 6-9. 使用 DeepSpeed 的训练器函数**
