
# 评估微调后的 GPT 模型

| 准确性 | 这在模型用于涉及事实信息任务时尤其重要，例如教育工具、信息机器人或任何提供正确信息至关重要的应用。准确性是通过模型响应与事实正确性和客观真理的一致性来衡量的。 |
| 语言流畅性 | 涉及模型输出的语法和句法正确性。即使模型高度相关、连贯和准确，较差的语言流畅性也会显著影响用户体验。这包括正确的语法、标点符号和风格，确保生成的文本正确且对最终用户来说读起来自然。 |

使用这些标准评估微调后的 GPT 模型涉及自动指标、人工审查和用户反馈的结合，确保模型满足其特定应用所需的高标准。

## 选择适当的指标

在微调模型时，选择合适的评估指标对于准确评估模型性能和改进至关重要 [1]。微调后，这些指标表明模型如何适应特定任务或领域。它们提供了对模型性能各个方面的见解，例如预测准确性、语言质量和特定任务的有效性。企业应尽可能寻找自动化的指标评估，并拥有一套定量和定性指标。

+ 定量指标：
  + 几个指标有助于衡量模型输出与人类参考文本之间的重叠。下一节将概述其中的一些（BLEU、ROUGE、METEOR 等）。
  + F1 分数评估了精确度和召回率之间的准确度权衡。
  + 困惑度评估生成文本的模型不确定性/信心。
  + 任务完成度用于目标导向的对话系统和成功任务解决的百分比。
+ 定性指标：
  + *流畅性*—评估输出的语法和可读性
  + *连贯性*—逻辑一致性和叙事流畅性
  + *简洁性*—避免重复和过多的文字
  + *事实准确性*—避免客观错误

指标的选择应与模型的预期应用相一致，无论是翻译、摘要、分类还是创意内容生成。如困惑度、BLEU 分数、ROUGE、F1 分数和人工评估等指标各自提供了对模型能力的独特视角，有助于确保对微调模型性能的全面和平衡评估。让我们更详细地看看每个指标：

+ *困惑度*—这是语言模型中的标准指标，用于量化模型预测样本的能力。它衡量语言模型在预测序列中的下一个标记时的不确定性 [2]。较低的困惑度分数表明模型在预测方面更加自信和准确。这在微调中尤为重要，因为它可以反映模型对新风格或文本领域的适应程度。它是评估语言生成任务改进的关键指标。
+ *BLEU 分数（双语评估辅助研究）*—BLEU 分数通过将其与参考翻译进行比较来评估机器翻译质量。它计算匹配的词组并基于这些匹配计算分数。较高的 BLEU 分数表明翻译质量更好，但它有局限性，可能无法捕捉语义准确性或流畅性 [3]。
+ *ROUGE（基于回忆的摘要评估的辅助研究）*—ROUGE 是用于自动摘要评估的指标。它通过衡量计算机生成的输出与参考摘要之间的重叠来评估摘要的质量。ROUGE 的不同变体提供了对摘要质量方面的见解 [4]。
+ *F1 分数*——F1 分数在分类任务（如情感分析和主题分类）中很有用。它平衡了精确度和召回率之间的权衡，提供了一个衡量模型在分类或分类文本中的准确性的单一指标。
+ *人工评估*——尽管自动化指标很有用，但人类判断仍然至关重要，特别是对于需要主观评估的任务，如故事生成、创意写作和对话代理。人工评估者可以提供关于文本的自然性、适宜性、创造力和甚至幽默或讽刺的细微差别等方面的见解。这种定性评估补充了定量指标，为模型性能提供了一个更全面的视角。

任务特定评估对于衡量模型在预期应用中的性能至关重要。它涉及根据任务使用不同的指标和考虑因素。例如，摘要模型使用 ROUGE 分数和人工摘要的连贯性和信息性评估进行评估。同样，问答模型根据给定问题的准确性和相关性进行评估。这种评估确保模型在一般指标上表现良好，并且对于其特定用例来说既有效又可靠。

## 错误分析

错误分析是评估过程中的一个关键组成部分，涉及对微调模型性能不佳的原因和地点的详细检查。这种分析有助于识别模型错误中的模式，这些模式可以按类别分解为语义错误、事实不准确或语言不一致。

例如，如果一个模型在理解某些类型的查询或生成包含事实错误的响应时持续出错，这将在错误分析中突出显示。理解这些错误模式对于进一步改进模型和进行有针对性的改进至关重要。它还有助于理解模型的局限性以及可能需要更多数据或更复杂的微调方法的领域。现在让我们进入微调阶段。

## 9.3.3 微调

现在我们已经准备好了并验证了数据集，我们可以启动微调过程。当我们需要微调时，有两个步骤要执行。首先，我们上传在前几节中工作的数据集。上传后，每个文件都会获得一个唯一的文件 ID，我们需要保存这个 ID。这个文件 ID 是我们传递给微调作业的参数之一，以便它知道使用哪个文件进行哪个微调作业。

我们可以使用 API 或 GUI 来完成这个任务。我们将通过 Python SDK 和 Azure AI Studio 展示如何实现这一点。我不会在 GUI 手册中展示所有步骤，但这些详细信息可以在配套的 GitHub 仓库[`bit.ly/GenAIBook`](https://bit.ly/GenAIBook)中找到。让我们先使用 SDK。

### 使用 SDK 进行微调

以下列表显示了如何使用 SDK 和 `files.create()` 方法，传入文件名，并指定文件的目的（`fine-tune`）。

#### 列表 9.5 上传数据集进行微调

```py
import os
from openai import AzureOpenAI

API_VERSION = '2023-09-15-preview'                   #1

client = AzureOpenAI(
    api_key=os.getenv('AOAI_FT_KEY'),                #2
    api_version=API_VERSION,
    azure_endpoint = os.getenv('AOAI_FT_ENDPOINT'))  #2

TRAINING_FILENAME = 'data/emoji_FT_train.jsonl'       #3

# Upload the training dataset files
file = client.files.create(
    file=open(TRAINING_FILENAME, "rb"),
    purpose="fine-tune"
)

print("Training file ID:", file.id)
print("Training file name:", file.filename)
```

#1 此版本（或更高版本）是进行微调所必需的。
#2 包含连接详情的环境变量
#3 我们需要用于训练的数据集

当我们运行此代码片段时，我们获得以下输出，其中包含我们在运行第二步时需要了解的文件 ID：

```py
Response:
 Training file ID: file-ca4c57d7ad814211a2db49e0382c5a77
 Training file name: emoji_FT_train.jsonl
```

在上传我们的文件后，我们必须开始微调作业。当使用 SDK 时，这是通过 `fine_tunings.jobs.create()` 方法完成的。此函数需要来自之前步骤的训练数据集文件 ID 和要使用的模型。在我们的情况下，我们想要微调 GPT-3.5 Turbo，特别是 0613 模型。我们还指定了微调所需的 epoch 数。最后，`suffix` 参数是我们可以用来自助跟踪和管理微调模型的一个参数。

#### 列表 9.6 开始微调作业

```py
import os
from openai import AzureOpenAI

API_VERSION = '2023-09-15-preview'                        #1

# Connect to the servvice
client = AzureOpenAI(
    api_key=os.getenv('AOAI_FT_KEY'),                      #2
    api_version=API_VERSION,
    azure_endpoint = os.getenv('AOAI_FT_ENDPOINT'))        #2

# Begin by creating the fine-tuning job
ft = client.fine_tuning.jobs.create(
    training_file="file-ca4c57d7ad814211a2db49e0382c5a77",   #3
    model="gpt-35-turbo-0613",
    hyperparameters={
        "n_epochs":3
    },
    suffix="emoji"
)
print("Finetuning job ID:", ft.id)
```

#1 此版本（或更高版本）是进行微调所必需的。
#2 包含连接详情的环境变量
#3 您看到的文件 ID 将与此不同。

此代码片段提交了一个微调作业，该作业将被排队；根据特定区域和数据中心可用的容量，作业将被执行。提醒一下，使用 Azure OpenAI，您可以在多个区域进行微调。我们的示例显示了我们的 API 调用中的作业 ID：

```py
Finetuning job ID: ftjob-367ee1995af740a0bf24876221585f7a
```

根据数据集大小、我们想要微调的模型以及微调超参数，微调作业可能需要几个小时。微调作业 API 有一个名为 `list()` 的函数调用，我们可以使用它来查看我们已提交的所有微调作业。

#### 列表 9.7 列出所有微调作业

```py
import os
from openai import AzureOpenAI

API_VERSION = '2023-09-15-preview'

client = AzureOpenAI(
    api_key=os.getenv('AOAI_FT_KEY'),
    api_version=API_VERSION,
    azure_endpoint = os.getenv('AOAI_FT_ENDPOINT'))

# List all the FT jobs
ft_jobs = client.fine_tuning.jobs.list()

for ft_job in ft_jobs:
    print(ft_job.id, ft_job.status)
```

列表 9.8 中的输出是一个示例，我们看到我们已经完成了两个微调作业，如 `succeeded` 状态所示；一个作业目前处于 `running` 状态，这意味着有一个正在进行的活跃微调作业。我们刚刚提交的最后微调作业（`ftjob-367ee1995...`）处于 `pending` 状态，这意味着作业将在未来的某个时间排队运行。

#### 列表 9.8 微调作业列表输出

```py
ftjob-367ee1995af740a0bf24876221585f7a pending
ftjob-c41a9dc551834a1aa0be8befe788a22b running
ftjob-1a7faac8856d46e48a038c02555fe6e5 succeeded
ftjob-505d5a8bd321406dbf4605b636b0c0cd succeeded
```

对于一个特定的微调作业，我们还可以看到与该作业相关的各种事件。以下列表显示了此示例，再次使用我们新提交作业的 ID (`ftjob-367ee1995...`)。

#### 列表 9.9 从微调作业中列出事件

```py
import os
from openai import AzureOpenAI

API_VERSION = '2023-09-15-preview'

client = AzureOpenAI(
    api_key=os.getenv('AOAI_FT_KEY'),
    api_version=API_VERSION,
    azure_endpoint = os.getenv('AOAI_FT_ENDPOINT'))

#List all the FT events for the job from
#earier: ftjob-367ee1995af740a0bf24876221585f7a
ft_job_events = client.fine_tuning.jobs.list_events(
    fine_tuning_job_id="ftjob-367ee1995af740a0bf24876221585f7a ", 
    limit=2)

# Loop through the events and print the details
for ft_job_event in ft_job_events:
    print(ft_job_event.id, ft_job_event.message)
```

在这种情况下，输出如下

```py
ftevent-1e89dc7cc62046048bcea50de1cccbb9 Jobs ahead in queue: 1
ftevent-42649f5c7677472f83eaa6cd4cde0dba Job enqueued. 
↪Waiting for jobs ahead to complete.
```

我们还可以每隔几秒轮询检查作业状态，并使用此信息启动另一个工作流程。在这种情况下，此作业运行了大约两小时后完成。为此，我们需要 `IPython` 包，该包可以通过 conda 使用 `conda install ipython` 安装，或者如果使用 pip，则通过 `- pip install ipython` 安装。

#### 列表 9.10 检查微调作业状态

```py
# Define the API version
API_VERSION = '2023-09-15-preview'

# Create an instance of the AzureOpenAI client
client = AzureOpenAI(
    api_key=os.getenv('AOAI_FT_KEY'),
    api_version=API_VERSION,
    azure_endpoint = os.getenv('AOAI_FT_ENDPOINT'))

# Define the job ID of the fine-tuning job to track
JOB_ID = "ftjob-367ee1995af740a0bf24876221585f7a"

# Record the start time of the tracking
start_time = time.time()

# Get the status of the fine-tuning job
ft_job = client.fine_tuning.jobs.retrieve(JOB_ID)
status = ft_job.status

# If the job is not yet done, continue to poll its status every 30 seconds
while status not in ["succeeded", "failed"]:
    ft_job = client.fine_tuning.jobs.retrieve(JOB_ID)
    print(ft_job)

    # Update the status
    status = ft_job.status

    # Print the elapsed time since the start of tracking
    print("Elapsed time: {} minutes {} seconds".format( 
        ↪int((time.time() - start_time) // 60), 
        ↪int((time.time() - start_time) % 60)))

    # Print the current status
    print(f'Status: {status}')

    # Clear the output before displaying new output – prevents flickering
    clear_output(wait=True)

    # Wait for 30 seconds before the next poll
    time.sleep(30)

# Once the job is done, print its final status
print(f'Fine-tuning job {JOB_ID} finished with status: {status}')
```

根据模型和排程微调任务的队列长度，一个微调作业可能需要几个小时才能完成。在这段时间内，我们获得训练指标，帮助我们了解训练过程。

## 9.3.4 微调训练指标

如前所述，训练可能需要几个小时；对于更复杂和更大的模型，可能需要几天。这段时间内的训练不是一个黑盒；我们可以在过程中获取关键指标的具体细节，以获得对正在发生的事情的高级了解。我们有三个关键指标可以跟踪——训练损失、平均标记准确率和标记准确率。

### 损失

损失有两个方面：训练损失和验证损失。训练损失衡量模型预测与实际结果之间的差异。损失值越低，意味着模型越准确，错误越少。较低的损失值表示模型性能更好，表明模型的预测更接近实际数据。

如果我们有验证数据集（强烈推荐），那么我们还有额外的指标，可以让我们衡量模型的表现。验证损失是一个衡量模型在验证集上误差的指标，验证集是从数据集中预留出来以评估模型在新或未见数据上的性能的部分。验证损失是通过使用与训练损失相同的成本函数，对验证集中的每个示例的错误进行求和来计算的。验证损失通常在每个 epoch 后测量，即对训练集的完整遍历。

图 9.3 展示了使用 Azure OpenAI 微调时的损失示例以及训练过程中的模型性能。图 9.3 中显示的微调训练结果的训练损失图说明了模型从训练数据中学习的效果。
