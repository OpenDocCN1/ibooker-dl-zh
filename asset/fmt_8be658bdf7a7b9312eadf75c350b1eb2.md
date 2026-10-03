
对于其他四个相关的上下文因素也进行同样的操作，描述它们是如何收集的，如何增强虚拟助手的交互，以及它们对用户参与度的潜在影响。

1. 设计上下文感知的响应。创建三个虚拟助手可能收到的不同用户查询。对于每个查询，指定至少两件可能用于个性化响应的上下文信息。写出虚拟助手会提供的上下文感知响应，解释上下文如何改善交互。以下是一个示例：

   - 用户查询：“今天的天气怎么样？”
   - 上下文信息：用户的位置，当前时间
   - 上下文感知响应：“早上好！今天早上旧金山的天气是晴朗的，下午晚些时候最高气温将达到 75°F。”

## 9.2 理解模式

*模式*指的是用户与虚拟助手沟通的各种渠道或方法。模式包括文本、语音、图像和多模态交互。每种模式都带来其优势和挑战，影响着用户如何与助手互动以及助手如何有效地满足用户需求。有效利用模式可以使聊天机器人从功能型转变为直观和吸引人的。虚拟助手的性能受模式的影响。

例如，基于文本的交互可能更适合于打字更方便的环境，而语音交互则提供了免提便利和更自然的对话流程。多模态交互结合了文本、语音和视觉元素，可以提供更丰富和更灵活的用户体验。

评估这些模式如何影响用户参与度和交互设计对于持续改进至关重要。开发者可以应用特定的评估技术来评估不同模式下的现有虚拟助手流程，以确定改进的区域。这确保了一个有效的持续改进过程。

### 9.2.1 模式比较

聊天机器人最初是简单的基于文本的对话应用，但已演变为多模态通信。一般来说，*模式*指的是某事发生或体验的方式或模式。聊天机器人的模式与它如何与用户交互以及处理信息相关：

- *文本*—文本是主要且最常用的模式。用户输入他们的问题或命令，聊天机器人以文本形式回应。
- *视觉*—视觉模式是主要与文本交互的聊天机器人。尽管如此，其回应使用视觉元素，如按钮、图片、轮播图、视频和其他图形界面，以促进交互。添加视觉元素可以通过提供视觉提示和选项来增强用户体验。
- *语音*—语音机器人以口语与用户交互。这包括理解口语输入和生成口语回应。通常，底层的对话引擎仍然处理文本，因此发生语音到文本和文本到语音的转换。
- *多模式*—这些机器人结合了多种模式，如文本、语音、图像和视频。例如，一个多模式聊天机器人可以理解口语查询，显示相关图片，并以文本和语音的形式回应。

表 9.3 简要比较了不同的模式，突出了它们的独特特征、优点和挑战，以及何时使用它们及其对用户参与度的影响。此外，它还概述了每种模式的相应评估技术。

##### 表 9.3 模式概述

| | 文本模式 | 视觉模式 | 语音模式 | 多模式交互 |
|---|---|---|---|---|
| 描述 | 通过书面文字进行沟通 | 使用带有额外视觉元素的文本进行沟通 | 通过口语进行交互 | 结合文本、语音和视觉元素 |
| 优点 | 精确的沟通。易于参考。 | 通过引人入胜的界面通过视觉增强理解 | 免提操作。自然的对话流程。 | 更丰富的用户体验。多变的交互。 |
| 挑战 | 打字可能很慢。可能缺乏情感色调。 | 视觉元素可能会分散注意力或被忽略。需要精心设计的用户界面。 | 需要清晰的语音。可能误解口音。 | 设计和实现更复杂。 |
| 何时使用 | 在打字方便的情况下 | 需要视觉辅助工具以更好地理解的场景 | 需要免提交互的场景 | 需要多种模式结合的场景 |
| 对用户参与度的影响 | 文本为中心的上下文中高度参与 | 通过视觉辅助工具提高参与度 | 由于自然交互而增加的参与度 | 通过多种交互模式增强的参与度 |
| 评估技术 | 用户反馈调查。可用性测试。 | 用户界面测试。视觉有效性分析。 | 语音识别准确性测试。用户反馈。 | A/B 测试。多模式可用性评估。 |

选择模态直接影响到用户如何与对话式人工智能互动和参与。通过识别文本、语音、视觉和跨模态交互的优势和挑战，聊天机器人设计者可以更好地定制体验以满足用户需求和偏好。虚拟助手的性能在于能够无缝集成和优化这些不同的通信渠道。

### 9.2.2 设计虚拟助手流程中模态的重要性

设计虚拟助手流程时，模态的选择对用户体验和交互的有效性有显著影响。

需要考虑的一个关键方面是虚拟助手将被使用的上下文。例如，当用户需要免提操作时，如驾驶或烹饪，语音模态至关重要。这需要强大的自然语言处理能力来准确解释语音命令并提供适当的响应。此外，设计者必须确保语音交互感觉自然流畅，避免机器人或过度重复的响应，这可能会让用户感到沮丧。嘈杂的环境也起着作用，因为背景噪音可能会影响语音识别系统的性能，需要先进的降噪技术。

对于语音交互来说，简洁性至关重要。由于语音是一种比文本更受限的媒介，用户从简短、直接且易于处理和理解的响应中受益。在语音交互中，冗长的回答会显得繁琐，因为用户无法像在文本中那样视觉扫描或跳过部分内容。因此，虚拟助手应优先提供清晰、简洁的信息，以支持高效的免提体验。

在基于文本的交互中，清晰度至关重要，但模态允许更广泛的沟通。虽然简洁性很有价值，但文本交互为用户提供了自己节奏下回顾细节的选项。基于文本的虚拟助手仍应设计为处理简洁的用户输入并提供可操作的回复，但它们可以使用自动建议和预测文本等特性来使交互更快更高效。使用文本时，用户可能需要多任务处理或需要更详细的响应，他们可以按需快速浏览或重新查看，因此助手的提示和响应应精心设计，以免让用户感到不知所措。

小贴士：考虑可访问性选项，例如为视障用户提供的屏幕阅读器，确保基于文本的交互具有包容性和用户友好性。

视觉模式，它结合了文本和视觉元素，如图像、按钮和图标，需要不同的设计方法。通过提供上下文线索并使界面更具互动性，视觉元素可以显著增强理解和参与度。您必须在视觉吸引力和功能性之间保持平衡。界面中过多的视觉元素可能会降低清晰度和可用性。设计师应优先考虑视觉清晰度和一致性，使用视觉元素来补充和增强文本，而不是掩盖它。这种模式在视觉辅助可以简化复杂信息的情况下尤其有效，例如故障排除指南或交互式教程。确保视觉元素能够适应不同的屏幕尺寸和分辨率，对于在多种设备上保持无缝的用户体验至关重要。

### 9.2.3 模式如何影响用户体验的例子

列表 9.1 和 9.2 探讨了技术支持聊天机器人如何使用不同的模式——基于文本的说明与视觉（视频）说明——来帮助用户设置新的打印机。

##### 列表 9.1 文本模式交互

```py
User: I need help setting up my new printer.
Chatbot: I would be happy to help you set up your printer. Please follow these steps: Unbox the printer, place it on a stable surface and connect it to power. Install the ink cartridges and load the paper. Download and install the printer drivers from our website for software setup, then connect the printer to your computer via USB or Wi-Fi.
User: I can't open the printer cover to install the ink cartridges. Where is the latch?
Chatbot: The latch should be on the front panel. Please check around the edges for any visible buttons or levers.
User: I've checked, and there's no latch. Can you help me find it?
Chatbot: I apologize for the inconvenience. If you're unable to locate the latch, please refer to the user manual for your specific printer model for detailed instructions.
User: I don't have the manual. This is really frustrating.
Chatbot: I understand your frustration. Unfortunately, I can't provide more specific guidance without knowing the exact model. Could you check our website for the support page for your model?
User: This is useless. Forget it!
```

##### 列表 9.2 视觉模式交互

```py
User: I need help setting up my new printer.
Visual bot: I would be happy to help you set up your printer. Would you like to watch a video tutorial on the setup process?
User: Yes, please.
Visual bot: Great! Here is a video that will guide you through the entire setup process step by step. [*Video starts to play*] *[User watches the video, then starts setting up the printer]*
User: I am at the part where I need to insert the ink cartridges. Can you show that part again?
Visual bot: Sure! Here is the specific section of the video that demonstrates how to insert the ink cartridges. *[Video goes to specific timestamp]*
User: Perfect, that makes it so much clearer. Thanks!
```

基于文本的说明，虽然提供了详细和精确的步骤指导，但当用户遇到说明中未明确涵盖的问题时，往往会引起困惑和挫败感。没有视觉提示，用户可能难以理解特定的步骤，例如在打印机上找到闩锁，这可能导致令人沮丧的来回交互。这种类型的交互在用户对不清晰的闩锁位置的日益增长的挫败感中表现得非常明显。需要额外的澄清可能会导致用户体验效率降低和更加恼人。

相比之下，视觉模式，尤其是通过视频说明，提供了一种更清晰、更吸引人的方式来遵循设置过程。视频使用户能够看到实际的过程，使其更容易理解和复制每一步。这减少了多次交互的需要，并最小化了误解或错误解释指令的机会。如果需要进一步的澄清，用户可以参考视频的特定部分，确保设置体验更加顺畅和高效。

考虑一个用户在语音聊天机器人的帮助下询问航班信息的场景。一个设计良好的语音机器人可以提供适当的信息，甚至在适当的时候切换到不同的模式，而一个设计不佳的机器人可能会导致更多的困惑和挫败感，如列表 9.3 和 9.4 所展示的。

##### 列表 9.3 设计不佳的语音机器人交互

```py
User: I need to check the status of my flight.
Voice bot: Please provide your flight number and departure date to check the flight status.
User: It's flight XY123 on August 5th.
Voice bot: Please hold while I retrieve this information... Your flight UA123 on August 5th is on time. Please visit our website if you need more details or have other inquiries.
User: Can you tell me the gate number?
Voice bot: Please visit our website for more details about your flight.
User: [..hangs up..]
```

##### 列表 9.4 设计良好的语音机器人交互

```py
User: I need to check the status of my flight.
Voice bot: I can help with that. What is your flight number and the departure date?
User: Flight UA123 on August 5th.
Voice bot: One moment while I check that for you... Flight UA123 on August 5th is scheduled to depart on time from gate 22B. Would you like information on baggage claim or perhaps directions to the gate?
User: Yes, directions please.
Voice bot: Gate 22B is located in Terminal C. After security, follow the signs towards the terminal. It's near the food court. Would you like a map sent to your phone?
User: Yes, please.
Voice bot: I've sent it to your phone. Anything else I can help with today?
```

在设计不佳的交互中，语音机器人未能提供全面的支持，迫使用户通过不那么方便的渠道寻求额外信息。相比之下，设计良好的语音机器人能够预见用户的需求，通过提供所需信息以及相关细节的补充，从而提供全面的帮助，改善整体用户体验。

### 9.2.4 语音机器人设计考虑因素

正如前一个章节的示例所示，为了确保积极的语音机器人用户体验，必须考虑几个关键因素。首先，沟通的清晰性和简洁性是必不可少的。机器人应该将信息分解成可管理的、一步一步的指令，使用户能够轻松跟随并理解每个过程的部分。情境意识是另一个关键方面；机器人应该识别用户的进度并提供针对他们情况的相应指导。此外，结合错误处理并提供澄清或重复的选项可以帮助可能需要额外帮助的用户。设计机器人时，保持自然、对话的语调也很重要，这样可以使交互感觉更直观，而不是那么机械。最后，提供多模态支持，例如提供视频教程或视觉辅助链接，可以满足不同的学习偏好，并增强语音机器人的整体效果。通过优先考虑这些考虑因素，设计师可以创建出既高效又用户友好，并能适应广泛用户需求和情境的语音交互。

下面是语音机器人设计的五大设计考虑因素：

- *提供逐步步骤*。语音机器人应该将任务分解成清晰、可管理的步骤，并按顺序引导用户完成每个步骤。这一点至关重要，因为用户不能像阅读文本那样视觉扫描步骤。通过允许用户在继续之前确认完成，机器人确保他们不会在过程中感到不知所措或迷失方向。
- *设计鲁棒的错误处理策略*。由于口音、发音障碍或背景噪音导致的语音识别错误，在语音交互中误解更为常见。设计策略以优雅地处理这些错误，而不让用户感到沮丧，这是至关重要的。这包括澄清问题和简化响应，以便重新回到正确的轨道。
- *考虑自适应响应时间*。语音机器人必须有效地管理交互的节奏。由于用户不能像阅读文本那样回顾听到的词语，机器人需要调整其说话速度，允许用户自然地处理暂停，并对用户可能需要更多时间的提示保持敏感。
- *在采取行动前提供确认*。语音机器人应在采取重大行动前与用户确认。这在语音交互中尤为重要，因为它可以防止对语音命令的误解导致意外行动。
- *支持不同的语音模式*。设计语音识别系统以理解不同的口音、方言和语音模式。这种包容性确保了更广泛的用户能够有效地与机器人交互。

值得注意的是，这些语音机器人设计考虑因素对于聊天机器人设计同样相关。两种模态都依赖于基本元素，如用户需求分析、上下文感知、自然语言处理和强大的错误处理。此外，个性化、可访问性、无缝切换和持续改进等因素在创建引人入胜、用户友好的虚拟助手方面发挥作用，无论它们使用语音还是文本。

然而，某些设计因素必须适应语音交互。与基于文本的聊天机器人不同，语音机器人需要处理语音模式、口音和背景噪音的变化，所有这些都可能影响理解和使用体验。语音交互最关键的因素是简洁和清晰，因为用户可能需要帮助来保留长的口头指令。相比之下，基于文本的聊天机器人可以提供用户可以按自己的节奏阅读的详细信息，这使得参考对话的先前部分更容易。通过针对每种模态的独特特征调整这些考虑因素，设计师可以创建更直观、更有效的虚拟助手。

##### 练习

1. 审查以下场景，并确定每种场景（文本、语音、视觉或多模态）最有效的模态。为你的选择提供简要的解释：

   - 场景 1：用户需要帮助设置复杂的软件程序。
   - 场景 2：用户在驾驶时询问最近的加油站。
   - 场景 3：用户想要浏览新服装商品的目录。
   - 场景 4：用户请求每日励志名言。

2. 选择一个虚拟助手可能帮助用户完成的任务，例如预订航班或选择信用卡。设计一个至少包含两种不同模态的多模态交互。解释每个模态如何增强用户体验并有助于完成任务。以下是一个示例：

   - 任务：预订航班
   - 交互：助手使用语音提出初始问题（例如，目的地、日期），并使用文本显示带有图片和价格的航班选项。
   - 说明：语音交互提供了一种快速、自然的信息收集方式，而文本和视觉帮助用户比较选项并做出明智的决定。

## 9.3 使用 RAG 增强上下文感知和提升整体用户体验
