
```py
driver.get("https://automationintesting.online");

ContactFormPage contactFormPage = new ContactFormPage(driver);
contactFormPage.enterName("John Smith");
contactFormPage.enterEmail("test@email.com");
contactFormPage.enterPhone("07123456789");
contactFormPage.enterSubject("Testing");
contactFormPage.enterDescription("This is a test message");
contactFormPage.clickSubmitButton();
```

并开始输入`MessagePayload`以触发 Copilot 添加以下代码作为替换：

```py
MessagePayload messagePayload = new MessagePayload();
messagePayload.setName("Test User");
messagePayload.setEmail("test@email.com");
messagePayload.setPhone("0123456789");
messagePayload.setSubject("Test Subject");
messagePayload.setDescription("Test Description");

MessageRequest messageRequest = new MessageRequest();
messageRequest.sendRequest(messagePayload);
```

类似于我们使用 Copilot 在 UI 中生成创建消息的代码时，第一次运行自动检查会导致失败。检查将运行并收到 400 状态码，因为预测的测试数据再次不符合验证规则。因此，为了确保我们的消息符合必要的验证规则，我们需要更新以下方法以正确的测试数据：

```py
messagePayload.setPhone("074123456789");
messagePayload.setDescription("Test Description that is larger");
```

一旦这些测试数据参数被更新，我们现在应该看到自动检查再次通过。

这个例子表明我们可以使用 LLM 工具帮助我们更新自动检查的特定方面，使其更加健壮，但这要求我们具备良好的分析眼光，了解哪些检查区域需要改进。这是因为我们既了解我们的自动检查，也了解我们正在测试的系统，这是我们迄今为止使用的工具所缺乏的。例如，如果我们给 ChatGPT 以下提示：

| 建议如何改进这个自动测试，使其不那么不可靠 |
| --- |

然后将我们的自动检查代码添加到提示中，这些将是返回的建议（总结）：

| • 添加显式等待<br>• 使用稳定的定位器<br>• 处理异步操作<br>• 隔离测试<br>• 重试失败的操作<br>• 检查错误条件<br>• 审查和更新测试环境 |
| --- |

这些都是合法的考虑因素，但它们是通用问题，并不一定给我们提供足够的信息来解决具体问题。因此，我们构建改进过程，寻求工具帮助我们快速生成必要的代码。

## 活动七.3

使用 ChatGPT 和 Copilot，尝试将登录过程转换为 API 调用。为此练习，你需要编写代码：

- 将凭证发送到`POST /auth/login`
- 从 HTTP 响应中提取`token`值
- 在前往消息页面之前，将`token`值存储为浏览器中的 cookie

### 7.2.2 使用 AI 工具进入状态

本章展示了无论我们是在构建 UI 自动化、API 自动化还是其他完全不同的东西，使用 AI 的成功模式始终相同。我们对自动化检查的设计和结构的深入了解让我们知道何时何地使用 AI 工具来帮助我们完成特定任务以创建和维护有价值的自动化。围绕 AI 自动化的营销可能会让我们相信，当涉及 AI 时，我们在创建自动化检查中的作用是有限的。但如果我们想要帮助我们创建高质量产品的自动化，那么我们最好的行动方案是与 AI 工具建立关系，将我们的技能集置于工作的核心。

## 摘要

- 仅使用 ChatGPT 等工具尝试生成整个自动化 UI 测试可能需要大量的重做。相反，我们希望在 UI 自动化过程的特定点上选择性地使用 AI 工具。
- 使用 Copilot 等工具开始一个新项目可能会产生不同的结果。
- 我们在项目中添加的细节越多，Copilot 就会越准确。
- 通过提供 HTML 和转换指令，我们可以使用正确的提示类型在 ChatGPT 中快速生成页面对象。
- 我们可以通过结合 ChatGPT 和 Copilot（或类似工具）快速生成自动化检查。
- AI 工具的输出并非 100% 准确，因为它缺乏上下文——例如，使用测试数据或使用来自库的最新方法。
- 在创建自动化检查时使用 AI 工具的成功来自于在创建过程中使用 AI 工具完成特定任务。
- 我们领导创建过程，确定何时 AI 工具可以帮助我们加快速度。
- 如果我们能够识别自动化检查特定元素的可改进之处，我们可以使用 AI 工具更快地完成这些改进。
- 如果我们要求 LLMs 评估我们的检查并提出改进意见，我们会得到一些通用的答案作为回报。
- 我们可以使用在自动化检查中特定任务上使用 AI 工具的相同过程来维护它们。
