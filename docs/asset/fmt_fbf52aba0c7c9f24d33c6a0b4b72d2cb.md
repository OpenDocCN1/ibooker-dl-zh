
# Markdown 排版指南

另一个有用的特性是缩进通常不重要，但对于技术内容（例如，源代码），您可以使用三重反引号打开和关闭的块（```py）。

- 如果您想直接向用户显示模型输出，Markdown 非常易于渲染。
- Markdown 的超链接功能允许模型包含易于解析的链接，这可以帮助您验证来源并以编程方式检索内容。

此外，通常在 Markdown 文件的开头提供目录，这非常有用。目录可以作为长提示词引言的有用部分，因为它帮助模型定位自身，就像帮助人类一样。它还可以成为控制完成过程的绝佳工具，有两种方式：

1. 对于思维链提示或管理过于冗长的模型，您可以使用草稿本方法。在目录中的`# Conclusion`之前添加诸如`# Ideas`或`# Analysis`之类的部分，有助于引导模型得出更明智的结论，同时允许您忽略前面的部分。
2. 您可以通过在结论之后添加诸如`# Appendix`或`# Further Reading`之类的部分，轻松指示模型的响应应在何时结束。将`# Further Reading`设置为停止序列，可确保模型完成任务，从而节省计算资源。

目录的这两种用例都在图 6-3 中进行了演示。请注意，由于这是一个示例，上下文量少于模型为正确回答此类问题所需的数量。此外，LLM 并非神谕：在适当的上下文中，模型可以成为构思的良好工具，但没有理由认为其意见应比会计部门的 Jerry 的意见更重要。

![A screenshot of a computer  Description automatically generated](img/pefl_0603.png)

###### Figure 6-3\. 使用目录的 Markdown 报告（使用 OpenAI 的 text-davinci-003 获得的完成）

## 结构化文档

结构化文档遵循正式规范，允许您对完成形式做出强有力的假设。这使得解析更容易，包括复杂输出的解析。

一个很好的例子是 Anthropic 的 Artifacts 提示词。Artifacts 将在本书最后一章再次出现，但就目前而言，您应该知道 Artifacts 是用户和助手协作完成的自包含文档。Artifacts 的示例包括 Python 脚本、小型 React 应用、mermaid 图表和可缩放矢量图形（SVG）图表。Artifacts 在 UI 中显示为对话右侧窗格中的文本，对于 React、Mermaid 和 SVG，它们会被渲染为功能性或视觉原型。

表 6-3 显示了 Artifacts 提示词的删节版本（[此提示词由 @elder_plinius 提取](https://oreil.ly/Lwsp1)）。为了使 Artifacts 正常工作，提示词使用 XML 文档结构，清晰划分了交互的各个部分。`artifacts_info` 提示词持有相当于系统消息的内容，解释 Artifacts 的工作原理。它包含一个`examples`部分，其中有几个`example`块。每个示例都有一个`user_query`和一个`assistant_response`。

在`assistant_response`内部，事情变得最有趣。首先，助手开始其响应，然后注入一个`antThinking`块，以便助手可以“思考”用户的请求是应该使用 Artifact 还是仅作为正常对话处理。如果决定使用 Artifact，则文本将包含一个`antArtifact`块，其中包含 Artifact 的文本。请注意，`antArtifact`标签还包含诸如 Artifact 标题和所用语言之类的属性。

表 6-3\. 用于生成 Artifacts 的 Anthropic 结构化提示词（删节版）

| 提示词 | `The assistant can create and reference Artifacts during conversations. Artifacts are for substantial, self-contained content that users might modify or reuse, and they are displayed in a separate UI window for clarity.``Here are some examples of correct usage of Artifacts by other AI assistants:` |
|---|---|
| | ```<examples>````<example>````<user_query>你能帮我创建一个计算阶乘的 Python 脚本吗？``factorial of a number?</user_query>````<assistant_response>``当然！这里有一个计算阶乘的 Python 脚本``数字：````<antThinking>创建一个计算阶乘的 Python 脚本符合``良好文件的标准。因此，我正在创建一个新的``文件。````</antThinking>````<antArtifact identifier="factorial-script"``type="application/vnd.ant.code" language="python"``标题="简单的 Python 阶乘脚本">``def factorial(n):``    if n == 0:``        return 1``    else:``        return n * factorial(n - 1)``...``</assistant_response>``</example>``[...省略几个示例...]``</examples>`````py````The assistant should always take care to not produce Artifacts that would be highly hazardous to human health or wellbeing if misused, even if is asked to produce them for seemingly benign reasons.``</artifacts_info>``Claude is now being connected with a human.``<user_query>**Can you help me create a Python script to factor a number into its prime factors?**</user_query>``<assistant_response>` |
| 完成 | `Sure! Here's a Python script that factors a number into its prime factors:``<antThinking>This code is self-contained and is likely to be reused or modified. Therefore, I'm creating a new Artifact.</antThinking>``<antArtifact identifier="prime-factor-script"``type="application/vnd.ant.code" language="python"``title="Python script for factoring numbers into prime factors"> def prime_factors(n):` |
| | ```factors = []``d = 2``while n > 1:``    while n % d == 0:``        factors.append(d)``        n /= d``    d += 1``return factors``print(prime_factors(24)) # 输出: [2, 2, 2, 3]``</assistant_response> |
