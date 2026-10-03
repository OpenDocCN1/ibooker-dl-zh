
```py
root = py_trees.composites.Sequence("RootSequence", memory=True)

thread = api.create_thread()     #1
challenge = textwrap.dedent("""
 #2
""")
judge_test_cases = textwrap.dedent("""
 #3
""")

hacker = create_assistant_action_on_thread(   
    thread=thread,      #4
    action_name="Hacker",
    assistant_name="Python Coding Assistant",
    assistant_instructions=textwrap.dedent(f"""
    Challenge goal: 
    {challenge}      #5
    Solve the challenge and output the 
final solution to a file called solution.py        
    """),
)
root.add_child(hacker)

judge = create_assistant_action_on_thread(    
    thread=thread,      #6
    action_name="Judge solution",
    assistant_name="Coding Challenge Judge",
    assistant_instructions=textwrap.dedent(
        f"""
    Challenge goal: 
    {challenge}      #7
    Load the solution from the file solution.py.
    Then confirm is a solution to the challenge 
and test it with the following test cases:
    {judge_test_cases}      #8
    Run the code for the solution and confirm it passes all the test cases.
    If the solution passes all tests save the solution to a file called 
judged_solution.py
    """,
    ),
)
root.add_child(judge)

# verifier operates on a different thread, essentially in closed room
verifier = create_assistant_condition(     #9
    condition_name="Verify solution",
    assistant_name="Python Coding Assistant",
    assistant_instructions=textwrap.dedent(
        f"""
    Challenge goal: 
    {challenge}      #10
    Load the file called judged_solution.py and 
verify that the solution is correct by running the code and confirm it passes 
all the test cases:
    {judge_test_cases}      #11
    If the solution is correct, return only the single word SUCCESS, otherwise 
return the single word FAILURE.
    """,
    ),
)
root.add_child(verifier)

tree = py_trees.trees.BehaviourTree(root)

while True:
    tree.tick()
    time.sleep(20)      #12
    if root.status == py_trees.common.Status.SUCCESS:    #13
        break
### Required assistants – 
### Python Coding Assistant and Coding Challenge Judge 
### install these assistants through the Playground
```

#1 创建一个将由黑客和裁判共享的消息线程

#2 如示例列表 6.5 所示，挑战的体现

#3 如示例列表 6.6 所示，测试的体现

#4 创建一个将由黑客和裁判共享的消息线程

#5 如示例列表 6.5 所示，挑战的体现

#6 创建一个将由黑客和裁判共享的消息线程

#7 如示例列表 6.5 所示，挑战的体现

#8 如示例列表 6.6 所示，测试的体现

#9 调用创建一个新的消息线程

#10 如示例列表 6.5 所示，挑战的体现

#11 如示例列表 6.6 所示，测试的体现

#12 睡眠时间可以根据需要调整上下，并且可以用来限制发送给 LLM 的消息。

#13 过程将继续，直到验证成功。

通过在 VS Code 中加载文件或使用命令行来运行 ABT。在终端中跟踪输出，并观察助手如何通过树中的每个步骤工作。

如果在条件节点上解决方案未能通过验证，则过程将按照树继续进行。即使在这个简单解决方案中，您也可以快速创建许多变体。您可以通过添加更多节点/步骤和子树来扩展树。也许您希望一个黑客团队分解和分析挑战，例如。

本例的工作主要使用 Playground 代码完成，使用了辅助函数 `create_assistant_condition` 和 `create_assistant_action_on_thread`。此代码使用几个类来集成 `py_trees` 行为树代码和 OpenAI 助手代码，这些代码被封装在 Playground 中。如果您想了解底层细节，请审查项目中的代码。

### 6.3.3 对话式 AI 系统与其他方法

当我们在第四章中查看 AutoGen 时，我们已经研究了对话式多智能体系统。ABT 可以通过对话（通过线程）和其他方法（如文件共享）的组合来工作。让您的助手/智能体传递文件有助于减少嘈杂和重复的思想/对话的数量。相比之下，对话式系统从潜在的涌现行为中受益。因此，使用两者可以帮助进化更好的控制和解决方案。

列表 6.7 中的简单解决方案可以扩展以处理更多现实世界的编码挑战，甚至可能作为编码 ABT 工作。在下一节中，我们将构建一个不同的 ABT 来处理不同的问题。

### 6.3.4 将 YouTube 视频发布到 X

在本节的练习中，我们查看一个可以执行以下操作的 ABT：

1.  搜索 YouTube 上特定主题的视频并返回最新视频。

1.  下载您搜索提供的所有视频的转录内容。

1.  总结转录内容。

1.  审查总结的转录内容，并选择一个视频来撰写 X（以前称为 Twitter）帖子。

1.  写一篇引人入胜的帖子关于视频，确保其字符数少于 280 个。

1.  审查帖子，然后在 X 上发布。

图 6.9 显示了使用每个不同助手组装的 ABT。在这个练习中，我们使用序列节点作为根节点，每个助手执行不同的操作。此外，为了保持简单，每个助手的交互将始终在新的线程中发生。这将每个助手的交互隔离成简短的对话，如果出现问题，更容易调试。

![图](img/6-9.png)

##### 图 6.9 YouTube 社交媒体 ABT

### 6.3.5 需要的 X 设置

如果您计划运行本练习中的代码，您必须将您的 X 凭证添加到 `.env` 文件中。`.env.default` 文件显示了凭证需要的样子，如列表 6.8 所示。您不必输入您的凭证。这意味着最后一步，发布，将会失败，但您仍然可以查看文件（`youtube_twitter_post.txt`）以查看生成了什么。

##### 列表 6.8 配置凭证

```py
X_EMAIL = "twitter email here"
X_USERNAME = "twitter username here"
X_PASSWORD = "twitter password here"
```

##### YouTube 搜索和垃圾邮件

如果您打算真正运行这个练习并让它发布到您的 X 账户，请注意 YouTube 存在一些垃圾邮件问题。助手已被配置为尝试避免视频垃圾邮件，但其中一些可能仍然会通过。构建一个可以在避免垃圾邮件的同时处理视频的 ABT 具有一些合适的应用。

列表 6.9 仅显示了创建助手操作的代码。这个 ABT 使用了三个不同的助手，每个助手都有自己的任务指令。请注意，每个助手都有定义其角色的独特指令集。您可以通过 Playground 查看每个助手的指令。

##### 列表 6.9 `agentic_btree_video_poster_v1.py`

```py
root = py_trees.composites.Sequence("RootSequence", memory=True)

search_term = "GPT Agents"
search_youtube_action = create_assistant_action(
    action_name=f"Search YouTube({search_term})",
    assistant_name="YouTube Researcher v2",
    assistant_instructions=f"""
    Search Term: {search_term}
    Use the query "{search_term}" to search for videos on YouTube.
    then for each video download the transcript and summarize it 
for relevance to {search_term}
    be sure to include a link to each of the videos,
    and then save all summarizations to a file called youtube_transcripts.txt
    If you encounter any errors, please return just the word FAILURE.
    """,
)
root.add_child(search_youtube_action)

write_post_action = create_assistant_action(
    action_name="Write Post",
    assistant_name="Twitter Post Writer",
    assistant_instructions="""
    Load the file called youtube_transcripts.txt,
    analyze the contents for references to search term at the top and 
then select
    the most exciting and relevant video related to: 
    educational, entertaining, or informative, to post on Twitter.
    Then write a Twitter post that is relevant to the video,
    and include a link to the video, along
    with exciting highlights or mentions, 
    and save it to a file called youtube_twitter_post.txt.
    If you encounter any errors, please return just the word FAILURE.
    """,
)
root.add_child(write_post_action)

post_action = create_assistant_action(
    action_name="Post",
    assistant_name="Social Media Assistant",
    assistant_instructions="""
    Load the file called youtube_twitter_post.txt and post the content 
to Twitter.
    If the content is empty please do not post anything.
    If you encounter any errors, please return just the word FAILURE.
    """,
)
root.add_child(post_action)
### Required assistants – YouTube Researcher v2, Twitter Post Writer, 
and Social Media Assistant – install these assistants through the Playground
```

按照常规方式运行代码，几分钟后，在`assistants_output`文件夹中会出现一个新的帖子。图 6.10 显示了使用此 ABT 生成的帖子示例。如果每天运行此 ABT 生成超过几个帖子，可能会被 X 账户封禁。如果您已配置 X 凭证，您将在您的动态中看到帖子。

![figure](img/6-10.png)

##### 图 6.10 ABT 的一个示例 X 帖子

这个 ABT 仅用于演示目的，并不适用于生产或长期使用。这个演示的主要功能是展示搜索和加载数据，摘要和过滤，然后生成新内容，最后突出多个自定义操作和与 API 的集成。

## 6.4 构建对话式自主多智能体

多智能体系统的对话方面可以驱动反馈、推理和涌现行为等机制。使用隔离助手/智能体的 ABTs 来驱动智能体可以有效地控制结构化过程，正如我们在 YouTube 发布示例中看到的那样。然而，我们也不愿错过智能体/助手之间对话的好处。

幸运的是，Playground 提供了将助手隔离或加入对话线程的方法。图 6.11 显示了助手如何以各种组合被隔离或混合到线程中。将隔离与对话相结合提供了两种模式的最佳效果。

![figure](img/6-11.png)

##### 图 6.11 隔离和对话助手的各种布局

我们将检验一个简单但实用的练习，以展示对话模式的效率。对于下一个练习，我们将使用两个助手在一个 ABT 中进行对话。下面的列表显示了代码中树的结构以及相应的助手。

##### 列表 6.10 `agentic_conversation_btree.py`
