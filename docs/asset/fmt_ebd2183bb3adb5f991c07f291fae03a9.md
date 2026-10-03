
```py
async def main() -> None:
    chatting = True
    context = kernel.create_new_context()

    print("Welcome to your first AI Agent\     #1
\n  Type 'exit' to exit.\
\n  Ask to get a list of currently playing movies by genre."
    )
    while chatting:     #2
        chatting, context = await chat(context)     #3

if __name__ == "__main__":
    asyncio.run(main())
```

#1 对用户的介绍

#2 继续直到聊天为 False

#3 异步调用聊天函数

运行聊天界面，运行文件（F5），然后询问特定类型的电影或电视节目。示例对话会话如列表 5.24 所示。此输出显示了请求列出两个类型的电影时，聊天界面如何多次调用 `get_top_movie_by_genre` 函数。

##### 列表 5.24 `SK_service_chat.py`（示例对话）

```py
Welcome to your first AI Agent
  Type 'exit' to exit.
  Ask to get a list of currently playing movies by genre.
User:> Input: can you give me list of the current top playing movies for 
the action and comedy genres?

Function name: get_top_movies_by_genre     #1
Arguments:
  genre = action
Function name: get_movie_genre_id     #2
Arguments:
  genre_name = action
Function name: get_top_movies_by_genre    #1
Arguments:
  genre = comedy
Function name: get_movie_genre_id    #2
Arguments:
  genre_name = comedy
Agent:> Here are the current top-playing movies 
for the action and comedy genres:

**Action:**     #3
1\. The Hunger Games: The Ballad of Songbirds & Snakes
2\. Rebel Moon - Part One: A Child of Fire
3\. Aquaman and the Lost Kingdom
4\. Silent Night
5\. The Family Plan
6\. Freelance
7\. Migration
8\. Sound of Freedom
9\. Godzilla Minus One

**Comedy:**     #4
1\. The Family Plan
2\. Wonka
3\. Freelance
4\. Saltburn
5\. Chicken Run: Dawn of the Nugget
6\. Trolls Band Together
7\. There's Something in the Barn
8\. Migration

Please note that some movies may overlap in both genres, such as 
"The Family Plan" and "Freelance ."
```

#1 LLM 调用两次 `get_top_movies_by_genre`。

#2 内部调用以获取类型 ID

#3 当前最受欢迎的动作电影列表

#4 当前最受欢迎的喜剧电影列表

一定要探索聊天界面的边界以及你可以从 TMDB 服务中请求的内容。例如，尝试请求电影或电视节目的类型列表。这个服务是一个很好的尝试，但我们可能做得更好，正如我们将在下一节中看到的那样。

## 5.6 在编写语义服务时进行语义思考

现在我们已经看到了将 API 转换为语义服务接口的出色演示。就目前而言，这些函数返回当前正在上映的电影和电视节目的标题。然而，仅仅返回标题，我们限制了 LLM 自行解析结果的能力。

因此，我们将创建 `TMDbService` 的 v2 版本来纠正这个问题，并将结果作为 JSON 字符串返回。在 VS Code 中打开文件 `tmdb_v2.py`，并滚动到 `get_top_movies_by_genre` 函数。

##### 列表 5.25 `tmdb_v2.py`（`get_top_movies_by_genre`）

```py
def get_top_movies_by_genre(self, genre: str) -> str:
        print_function_call()
        genre_id = self.get_movie_genre_id(genre)
        if genre_id:
            #same code …

            return json.dumps(filtered_movies)     #1
        else:
            return ""
```

#1 现在返回一个过滤后的 JSON 字符串列表

现在在 VS Code 中打开 `SK_service_chat.py` 文件，注释和取消注释列表 5.26 中显示的行。这将使用 `TMDbService` 的第 2 个版本，该版本以单个字符串的形式输出完整的 JSON 文档结果。

##### 列表 5.26 `SK_service_chat.py`（修改导入）

```py
#from skills.Movies.tmdb import TMDbService     #1
from skills.Movies.tmdb_v2 import TMDbService     #2
```

#1 注释掉这一行。

#2 取消注释此行以使用服务的第 2 个版本。

在 VS Code 中重新运行 `SK_service_chat.py` 文件，并稍微修改你的查询，如下列所示。

##### 列表 5.27 `SK_service_chat.py`（`TMDb_v2` 服务输出）

```py
User:> get a list of currently playing movies for the 
action genre and only return movies about space     #1
Agent:> To find currently playing action movies that are specifically 
about space, I will need to manually filter the provided list for those 
that have space-related themes in their overview. Here's what fits that 
criteria from the list:

1\. **Rebel Moon - Part One: A Child of Fire**     #2
   - Release Date: 2023-12-15
   - Overview: When a peaceful colony on the edge of the galaxy finds 
itself threatened by the armies of the tyrannical Regent Balisarius, 
they dispatch Kora, a young woman with a mysterious past, to seek out 
warriors from neighboring planets to help them take a stand.

This is the only movie from the provided list that clearly mentions a 
space-related theme in its overview. …
```

#1 新查询要求包含额外的空间过滤器

#2 LLM 调用服务并审查匹配过滤器的返回结果。

由于语义服务函数现在以 JSON 格式返回完整的电影列表，LLM 可以应用额外的过滤。这是语义服务的真正力量，允许你通过 LLM 处理数据。我们不会仅仅通过返回标题列表来看到这种力量。

这个最后一个练习展示了在编写语义服务层时你需要做出的心态转变。通常，你通常会希望返回尽可能多的信息。返回更多信息可以利用 LLM 独立过滤、排序和转换数据的能力。在下一章中，我们将探讨使用行为树构建自主代理。

## 5.7 练习

完成以下练习以提高你对材料的了解：

+   *练习 1*—创建温度转换的基本插件

*目标*—熟悉创建简单的 OpenAI 聊天完成 API 插件。

*任务:*

+   +   开发一个在摄氏度和华氏度之间转换温度的插件。

    +   通过将其集成到简单的 OpenAI 聊天会话中测试插件，用户可以请求温度转换。

+   *练习 2*—开发天气信息插件

*目标*——学习创建执行独特任务的插件。

*任务:*

+   +   为 OpenAI 聊天完成 API 创建一个插件，从公共 API 获取天气信息。

    +   确保插件可以处理用户对不同城市当前天气状况的请求。

+   *练习 3*—制作一个创意语义功能

*目标*——探索创建语义功能。

*任务:*

+   +   开发一个基于用户输入写诗或讲述儿童故事的语义功能。

    +   在聊天会话中测试该函数，以确保它生成创意和连贯的输出。

+   *练习 4*—使用原生函数增强语义功能

*目标*——了解如何结合语义和原生函数。

*任务:*

+   +   创建一个使用原生函数增强其功能的语义功能。

    +   例如，开发一个生成餐单的语义功能，并使用原生函数获取食材的营养信息。

+   *练习 5*—使用语义内核封装现有的 Web API

*目标*——学习将现有的网络 API 封装为语义服务插件。

*任务:*

+   +   使用 SK 将新闻 API 封装并作为聊天代理中的语义服务插件公开。

    +   确保插件可以处理用户对各种主题的最新新闻文章的请求。

## 摘要

+   代理动作扩展了代理系统的功能，如 ChatGPT。这包括向 ChatGPT 和 LLM 添加插件以作为动作代理的能力。

+   OpenAI 支持在 OpenAI API 会话中定义函数和插件。这包括向 LLM API 调用中添加函数定义，并理解这些函数如何允许 LLM 执行额外的操作。

+   语义内核（SK）是微软的一个开源项目，可用于构建 AI 应用程序和代理系统。这包括语义插件在定义原生和语义函数中的作用。

+   语义函数封装了用于与 LLM 互动的提示/配置文件模板。

+   原生函数封装了使用 API 或其他接口执行或执行动作的代码。

+   语义函数可以与其他语义或原生函数结合，并在执行阶段相互层叠。

+   SK 可以在语义服务层上创建一个 GPT 接口，并在聊天或代理接口插件中暴露它们。

+   语义服务代表了 LLMs（大型语言模型）和插件之间的交互，以及这些概念在实际创建高效 AI 代理中的应用实现。
