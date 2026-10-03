
# 9.2.2 将工具连接到数据库

以下代码展示了`DataAssistantTools`类的完整实现，该类通过`QueryTools`工具类与数据库进行交互：

```py
public class DataAssistantTools [

    QueryTools queryTools = new QueryTools();                            ❶

    public DataAssistantTools() throws SQLException {                    ❶
    }                                                                    ❶

    @Tool("Create room records")                                         ❶
    public void createRooms(@P("Amount of room records to create")       ❶
    ➥int count) throws SQLException {                                   ❶

        for(int i = 1; i <= count; i++){                                 ❷
            queryTools.createRoom();                                     ❷
        }                                                                ❷
    }                                                                    ❷

    @Tool("Create booking records")                                      ❷
    public void createBookings(@P("Amount of booking records to          ❷
    ➥create") int count) throws SQLException {                          ❷

        for(int i = 1; i <= count; i++){                                 ❸
            queryTools.createBooking();q                                 ❸
        }                                                                ❸
    }                                                                    ❸

    @Tool("Show results of database")                                    ❸
    public void displayDatabase() throws SQLException {                  ❸

        System.out.println("Current ROOM database state:");              ❹
        queryTools.outputTables("SELECT * FROM ROOMS");                  ❹

        System.out.println("Current BOOKING database state:");           ❹
        queryTools.outputTables("SELECT * FROM BOOKINGS");               ❹
    }                                                                    ❹

}                                                                        ❹
```

❶ 在启动时创建数据库的新实例

❷ 通过循环请求的房间数量创建房间

❸ 通过循环请求的预订数量创建预订

❹ 输出每个表的内容

最后，我们更新`DataAssistant`类中的`main`方法以处理`SQLException`，结果如下完成代码：

```py
public static void main(String[] args) throws SQLException 

        OpenAiChatModel model = OpenAiChatModel
                .builder()
                .apiKey("API-KEY-HERE")
                .modelName(OpenAiChatModelName.GPT_3_5_TURBO)
                .build();

        DataAssistantService dataAssistantChat =
        ➥AiServices.builder(DataAssistantService.class)
                .chatLanguageModel(model)
                .tools(new DataAssistantTools())
                .build();

        while(true){
            Scanner scanner = new Scanner(System.in);
            System.out.println("What do you need?");

            String query = scanner.nextLine();
            String response = dataAssistantChat.sendPrompt(query);
            System.out.println(response);
        }
    }
```

一切就绪后，我们现在可以通过首先请求它创建我们所需的数据来测试代理：

| ![](img/logo-MW.png) | 创建我 2 个房间和 3 个预订 |
| --- | --- |

然后，它会产生如下响应：

| ![](img/logo-openai.png) | 成功创建了两个房间和三个预订。 |
| --- | --- |

我们可以通过向代理发送额外的指令来确认我们的数据已被创建：

| ![](img/logo-MW.png) | 列出数据库内容 |
| --- | --- |

结果输出看起来类似于：

| ![](img/logo-openai.png) | 当前房间数据库状态：101 单间 1 真实 /images/room2.jpg 生成描述 [电视, WiFi, 安全] 100101 单间 1 真实 /images/room2.jpg 生成描述 [电视, WiFi, 安全] 100 当前预订数据库状态：1 詹姆斯·迪恩 真实 2022-02-01 2022-02-051 詹姆斯·迪恩 真实 2022-02-01 2022-02-051 詹姆斯·迪恩 真实 2022-02-01 2022-02-05 |
| --- | --- |

现在，我们的数据库查询已经连接到我们的工具中，我们可以与我们的代理交互，并让他们执行我们的任务。然而，让我们更进一步，让我们的代理能够运行多个工具链，利用一个工具中创建的数据在另一个工具中使用。

### 9.2.3 将工具链接在一起

目前，我们的工具是相互独立的。房间工具创建房间，并为每一行生成唯一的`roomid`键，但我们创建新的预订时并没有使用它们。我们只是简单地硬编码我们的值。因此，为了使我们的代理更加动态，并给它一个更复杂的问题，让我们看看我们如何将最近创建的房间的`roomid`传递给预订工具。

首先，我们需要在`QueryTools`类中创建一个额外的方法，该方法将返回最近创建的房间的`roomid`，如果没有房间在数据库中，则返回`id`为`0`：

```py
public int getRoomId() throws SQLException 
    Statement st = connection.createStatement();
    ResultSet rs = st.executeQuery("SELECT roomid FROM ROOMS
    ➥ORDER BY roomid DESC");

    if(rs.next()){
        return rs.getInt("roomid");
    } else {
        return 0;
    }
}
```

在我们的新方法到位后，我们接下来使用`DataAssistantTools`创建一个新的工具：

```py
@Tool("Get most recent roomid from database after
➥rooms have been created")
public int getRoomId() throws SQLException {
        return queryTools.getRoomId();
}
```

注意我们在`@Tool`注解中设置，我们期望在创建房间之后运行此提示。虽然这并不保证我们的 LLM 按特定顺序运行工具，但如果我们要按特定顺序创建房间和预订，它充当了一个安全网。此外，我们将`roomid`整数返回给 LLM。然后，我们将它在原始提示的上下文中存储起来以供将来使用，我们将通过更新我们的`createBookings`方法来实现这一点：

```py
@Tool("Create booking records")
public void createBookings(@P("Amount of booking records to create")
➥int count, @P("Most recent roomid") int roomid) throws SQLException {
    System.out.println("I will create the bookings for room: " +
    ➥roomid);
    for(int i = 1; i <= count; i++){
        queryTools.createBooking(roomid);
    }
}
```

在此更新方法中，我们添加了一个新的参数，形式为`@P("Most recent roomid") int roomid`。为了看到`roomid`提取的实际效果，我们在将其传递给`createBooking`用于我们的`INSERT`语句之前，使用`System.out`打印出`roomid`。为了使用`roomid`，我们返回到`QueryTools`并更新`createBooking`方法，使其接受`roomid`作为参数并将其添加到我们的 SQL 语句中：

```py
public void createBooking(int roomid) throws SQLException 
    Statement st = connection.createStatement();
    st.executeUpdate("""
            INSERT INTO BOOKINGS (roomid, firstname, lastname,
            ➥depositpaid, checkin, checkout)
            VALUES (
                ?,
                'James',
                'Dean',
                true,
                '2022-02-01',
                '2022-02-05'
            );
        """.replace("?", Integer.toString(roomid)));
}
```

现在我们已经为我们的代理编写了代码，使其能够使用一个工具从我们的数据库中提取`roomid`，然后使用第二个工具中的该值，我们可以在以下示例聊天中看到这一功能是如何工作的：

| ![](img/logo-openai.png) | 你需要什么？ |
| --- | --- |

| ![](img/logo-MW.png) | 创建三个房间 |
| --- | --- |

| ![](img/logo-openai.png) | 已成功创建三个房间。 |
| --- | --- |

| ![](img/logo-openai.png) | 你需要什么？ |
| --- | --- |

| ![](img/logo-MW.png) | 列表数据库 |
| --- | --- |

| ![](img/logo-openai.png) | 当前房间数据库状态：101 单人间 1 可用 /images/room2.jpg 生成描述 [电视，WiFi，保险箱] 100101 单人间 1 可用 /images/room2.jpg 生成描述 [电视，WiFi，保险箱] 100101 单人间 1 可用 /images/room2.jpg 生成描述 [电视，WiFi，保险箱] 100 当前预订数据库状态：数据库已成功显示。 |
| --- | --- |

| ![](img/logo-openai.png) | 你需要什么？ |
| --- | --- |

| ![](img/logo-MW.png) | 创建三个预订 |
| --- | --- |

| ![](img/logo-openai.png) | 我将为房间：3 创建预订。当前房间数据库状态：

```py
101 single 1 TRUE /images/room2.jpg A generated description [TV, WiFi, Safe] 100 
101 single 1 TRUE /images/room2.jpg A generated description [TV, WiFi, Safe] 100 
101 single 1 TRUE /images/room2.jpg A generated description [TV, WiFi, Safe] 100 
```

当前预订数据库状态：

```py
3 James Dean TRUE 2022-02-01 2022-02-05 
3 James Dean TRUE 2022-02-01 2022-02-05 
3 James Dean TRUE 2022-02-01 2022-02-05 
```

已成功创建三个预订。 |
| --- | --- |

分析输出，我们可以看到创建了三个房间，我们通过询问代理列出当前房间来确认这一点。然后，代理被要求创建三个预订，它成功地完成了这一任务。注意在`Current BOOKING database state:`之下的每一行的第二个整数。看看它们现在是如何使用`roomid: 3`的。代理已成功利用这两个工具查询最新的`roomid`，在将检索到的`id`传递给第二个工具（为我们创建预订的工具）之前。我们可以通过创建更多的房间来确认这不是巧合，以确认在创建的额外预订中添加了更新、更近的`roomid`（在这个例子中，`roomid`增加到`6`）。

| ![](img/logo-openai.png) | 你需要什么？ |
| --- | --- |

| ![](img/logo-MW.png) | 创建另外三个房间 |
| --- | --- |

| ![](img/logo-openai.png) | 已成功创建三个房间。以下是更新后的数据库： |
| --- | --- |

| ![](img/logo-openai.png) | 你需要什么？ |
| --- | --- |

| ![](img/logo-MW.png) | 创建另外三个预订 |
| --- | --- |

| ![](img/logo-openai.png) | 我将为房间：6 创建预订。三个预订已成功创建。 |
| --- | --- |

| ![](img/logo-openai.png) | 你需要什么？ |
| --- | --- |

| ![MW 标志](img/logo-MW.png) | 列表数据库 |
| --- | --- |

| ![OpenAI 标志](img/logo-openai.png) | 当前 ROOM 数据库状态：101 单间 1 真实 /images/room2.jpg 生成描述 [电视，WiFi，保险箱] 100101 单间 1 真实 /images/room2.jpg 生成描述 [电视，WiFi，保险箱] 100101 单间 1 真实 /images/room2.jpg 生成描述 [电视，WiFi，保险箱] 100101 单间 1 真实 /images/room2.jpg 生成描述 [电视，WiFi，保险箱] 100101 单间 1 真实 /images/room2.jpg 生成描述 [电视，WiFi，保险箱] 100Current BOOKING 数据库状态：3 詹姆斯·迪恩 真实 2022-02-01 2022-02-053 詹姆斯·迪恩 真实 2022-02-01 2022-02-053 詹姆斯·迪恩 真实 2022-02-01 2022-02-056 詹姆斯·迪恩 真实 2022-02-01 2022-02-056 詹姆斯·迪恩 真实 2022-02-01 2022-02-056 詹姆斯·迪恩 真实 2022-02-01 2022-02-05 |
| --- | --- |
