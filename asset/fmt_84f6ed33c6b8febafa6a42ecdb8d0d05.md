
如我们所见，`AiServices`构建器开始展示其价值，它允许我们设置我们想要使用的模型和我们想要代理使用的工具。我们还更新了输入提示的方式，以便应用程序可以持续运行，我们可以用不同的指令测试我们的新代理。因此，当我们运行代理并被询问

| ![](img/logo-openai.png) | 您需要什么？ |
| --- | --- |

我们可以提交我们的提示：

| ![](img/logo-MW.png) | 你能为我创建 4 个房间和 2 个预订，并告诉我数据库里有什么吗 |
| --- | --- |

将返回以下响应：

| ![](img/logo-openai.png) | 你希望我创建 4 个房间。你希望我创建 2 个预订。然后我会分享当前数据库的详细信息 |
| --- | --- |

让我们分析一下这个输出是如何生成的。首先，我们将我们的提示和工具发送到`gpt-3.5-turbo`进行处理。然后，模型评估提示的细节，并查看我们使用`@Tool`注解标记的工具列表，以找到与我们的指令相关的工具。在提示的开始，我们请求“你能为我创建 4 个房间”，LLM 确定我们的`createRooms`工具应该运行，因为它与注解`@Tool("Create room records")`相关。接下来，我们可以看到输出正确地说明了我们想要创建四个房间。这是因为我们通过 LangChain 的`@P`注解以`@P("Amount of room records to create") int count`的形式将一个参数传递给我们的`createRooms`方法。注意我们如何再次以类似`@Tool`注解的方式为`@P`注解提供自然语言上下文。这允许 LLM 进行类似的相关性匹配，从我们的提示中提取它认为必要的数据，并将其作为参数输出。

我们现在也可以通过给出不同的指令来测试我们的智能体自主确定使用哪个工具的能力。这次，当被要求给出一个提示时，我们发送了

| ![](img/logo-MW.png) | 你能告诉我数据库当前有什么吗 |
| --- | --- |

运行此操作将返回以下响应：

| ![](img/logo-openai.png) | 然后，我会分享当前数据库的详细信息 |
| --- | --- |

在这种情况下，LLM 只触发了我们的一个工具，即被标注为`@Tool("Show results of database")`的`displayDatabase`。由于我们在提示中未提及创建房间或预订，相关的工具被认为与我们的指令不相关，因此被忽略。这展示了智能体的强大能力。想象一下，如果我们不仅仅有 3 个工具，而是有 10 个、20 个或更多，那么我们给智能体提供反应我们指令和解决我们请求的问题的方式就更多了。

### 9.2.2 给我们的 AI 智能体执行函数

我们已经设置了决策过程，现在让我们完成智能体，并给它执行一些数据库查询的能力。为此，我们将使用`h2`创建一个模拟数据库，其中包含一些基本表，以展示我们如何让我们的 LLM 为我们执行操作。为此，我们将首先创建一个新的类`QueryTools`并添加以下代码：

```py
public class QueryTools {

    private final Connection connection;

    public QueryTools() throws SQLException {                            ❶
        connection = DriverManager.getConnection("jdbc:h2:mem:testdb");  ❶
        Statement st = connection.createStatement();                     ❶
        st.executeUpdate("""                                             ❶
            CREATE TABLE BOOKINGS (                                      ❶
                bookingid int NOT NULL AUTO_INCREMENT,                   ❶
                roomid int,                                              ❶
                firstname varchar(255),                                  ❶
                lastname varchar(255),                                   ❶
                depositpaid boolean,                                     ❶
                checkin date,                                            ❶
                checkout date,                                           ❶
                primary key (bookingid)                                  ❶
            );                                                           ❶
            CREATE TABLE ROOMS (                                         ❶
                roomid int NOT NULL AUTO_INCREMENT,                      ❶
                room_name varchar(255),                                  ❶
                type varchar(255),                                       ❶
                beds int,                                                ❶
                accessible boolean,                                      ❶
                image varchar(2000),                                     ❶
                description varchar(2000),                               ❶
                features varchar(100) ARRAY,                             ❶
                roomPrice int,                                           ❶
                primary key (roomid)                                     ❶
            );                                                           ❶
        """);                                                            ❶
    }                                                                    ❶

    public void createRoom() throws SQLException {                       ❷
        Statement st = connection.createStatement();                     ❷
        st.executeUpdate("""                                             ❷
            INSERT INTO ROOMS (room_name, type, beds, accessible,        ❷
            ➥image, description, features, roomPrice)                   ❷
            VALUES (                                                     ❷
                '101',                                                   ❷
                'single',                                                ❷
                1,                                                       ❷
                true,                                                    ❷
                '/images/room2.jpg',                                     ❷
                'A generated description',                               ❷
                ARRAY['TV', 'WiFi', 'Safe'],                             ❷
                100);                                                    ❷
        """);                                                            ❷
    }                                                                    ❷

    public void createBooking() throws SQLException {                    ❸
        Statement st = connection.createStatement();                     ❸
        st.executeUpdate("""                                             ❸
            INSERT INTO BOOKINGS (roomid, firstname, lastname,           ❸
            ➥depositpaid, checkin, checkout)                            ❸
            VALUES (                                                     ❸
                1,                                                       ❸
                'James',                                                 ❸
                'Dean',                                                  ❸
                true,                                                    ❸
                '2022-02-01',                                            ❸
                '2022-02-05'                                             ❸
            );                                                           ❸
        """);                                                            ❸
    }                                                                    ❸

    public void outputTables(String query) throws SQLException {         ❹
        Statement st = connection.createStatement();                     ❹
        ResultSet rs = st.executeQuery(query);                           ❹
        ResultSetMetaData rsmd = rs.getMetaData();                       ❹

        int columnsNumber = rsmd.getColumnCount();                       ❹
        while (rs.next()) {                                              ❹
            for(int i = 1 ; i <= columnsNumber; i++){                    ❹
                System.out.print(rs.getString(i) + " ");                 ❹

            }                                                            ❹
            System.out.println();                                        ❹
        }                                                                ❹
    }                                                                    ❹
}
```

❶ 启动时创建包含必要表的数据库

❷ 创建房间的基本方法

❸ 创建预订的基本方法

❹ 输出每个表内容的基本方法

在创建了`QueryTools`类之后，我们就可以通过更新`DataAssistantTools`来扩展我们的工具，以便以我们喜欢的方式与数据库交互。
