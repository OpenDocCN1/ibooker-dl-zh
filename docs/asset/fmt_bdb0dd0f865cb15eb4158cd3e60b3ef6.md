
在运行 DAG 之前，请确保你的 API 正在运行。导航回 Airflow UI，你将看到你的 DAG 列出，如图 图 10-8 所示。用户界面功能众多，本章无法全部涵盖，但你可以在 [“UI / Screenshots”](https://oreil.ly/DfOSC) 中了解用户界面。

![DAG 列在 Airflow 主页上](img/haad_1008.png)

###### 图 10-8\. DAG 列在 Airflow 主页上

点击 `recurring_player_api_insert_update_dag`，然后点击图形。你将看到使用你在代码中指定的 `task_id` 名称的 Airflow 任务序列，如图 图 10-9 所示。

![第一个 DAG 的图形视图](img/haad_1009.png)

###### 图 10-9\. 第一个 DAG 的图形视图

点击触发 DAG 按钮，该按钮有一个指向你的 DAG 的三角形图标。如果代码和连接配置正确，你的 DAG 中的每个任务应在约一分钟内完成，绿色框表示成功。点击第一个框，标记为 `check_api_health_check_endpoint`。你的视图应类似于 图 10-10。如果你遇到错误，点击有错误的任务，然后点击日志来诊断问题。

![成功的 DAG 运行](img/haad_1010.png)

###### 图 10-10\. 成功的 DAG 运行

为了确认你的分析数据库已成功更新，请回到终端并使用 SQLite 打开数据库。查询 Player 表，以确认表中存在 1,018 条玩家记录。这些是从你的 API 中检索到的记录：

```py
.../dags (main) $ sqlite3 analytics_database.db
SQLite version 3.45.3 2024-04-15 13:34:05
Enter ".help" for usage hints.
sqlite> select count(*) from player;
1018
```

恭喜！你创建了一个使用 API 记录更新数据库的数据管道！

# 摘要

在本章中，你学习了如何创建一个数据管道，通过调用 API 来维护分析产品的当前数据。你安装并配置了 Apache Airflow，并创建了一个包含多个任务的 DAG，用于从 API 更新你的数据库。

在 第十一章 中，你将使用 API 的数据创建一个 Streamlit 数据应用。
