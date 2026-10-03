
如库的总结所述，`pytest` 易于使用。`pytest` 期望一些命名约定。任何包含测试的文件都将有一个以 `test` 开头，后跟下划线或以下划线结尾后跟 `test` 的文件名。在测试文件中，`pytest` 将执行任何以 `test` 开头的函数名。

在测试函数内部，您将包含一个 `assert` 语句。如果它返回 `true`，则流程继续。如果测试中的所有断言都评估为 `true`，则测试以成功状态返回。如果断言评估为 `false`，则代码引发一个 `AssertionError`，并且测试评估为 `false`。

您的单元测试将非常基础：它们将检查从您的 `SQLAlchemy` 类返回的行数是否与您在之前的 `SQL` 查询中检查的值匹配。

创建一个名为 `test_crud.py` 的文件，内容如下：

```py
"""Testing SQLAlchemy Helper Functions"""
import pytest
from datetime import date

import crud
from database import SessionLocal

# use a test date of 4/1/2024 to test the min_last_changed_date.
test_date = date(2024,4,1)

@pytest.fixture(scope="function")
def db_session():
    """This starts a database session and closes it when done"""
    session = SessionLocal()
    yield session
    session.close()

def test_get_player(db_session):
    """Tests you can get the first player"""
    player = crud.get_player(db_session, player_id = 1001)
    assert player.player_id == 1001

def test_get_players(db_session):
    """Tests that the count of players in the database is what is expected"""
   players = crud.get_players(db_session, skip=0, limit=10000,
                               min_last_changed_date=test_date)
    assert len(players) == 1018

def test_get_players_by_name(db_session):
    """Tests that the count of players in the database is what is expected"""
    players = crud.get_players(db_session, first_name="Bryce", last_name="Young")
    assert len(players) == 1
    assert players[0].player_id == 2009

def test_get_all_performances(db_session):
 """Tests that the count of performances in the database is
   what is expected - all the performances"""
    performances = crud.get_performances(db_session, skip=0, limit=18000)
    assert len(performances) == 17306

def test_get_new_performances(db_session):
"""Tests that the count of performances in the database is
   what is expected"""
   performances = crud.get_performances(db_session, skip=0, limit=18000,
                                        min_last_changed_date=test_date)

#test the count functions
def test_get_player_count(db_session):
    player_count = crud.get_player_count(db_session)
    assert player_count == 1018
```

首先，看看这个文件是如何遵循 `pytest` 期望的约定的。文件被命名为 `test_crud.py`，因此它将被自动识别为测试文件。文件包含以 `test_` 开头的六个函数名。这些函数将在文件运行时被执行。这些测试函数每个都以一个 `assert` 语句结束。

第一个函数需要一些解释。在函数上方是装饰器 `@pytest.fixture(scope="function")`。在 *arrange* 阶段，即准备测试设置时，会使用到 `fixture`。此 `fixture` 使用 `session` 范围，这意味着它将为每个函数运行一次：

```py
@pytest.fixture(scope="function")
```

`db_session()` 函数的主体创建了一个数据库会话，在测试函数使用会话（通过 `yield` 语句）时暂停，并在测试完成后关闭会话：

```py
def db_session():
    """This starts a database session and closes it when done"""
    session = SessionLocal()
    yield session
    session.close()
```

为了验证基于日期的查询是否正常工作，`performance` 查询首先检查完整结果，然后使用 `last_changed_date` 限制的结果。首先记住，在之前的 `SQL` 查询中，你得到了以下 `performance` 表的结果：

```py
sqlite> select count(*) from performance;
17306
sqlite> select count(*) from performance where last_changed_date > '2024-04-01';
2711
```

要使用 `pytest` 验证第一个结果，这个函数不包括数据参数：

```py
def test_get_all_performances(db_session):
   """Tests that the count of performances in the database is
   what is expected - all the performances"""
    performances = crud.get_performances(db_session, skip=0, limit=18000)
    assert len(performances) == 17306
```

为了验证第二个结果，下一个函数使用了一个 `last_changed_date` 值为 `2024_04_01`，这个值设置在测试代码顶部的 `test_date` 变量中。这个日期比所有记录都要早 2,711 条：

```py
 """Tests that the count of performances in the database is
   what is expected"""
   performances = crud.get_performances(db_session, skip=0, limit=10000,
                                        min_last_changed_date=test_date)
    assert len(performances) == 2711
```

最后一个测试验证了一个分析查询：

```py
def test_get_player_count(db_session):
    player_count = crud.get_player_count(db_session)
    assert player_count == 1018
```

要执行测试，请输入 `pytest test_crud.py` 命令，你应该会看到一个类似以下输出的结果：

```py
$ pytest test_crud.py
================== test session starts ===========================
platform linux -- Python 3.10.13, pytest-8.1.2, pluggy-1.5.0
rootdir: /workspaces/adding-more-data/chapter3
plugins: anyio-4.4.0
collected 5 items

test_crud.py                                              [100%]

=================== 5 passed in 0.22s ============================
```

你已经验证了你的 `SQLAlchemy` 类和一些辅助函数工作正常——做得好！数据库工作已完成。

# 其他资源

`SQL` 是数据专业人士必备的一项基本技能。可用的资源数量无限，但以下是一些开始的地方：

+   《*学习 SQL，第 3 版*》，由艾伦·博利厄（O’Reilly，2020 年）

+   《*SQL 实用指南，第 4 版*》，由爱丽丝·赵（O’Reilly，2021 年）

要了解更多关于 `SQLAlchemy` 的信息，请查看 [官方 SQLAlchemy 2 文档](https://oreil.ly/PhsUf)。

# 摘要

在本章中，你创建了数据库和用于读取的 `SQLAlchemy` 代码。以下是本章你取得的成就：

+   你设计了数据库表及其关系。

+   你使用 `SQLite` 创建了一个数据库，并使用 `SQL` 命令创建了所有表。

+   你从 `CSV` 文件中导入数据以加载数据表。

+   你创建了 `Python` 模型文件和数据库配置文件。

+   你创建了查询数据库的辅助函数。

+   你使用 `pytest` 对端到端数据库功能进行了单元测试。

在第四章，你将创建 `FastAPI` 代码来使用这些数据并将其发布为 `REST API`。
