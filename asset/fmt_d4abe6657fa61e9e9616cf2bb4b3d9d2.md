
```sql
CREATE VIEW combined_ach_records AS
SELECT 
    r1.ach_records_type_1_id AS primary_key, 
    r1.unparsed_record, 
    r1.sequence_number,
    r1.ach_files_id
FROM 
    ach_records_type_1 AS r1

UNION ALL

SELECT 
    r5.ach_records_type_5_id, 
    r5.unparsed_record, 
    r5.sequence_number,
    r1_r5.ach_files_id
FROM 
    ach_records_type_5 AS r5
JOIN ach_records_type_1 AS r1_r5 
    USING (ach_records_type_1_id)

…

UNION ALL

SELECT 
       r9.ach_records_type_9_id, 
       r9.unparsed_record, 
       r9.sequence_number,
       r1_r9.ach_files_id
  FROM 
       ach_records_type_9 AS r9
  JOIN ach_records_type_1 AS r1_r9 
 USING (ach_records_type_1_id)
```

`UNION ALL`将合并连续的`SELECT`语句的结果。

使用`USING`语句可以提供更简洁的语法，而不是使用`ON table.field = table.field`。在 PostgreSQL 中，当连接字段在表中具有相同的名称时，可以使用此功能。

![图片](img/CH05_F06_Kardell.png)

##### 图 5.6 更新后的表结构图

这段代码允许我们查看整个 ACH 文件，就像它被上传到系统中一样，没有任何我们的处理操作。有一种查看原始文件的方法，这允许我们向用户提供查看整个 ACH 文件内容以及异常记录的选项，以帮助他们调试任何问题。

## 5.10 编码更改

既然我们已经确定了数据库结构，那么编写任何额外的单元测试以及更新现有的测试以正确运行就是一个问题了。让我们看看为了支持这个结构所做的更改。

### 5.10.1 为未解析的 ACH 记录创建 Pydantic 架构

所有未解析的记录都共享`unparsed_record`和序列号字段。因此，这是一个创建一个将继承这些字段以避免每次都输入它们的类结构的好机会。我们创建了一个`AchRecordBaseSchema`。

##### 列表 5.32 我们 ACH 记录的基架构

```python
from abc import ABC
from pydantic import BaseModel

class AchRecordBaseSchema(ABC, BaseModel):
    unparsed_record: str
    sequence_number: int
```

导入`ABC`允许我们创建一个抽象类。

`BaseModel`是 Pydantic 所必需的。

我们的这个类从`ABC`和`BaseModel`继承。

所有 ACH 记录类中都将存在的字段

使用该架构，我们可以定义每个记录类型，如下所示。

##### 列表 5.33 ACH 记录架构

```python
from typing import Optional
from pydantic import UUID4

from ach_processor.schemas.ach_record.ach_record_base_schema import AchRecordBaseSchema

class AchRecordType1Schema(AchRecordBaseSchema):
    ach_records_type_1_id: Optional[UUID4] = None
    ach_files_id: UUID4
```

必须导入

由于它是`AchRecordBaseSchema`的子类，而`AchRecordBaseSchema`是`BaseModel`的子类，因此这是一个 Pydantic 类。

我们有一个标记为可选的 ID 字段，因为它将由数据库分配。

`ach_files_id`字段是必须的，因为它是一个外键，指向已上传的文件。

### 5.10.2 为解析的 ACH 记录创建 Pydantic 架构

在这个阶段，对于每个解析记录的 Pydantic 定义并不那么有趣，因为我们只是继承了 Pydantic 的`BaseModel`记录并定义了必要的字段。我们将在稍后扩展字段以用于约束、验证和文档。目前，我们只需保持它们简单。

##### 列表 5.34 ACH 批次头记录的 Pydantic 架构

```python
from pydantic import BaseModel, UUID4

class AchBatchHeaderSchema(BaseModel):
    ach_records_type_5_id: UUID4
    record_type_code: str
    service_class_code: str
    company_name: str
    company_discretionary_data: str
    company_identification: str
    standard_entry_class_code: str
    company_entry_description: str
    company_descriptive_date: str
    effective_entry_date: str
    settlement_date: str
    originator_status_code: str
    originating_dfi_identification: str
    batch_number: str
```

### 5.10.3 单元测试更改

作为代码重构的一部分，我们确保我们的测试也得到了清理（列表 5.35）。首先，我们更新了`setup_teardown_method`以将`autouse`设置为`true`，并确保首先执行`SqlUtils.truncate_all`方法。我们可能之前选择在测试运行后清除表，这是一个很好的实践，可以清理测试中的任何数据。然而，这也带来了一个不幸的副作用，即在测试失败时也会清理数据，这在我们想要在测试后检查数据库时并不是很有帮助。为了使调试和故障排除更容易，我们决定在测试之前清除数据。这也确保了数据库已准备好，因为我们不需要依赖之前的测试来自动清理。添加`autouse`参数意味着我们不再需要将 fixture 传递给我们的测试。我们还使用了`truncate_all`而不是特定的表，因为我们现在使用了多个表。

##### 列表 5.35 更新的`pytest`固定装置

```python
@pytest.fixture(autouse=True)
def setup_teardown_method():
    SqlUtils.truncate_all()
    yield
```

## 5.11 设计和不同方法

向数据库添加外键需要进行相当多的重工作，所以我们可能会考虑如何最小化这项工作，或者为什么我们没有一开始就添加它们。正如你所见，我们深入解析了 ACH 文件，并完成了这项工作，然后转向支持一些需要大量重工作的附加功能。这主要是因为我们想在深入研究整体功能和额外的数据库知识之前，先展示一些 ACH 基础知识。

然而，让我们考虑一下我们是否从功能角度来处理这个问题，以及用户是否需要上传这些文件。如果我们当时从`ach_files`表及其相关端点开始，我们就可以从一开始就包含外键。假设你具备处理 ACH 文件和 API 的知识和经验，这肯定是一个有效的方法。然后我们可以以同样的方式继续进行，唯一的区别是我们可能从一开始就有一个设计得更好的数据库。

这只是说明初始设计同样重要，以及你解决问题的方法。我们之前讨论过启用故事，或者也称为研究尖峰。它们可以降低风险，帮助我们理解需求，并更好地理解需要执行的任务。我们无法强调这些类型故事的重要性，尤其是在处理大型复杂系统时。我们总是可以预期在项目进行过程中需要一些重工作，无论是由于未识别的需求还是无法预见的范围变化。希望研究尖峰可以帮助最小化这些情况，但企业往往难以将时间投入到他们看不到即时收益的事情上。这显然会成为一个问题，因为通过启用故事暴露的问题将不会被识别，在 PI 规划中也不会被发现。我们最终可能采取的方法可能需要重工作，而这些重工作本可以在早期识别出来，因此我们的故事点可以正确分配。

我们已经看到了许多例子，一个项目似乎已经完成，但最终某个主题专家指出在系统演示中遗漏了某些内容，需要重工作。重要的是要记住，我们有时必须处理这些类型的状况，因为这可能会经常诱使我们选择一条更简单的出路。在这种情况下，我们需要评估变化对项目时间表、重工作量、风险、技术债务等因素的影响。例如，企业可能决定我们不需要数据库的引用完整性，并且编写一个可以手动运行的程序来检查悬空记录/缺失关系会更快。

## 摘要

+ 本章展示了如何创建一个数据库，该数据库可以存储我们的未更改记录，以及一个解析后的记录版本，这在需要提供加载的 ACH 文件详细信息时是有益的。

+ 我们在后期过程中看到了向数据库添加引用完整性的影响，以及需要重新编写我们的代码和单元测试的需求。当一个功能需要重新工作才能正确实现时，它通常可以被放在次要位置。

+ 尽管实现这些类型的功能需要额外的工作，但向团队成员和管理层倡导它们同样重要，以确保它们不会被忽视。

+ 定义数据库对于应用程序中的数据存储、查询和完整性至关重要。

+ 当作为平面文件处理时，ACH 文件在性能、查询和数据完整性方面面临挑战。

+ 关系数据库提供优势，例如主键和外键、约束和数据完整性。

+ 实现引用完整性可以防止孤立记录并确保数据库一致性。

+ 在评估数据库设计和各种实现方法时，研究激增（使能故事）是有益的。

+ ELT 和 ETL 在处理 ACH 文件和处理错误方面提供不同的好处。

+ Pydantic 帮助建模数据库表，抽象 SQL，并增强文档和验证。

+ 上传文件和集成 API 是扩展 ACH 系统功能的基础。

+ 数据和引用完整性对于关系数据库至关重要，可以防止错误并提高可靠性。

+ 持续测试、重构和重新审视初始设计选择有助于维护和改进数据库性能和结构。
