
# 解决数据返回但显示不正确的问题

让我们来看看能否精准定位这个“方枘圆凿”的问题。

![数据返回但显示不正确](img/CH07_F03_Kardell.png)

**图 7.3** 数据已返回但显示不正确

我们可以通过 OpenAPI 文档页面（或诸如 Postman 之类的工具）来探索 `/files` API 返回的数据。下面的清单展示了该 API 返回的数据。

**清单 7.1** 从 `/files` 端点返回的数据

```json
[
  {
    "ach_files_id": "dcb8e6a4-79e9-4cf7-a138-491e58c1ebb8",
    "file_name": "sample.ach",
    "file_hash": "application/octet-stream",
    "created_at": "2024-02-18T01:05:37.513025"
  }
]
```

如果我们将这段代码与 `RecentAchUploads.tsx` 中的 `AchUpload` 接口进行比较，就能看出差异。我们不仅缺少诸如 `creditTotal` 和 `debitTotal` 之类的字段，还需要映射一些字段名称。

**清单 7.2** `AchUpload` 接口

```typescript
接口 AchUpload {
    id: number;
    date: string;
    filename: string;
    creditTotal: number;
    debitTotal: number;
}
```

我们还应该审查数据库的配置方式，并将文件控制记录（类型 9）中的金额字段从 `VARCHAR` 改为 `NUMERIC`。这在一定程度上是个人的选择，因为有人可能会主张使用更灵活的字符值，因为字段本身可能格式不正确。进行这种更改是有益的，因为数据库现在将强制数据为数字类型。然而，我们可能会遇到以前能正常工作的现有功能出错的情况。为什么？因为可能我们之前无意中在该字段中放入了空格或其他非数字数据。当然，希望我们的单元测试能够防止这种情况，或者至少能迅速指出可能需要调整的地方。

在我们的案例中，我们试图创建一个既能处理未解析记录（用于记录无法解析的情况），又能处理更详细的已解析记录的数据库结构。下面的清单展示了数字字段的更新约束。请注意，这些字段长度为 12 个字符，精度为 2。

**清单 7.3** 文件控制记录的 ACH 表

```sql
total_debit_entry_dollar_amount NUMERIC(12, 2) NOT NULL,
total_credit_entry_dollar_amount NUMERIC(12, 2) NOT NULL,
```

解析程序也需要更新（见清单 7.4）。我们使用 `Decimal` 数据类型，因为它比使用浮点数更安全（[`docs.python.org/3/library/decimal.xhtml`](https://docs.python.org/3/library/decimal.xhtml)）。请注意，我们将字段作为字符串传递给 `Decimal`，而不是将其转换为 `int`/`float`，以避免意外转换。例如，`Decimal('0.1')` 和 `Decimal(0.1)` 的结果分别是 `Decimal('0.1')` 和 `Decimal('0.1000000000000000055511151231257827021181583404541015625')`，因此我们仍然需要小心。

**清单 7.4** 将字符串转换为包含小数点的数字

```python
total_debit_entry_dollar_amount= #1
    Decimal(f"{record[31:41]}.{record[41:43]}"),  #1
total_credit_entry_dollar_amount=  #1
    Decimal(f"{record[43:53]}.{record[53:55]}"),  #1
```
#1 使用 Decimal 对象和 f-strings 解析字段

不得不更新数据库，将存储的字符串转换为适当的数字类型，这增加了额外的工作量。如果我们在之前研究数据库时的技术预研中处理过这个问题，本可以减少一些工作量。此外，我们可能会问自己，对于一个 MVP 来说，将这些字段转换为数字类型是否值得。这是一个合理的问题。虽然我们至少能获得数据库强制数据格式化的好处，但代价是开发时间的增加。

不幸的是，没有正确的答案。根据项目、需求和时间表，您可能没有额外的时间来转换字段，并可能选择稍后再做。在这种情况下，我们转换字段，因为无论如何我们都需要更新 API 以返回这些字段。我们可以进行更改，并且根据 UI 中定义的接口（清单 7.2），我们知道期望的是数字类型。

现在我们已经正确存储了数据，我们需要将这些字段作为查询的一部分返回。这把我们带到了另一个需要重构的领域。我们创建了模式（schemas），并将它们用于数据库和表示从 API 返回的数据。我们希望保持模型分离，因为这种关注点分离为我们提供了更大的灵活性、更好的安全性和数据完整性。我们重构 `schemas` 目录，使其包含 `api` 和 `database` 文件夹，并将所有现有模式移动到 `database` 文件夹，结果结构如下清单所示（请注意，并非所有文件都显示）。

**清单 7.5** 新的 schemas 目录结构

```
schemas/
├─ api/
├─ database/
│  ├─ ach_record/
│  │  ├─ ach_record_base_schema.py
│  │  ├─ ...
│  ├─ ach_file_schema.py
│  ├─ ...
```

有了更新后的结构，我们就可以自由地添加我们的 API 响应了。我们现在可以继续按计划更新 API。如果您还记得，我们想要返回一个包含文件总额的响应。因此，我们创建了包含 `AchFilesResponse` 的 `ach_files_response.py`，如下清单所示。我们还利用更多 Pydantic 的功能，为字段提供文档说明。

**清单 7.6** `AchFilesResponse` 类

```python
from decimal import Decimal
from datetime import datetime
from pydantic import BaseModel, UUID4, Field

class AchFilesResponse(BaseModel):
    id: UUID4 = Field(...,  #1
        description="ACH 文件的唯一标识符", #2
        title="ID")  #3
    date: datetime = Field(...,
        description="上传 ACH 文件的日期和时间",
        title="日期")
    filename: str = Field(...,
        title="文件名",
        description="加载 ACH 文件的文件名",
        max_length=255, #4
    )
    credit_total: Decimal = Field(
        ...,
        description="ACH 文件中信用交易的总金额",
        ge=0, #5
        title="信用总额",
    )
    debit_total: Decimal = Field(
        ...,
        description="ACH 文件中借记交易的总金额",
        ge=0,
        title="借记总额",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "date": "2024-01-01T12:00:00",
                "filename": "ACH_20240101_123.ach",
                "creditTotal": "1000.00",
                "debitTotal": "5000.23",
            }
        }
```
#1 通过使用 Pydantic 的 Field，我们可以指定约束和文档。
#2 字段的描述，显示为 API 文档的一部分
#3 字段的标题，显示在我们的 API 文档中
#4 对字段的约束；长度超过 max_length 的字符串将导致验证错误。
#5 将字段约束为 ≥0

我们之前定义了一个简单的查询来返回 `ach_files` 表中的所有字段，但现在我们想要在响应中包含信用和借记总额，我们需要从已解析的类型 9 记录中提取它们。为了理清查询，我们可以使用 CloudBeaver 对我们的数据库运行查询，并确保结果是正确的。一旦我们开始提取所需数据，我们就可以创建一个新方法来返回检索到的数据。

**清单 7.7** 更新 SQL 查询

```python
def get_files_response(
    self, limit: Optional[int] = None, offset: Optional[int] = None
) -> list[AchFilesResponse]:
    with get_db_connection(row_factory=dict_row) as conn:
        result = conn.execute(
            """
            SELECT af.ach_files_id AS id, #1
                   af.file_name AS filename,  #1
                   af.created_at AS date,  #1
                   afcr.total_debit_entry_dollar_amount AS debit_total,  #1
                   afcr.total_credit_entry_dollar_amount AS credit_total #1
            FROM ach_files AS af
            INNER JOIN ach_records_type_1 AS art1 USING (ach_files_id) #2
            INNER JOIN ach_records_type_9 AS art9
                USING (ach_records_type_1_id)
            INNER JOIN ach_file_control_records AS afcr
                USING (ach_records_type_9_id)
            ORDER BY af.created_at DESC  #3
            LIMIT %s  #4
            OFFSET %s
            """,
            [limit, offset],
        )
        return result.fetchall()
```
#1 字段被重命名以匹配 AchFilesResponse。
#2 当连接的两个表中列名相同时，我们可以使用 USING。
#3 按 created_at 日期降序排序我们的结果
#4 虽然目前未使用，但我们可以限制结果数量，并在数据量大时允许分页。

最后，我们更新路由，不仅调用新的 `get_files_response`，还为最终用户提供更多文档，如下清单所示。这个文档步骤很重要，因为它使 API 对非技术利益相关者（例如可能帮助我们测试 API 的 QA 部门，或可能帮助我们编写文档的技术文档工程师）更易访问。此外，此更改可以使那些无法查看底层代码的 API 消费者受益。

**清单 7.8** 更新 API 路由

```python
@router.get( #1
    "",  #1
    response_model=list[AchFilesResponse],  #1
    summary="检索上传的 ACH 文件",  #1
    description="检索 ACH 文件的详细信息，包括信用/借记总额",  #1
    response_description="请求的 ACH 文件的详细信息"  #1
)
async def read_files() -> list[AchFilesResponse]:
    return AchFileSql().get_files_response()
```
#1 我们可以再次使用 Pydantic 来记录我们的字段。

至此，我们拥有了一个带有文档的可用 API。图 7.4 展示了我们使用 Pydantic 获得的部分文档结果。因此，除了能够帮助我们进行数据验证、解析和类型安全之外，Pydantic 还提供了这些文档方面的好处。

![API 文档截图](img/CH07_F04_Kardell.png)

**图 7.4** 我们的 API 现在有了带文档的字段和示例

我们快完成了——只需在 UI 侧进行一些最后的润色，一切就会连接起来。

在 `RecentAchUploads.tsx` 中，我们将接口命名为 `AchUploads`。我们希望将其命名为更接近 API 返回内容的名称。我们还有一个映射问题：字段定义为 `creditTotal`，但从 API 返回的是 `credit_total`。虽然 Python 使用 `snake_case` 作为变量名，但 TypeScript 更倾向于 `camelCase`。因此，我们将 `AchUploads` 重命名为 `AchFiles`，并保持字段不变。然后，我们创建一个新接口——`AchFilesResponse`。

**清单 7.9** `AchFilesResponse` 接口

```typescript
接口 AchFilesResponse {
    id: number;
    date: string;
    filename: string;
    credit_total: number;
    debit_total: number;
}
```

接下来，我们更新组件中对 `/files` 端点的调用，将响应从 `AchFilesResponse` 转换为 `AchFiles`。我们这样做是为了保持我们所使用语言的命名标准一致。然而，这也能带来另一个好处——在 UI 和 API 之间提供一层抽象。如果我们更换 API 供应商（在这种情况下不太可能），或者 API 内部发生意外变化（API 未保持向后兼容，或进行了未宣布的更改），这种好处就很有价值。下面的清单展示了更新后的 `axios` 调用。

**清单 7.10** 更新 `axios` 调用

```typescript
axios.get<AchFilesResponse[]>(`${apiUrl}/files`)
    .then(response => {
      console.log(`Response data ➥ ${JSON.stringify(response.data)}`); #1
      const transformedData: AchFiles[] = ➥ response.data.map((row: AchFilesResponse) => ({ #2
        id: row.id, #2
        date: row.date, #2
        filename: row.filename, #2
        credit
```
