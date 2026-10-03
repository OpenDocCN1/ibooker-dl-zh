
在我们的 PPD 文件加载通过和 IAT 文件加载失败之后，我们开始着手支持 IAT 所需的更改。随着我们进行更改，我们将不断重新运行单元测试，以确保我们在解析 IAT 文件方面取得进展，并且我们没有无意中破坏任何东西。由于 ACH 文件是顺序处理的，我们需要首先处理的记录是批次头记录。我们添加了一个名为 `batch_type` 的新字段，我们将用它来解析批次头。以下列表显示了我们所添加的简单 `if`/`elif`/`else` 处理，根据 SEC 代码调用适当的例程，并在遇到我们不认识的头时记录异常。

### 列表 11.14 调用适当的解析例程

```py
if line[50:53] == "IAT": #1
 self._parse_iat_batch_header(ach_record_id, line) #2
 self.batch_type = "IAT" 
elif line[50:53] == "PPD": #2
    self._parse_batch_header(ach_record_id, line)  #3
 self.batch_type = "PPD" #3
else: #3
 self._add_exception( #4
 AchExceptionSchema( #4
 ach_files_id=ach_file_id, #4
 record_number=sequence_number, #4
 exception_code=AchExceptions.INVALID_BATCH_TYPE.value, #4
 ), #4
 line, #4
 ) #4
```

#1 无论批次头是 IAT 还是 PPD，SEC 代码都在相同的位置。这一行设置了适当的 `batch_type` 标志以适应 SEC 类型。

#2 之前，我们只是调用了 `_parse_batch_header` 方法。现在，我们确保我们正在处理一个 PPD 批次。

#3 我们希望在处理过程中记录异常，并且我们遇到了一个未识别的 SEC 代码。这可能是因为文件格式无效或我们尚未支持的代码。

在正确解析批次头的情况下，我们应该在 IAT 处理测试中前进一点，当然，PPD 处理不应该出错。我们遇到的下一个记录是类型 6 记录。随着 `batch_type` 变量的引入，它应该遵循相同的模式。请注意，我们继续像以前一样将未解析的记录写入数据库。然而，解析记录并将其写入适当表的需要取决于 `batch_type` 标志。以下列表显示了所需的更新。

### 列表 11.15 解析条目详情

```py
case "6": #1
    ach_record = AchRecordType6Schema(  #1
        ach_records_type_5_id=current_batch_header_id,  #1
        unparsed_record=line,  #1
        sequence_number=sequence_number,  #1
    )  #1
    ach_record_id = AchRecordsSqlType6()  #1
➥.insert_record(ach_record)  #1
    if self.batch_type == "IAT": #2
        self._parse_iat_entry_detail(  #2
            ach_file_id=ach_file_id,  #2
            current_batch_header_id=current_batch_header_id,  #2
            ach_records_type_6_id=ach_record_id,  #2
            sequence_number=sequence_number,  #2
            line=line,  #2
        )  #2
    elif self.batch_type == "PPD": #3
        self._parse_entry_ppd_detail(  #3
            ach_file_id=ach_file_id,  #3
            current_batch_header_id=current_batch_header_id,  #3
            ach_records_type_6_id=ach_record_id,  #3
            sequence_number=sequence_number,  #3
            line=line,  #3
        )  #3
    else: #4
        self._add_exception( #4
            AchExceptionSchema( #4
                ach_files_id=ach_file_id,  #4
                record_number=sequence_number,  #4
                exception_code=AchExceptions.INVALID_SEC.value,  #4
            ),  #4
            line,  #4
        )  #4
```

#1 我们未解析的记录的编写与我们正在处理的批次类型无关。

#2 当 `batch_type` 设置为 IAT 时，我们将调用适当的方法来解析条目。

#3 IAT 和 PPD 解析的参数完全相同。只有需要解析的字段不同。

#4 当我们遇到意外的 `batch_type` 时，添加一个异常

解析附加记录比解析批次头和条目详细记录稍微复杂一些。这是因为我们现在还必须处理不同的附加记录类型按照期望的顺序排列。类似于 `expected_record_type` 变量，我们也引入了一个 `expected_addenda_type` 变量。让我们直接跳到解析特定附加类型记录的方法，如下面的列表所示。在这里，这是标准的处理，增加了新的 `expected_addenda_type`。我们期望的下一条记录是另一个附加记录（类型 7），它应该是附加类型 11。

### 列表 11.16 `_parse_iat_addenda_710` 方法

```py
def _parse_iat_addenda_710(self, ach_records_type_7_id: UUID, line: str):
     self.expected_record_types = ["7"] #1
     self.expected_addenda_type = "11" #2

     ach_iat_addenda_record = AchRecordProcessor().parse_iat_addenda_710(
         ach_records_type_7_id, line
     )
     AchIat710AddendaSql().insert_record(ach_iat_addenda_record)
```

#1 `expected_record` 应该是另一个附加记录。

#2 `expected_addenda_type` 应该是下一个序列，因为记录需要按照特定的顺序排列。注意，我们只期望一种类型的记录，因此不需要像 `expected_record_type` 那样使用数组。

所有的附加记录将以相同的方式解析，`expected_addenda_type` 被设置为下一条记录。但是当我们到达最后的附加类型（类型 16）时会发生什么？下面的列表展示了我们如何重置我们的期望类型，一旦我们到达最后一个记录。

### 列表 11.17 重置期望类型

```py
self.expected_record_types = ["6", "7", "8"] #1
self.expected_addenda_type = "" #2
```

#1 在 IAT 处理中存在一些可选的附加记录，因此可能会遇到这些记录，以及新的条目或批次的结束。

#2 没有更多的强制附加记录，因此 `expected_addenda_type` 可以被设置为空字符串。

现在我们知道了如何解析附加记录，应该很容易将这些单独的方法组合起来——因为它们遵循与我们解析记录类型相似的方法。这个过程在下面的列表中展示。

### 列表 11.18 解析附加记录

```py
   def _parse_iat_addenda( #1
        self, ach_records_type_7_id: UUID,  #1
        line: str, sequence_number  #1
    ): 
        addenda_type = line[1:3] #2

        if addenda_type != self.expected_addenda_type: #3
            self._add_exception( 
                AchExceptionSchema( 
                    ach_files_id=ach_records_type_7_id, 
                    record_number=sequence_number, 
                    exception_code= 
                       AchExceptions.UNEXPECTED_ADDENDA_TYPE.value, 
                ) 
            ) 
            Return 

        if addenda_type == "10": #4
            self._parse_iat_addenda_710(ach_records_type_7_id, line)  #5
…  #5
        elif addenda_type == "16":  #5
            self._parse_iat_addenda_716(ach_records_type_7_id, line)  #5
        else:  #5
            self._add_exception(  #5
                AchExceptionSchema(  #5
                    ach_files_id=ach_records_type_7_id,  #5
                    record_number=sequence_number,  #5
                    exception_code=  #5
                        AchExceptions.INVALID_IAT_ADDENDA_TYPE.value,  #5
                )  #5
            )  #5
```

#1 定义我们的解析方法。这是一个总称，将调用其他方法来解析特定记录。

#2 提取附加类型，因为我们将会反复使用它

#3 如果附加类型不是我们期望的，我们需要记录一个异常。

#4 根据我们正在处理的附加类型，调用适当的方法。如果遇到意外的附加类型，它将记录不同的异常。

这样，我们应该能够解析包含 IAT 和 PPD 批次的文件。这些额外的 IAT 批次表将对我们的某些 API 产生影响。在下一节中，我们将简要看看具体哪些方面受到了影响。

## 11.6 对仪表板的影响

IAT 处理的增加如何影响我们仪表板的功能？这个问题可能需要与业务进行更多讨论。从 ACH 标准的角度来看，在同一文件中拥有 IAT 批次和 PPD 批次当然是可能的。然而，IAT 批次的字段和有趣信息可能不同，而且新表的出现确实会影响组件的功能。我们的组件是否应该包含所有批次或仅选择批次可能是一个需要来自最终用户输入的业务决策。目前，让我们看看我们如何将 IAT 批次纳入当前的仪表板组件中。

### 11.6.1 `get_batches` 方法

`/batches` 端点调用 `get_batches` 方法，该方法需要更新以包含 IAT 批次标题。我们知道 `ach_records_type_5_id` 只会存在于批次标题表中的一个。我们利用这一点和 `COALESCE` 命令来使用 `ach_batch_headers` 字段，如果它是 `NULL`，则使用 `ach_iat_batch_headers` 的值。这基于假设字段不会意外地成为 `NULL`。有关字段在数据库中都被标记为 `NOT NULL`。它们是否永远保持这种状态是另一个问题。目前，以下列表显示了我们可以如何更新我们的查询，而不会对当前的 API 过于侵入性。

### 列表 11.19 更新的 `get_batches`

```py
SELECT COALESCE(abh.company_name, '') AS company_name, #1
       COALESCE(abh.company_identification,       #2
       aibh.originating_dfi_identification) AS company_id,
       art5.ach_records_type_5_id AS id,
       COALESCE(abh.batch_number, aibh.batch_number) #3
                                           AS batch_number, 
       abcd.total_debit_entry_dollar_amount AS debit_total,
       abcd.total_credit_entry_dollar_amount AS credit_total, 
       abcd.entry_addenda_count 
  FROM ach_files AS af
…
LEFT JOIN ach_batch_headers AS abh #4
                USING (ach_records_type_5_id)  #4
LEFT JOIN ach_iat_batch_headers AS aibh  #4
                USING (ach_records_type_5_id)  #4
    WHERE af.ach_files_id = %s
 ORDER BY COALESCE(abh.originating_dfi_identification,  #5
                   aibh.originating_dfi_identification); 
```

#1 我们无法从标题记录中访问 IAT 的公司名称。

#2 如果公司识别信息不可用，则使用 `originating_dfi_identification`

#3 从批次中选择适当的批次号

#4 使用 `LEFT JOIN` 因为批次标题是针对一个表或另一个表的

#5 按适当的识别号排序

如此所示，我们有时可以更新现有的查询；然而，并非所有情况下都如此。实际上，这可能是例外。下一节将讨论当查询太大或太复杂，无法同时包含 PPD 和 IAT 时应该做什么。

### 11.6.2 批次条目

从端点 `/{file_id}/batches/{batch_id}/entries` 收集批次条目需要一些工作。构建 `AchBatchEntriesResponse` 的查询相当大，因此可能更有意义的是保留当前的查询，并创建一个针对 IAT 交易条目的新查询。因此，我们将采取 `get_entries` 方法，将现有的查询移动到 `_get_ppd_entries` 方法中，并创建一个新的 `_get_iat_entries` 方法。拆分的成果如下所示。

### 列表 11.20 更新的 `get_entries`

```py
@staticmethod
def get_entries(
    ach_file_id: UUID, ach_batch_id: UUID
    ) -> list[AchBatchEntriesResponse]:
    if AchFileSql._is_iat_batch(ach_file_id, ach_batch_id): #1
        return AchFileSql._get_iat_entries #2
➥(ach_file_id, ach_batch_id) 
    else:
        return AchFileSql._get_ppd_entries(ach_file_id, #3
                                           ach_batch_id) 
```

#1 确定文件中指定的批次是否为 IAT 批次的方法

#2 返回给定文件和批次的 IAT 条目的方法

#3 之前的 SQL 查询已移动到其自己的方法中。

在创建用于 IAT 记录的 `AchBatchEntriesResponse` 时，有一些细微的变化。对于获取附加记录数的原始 PPD 记录查询，我们必须自己收集这些信息。对于 IAT 批量，这些信息是记录的一部分；然而，我们仍然以相同的方式收集计数，部分是为了保持查询的相似性，但更重要的是，因为我们可能希望在将来对字段进行验证，这种方法提供了实现这一目标的机制（列表 11.21）。

### 列表 11.21 `_get_iat_entries` 查询
