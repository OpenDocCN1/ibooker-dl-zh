
接下来，让我们处理表格本身。列表 11.4 显示了`ach_iat_addenda_712_records`表。其中一些 IAT 附加记录是 ACH 标准的独特之处，因为它们包含在固定记录中分隔的字段。奇怪，对吧？

以前，其他格式严格是固定长度的记录。因此，我们可以说这 15 个字符是名称，并将存储在名称字段中。在这些附加记录中，我们有一个固定长度字段，例如 35 个字符的“发起城市及州/省”，它包含城市和州。数据元素由星号`*`分隔，反斜杠`\`是最后一个元素的终止符。这导致了一个具有七个字段（不包括 UUID）的表格，而记录有六个字段。请注意，其中一个字段是预留的，未使用，因此它未在表中表示。因此，这些解析字段在表中由单个字段表示。我们保留了每个单独字段的最大字段大小，因此尽管发起城市/州的名称都包含在一个 35 个字符的字段中，但我们仍然在表中保留了城市和州字段各 35 个字节，以避免混淆。

##### 列表 11.4 IAT 附加类型 12 的表格

```py
CREATE TABLE ach_iat_addenda_712_records
(
    ach_records_type_7_id        UUID UNIQUE NOT NULL 
            REFERENCES ach_records_type_7 (ach_records_type_7_id) 
            ON DELETE CASCADE ON UPDATE CASCADE,
    record_type_code             VARCHAR(1)  NOT NULL,
    addenda_type_code            NUMERIC(2)  NOT NULL DEFAULT 12, #1
    originator_city              VARCHAR(35), #2
    originator_state             VARCHAR(35),  #3
    originator_country           VARCHAR(35),  #3
    originator_postal_code       VARCHAR(35), 
    entry_detail_sequence_number NUMERIC(7)  NOT NULL
);
```

#1 **将附加类型代码默认设置为 12，因为此记录必须始终为 12**

#2 **将这些记录保留为 VARCHAR(35)，尽管我们可以考虑强制执行更严格的数据类型**

此代码提供了如何处理所有 IAT 表格的模式。接下来，我们看看记录是如何解析的。

## 11.4 IAT 记录解析

为了解析记录，我们使用我们之前构建的文件创建步骤来创建一个示例文件，然后使用单元测试来构建解析各种记录并将其存储在数据库中的所需代码（列表 11.5）。我们有我们正在寻找解析的样本记录——预期结果。我们通过调用`setup_iat_addenda_test`来设置必要的测试记录，它只是将所需标题和条目记录添加到数据库中，以便所有外键都能按预期工作。然后，我们调用`_parse_iat_addenda_712`和类`AchIat712AddendaSql`。有时，当我们确信我们的逻辑是正确的时候，简单地根据返回值定义`expected_result`可能会很有诱惑力。我们建议不要这样做，并确保记录以另一种方式解析，无论是手动还是使用其他工具，以便结果得到独立验证。

##### 列表 11.5 对 Type 12 附加的解析例程进行单元测试

```py
class TestParsingIat712AddendaRecords:

    TABLE_NAME: str = "ach_iat_addenda_712_records"

    @pytest.fixture(autouse=True)   #1
    def setup_teardown_method(self):  #1
        SqlUtils.truncate_all()  #1
        yield  #1

    def test_parse_iat_addenda_712_records(self):
        sample_addenda_record = "712BILBAO*BIZKAIA\ #2
➥                    ES*48001\                        #2
➥                  0000001" 

record
        expected_result = { #3
            "record_type_code": Literal["7"],  #3
            "addenda_type_code": 12,  #3
            "originator_city": "BILBAO",  #3
            "originator_state": "BIZKAIA",  #3
            "originator_country": "ES",  #3
            "originator_postal_code": "48001",  #3
            "entry_detail_sequence_number": 1,  #3
        }  #3

        _, ach_records_type_7_id = #4
            SqlUtils.setup_iat_addenda_test(  #4
               sample_addenda_record  #4
            )  #4

        parser = AchFileProcessor() #5
        parser._parse_iat_addenda_712(ach_records_type_7_id,  #5
                                      sample_addenda_record) 

        sql = AchIat712AddendaSql() #6
        retrieved_record =  #7
           sql.get_record(ach_records_type_7_id).model_dump(  #7
               exclude={"ach_records_type_7_id"}  #7
           )  #7

        assert SqlUtils.get_row_count_of_1( #7
            self.TABLE_NAME  #7
        ), f"Expected 1 row in {self.TABLE_NAME}"  #7
        assert (  #7
            retrieved_record == expected_result  #7
        ), f"Expected {expected_result}, #7
➥ but got {retrieved_record}"  #7
```

#1 确保数据库为空的固定值；由于我们已将 autouse 设置为 True，因此无需将其包含在我们的测试方法中，因为它将自动执行。

#2 从我们的测试 ACH 文件中取出的样本附加记录

#3 验证检索记录的预期结果字典

#4 设置所需的数据库记录以满足约束条件

#5 解析记录，这也会将其添加到数据库中

#6 检索记录，排除 UUID 字段。由于 UUID 由数据库分配，我们无法在预期结果中硬编码它。

#7 执行断言以确保只有一行，并且记录与之前定义的预期值匹配

之前的单元测试将失败，直到我们构建了所需的功能。我们将通过必要的代码片段来建立模式。所有这些单元测试都将遵循类似的方法。我们遇到的第一个缺失方法是`_parse_iat_addenda_712`方法，如下所示。尽管它很简单，但我们确实需要填充`expected_record_types`，这有助于解析器确定记录是否顺序错误。

我们在更新我们的 ACH 文件处理逻辑时，需要牢记验证，因为我们需要确定是否已收到所有必需的记录，并确保没有重复的附加记录。

##### 列表 11.6 `_parse_iat_addenda_712`记录

```py
def _parse_iat_addenda_712(self, ach_records_type_7_id: UUID, line: str):
     self.expected_record_types = ["6", "7", "8"] #1

     ach_iat_addenda_record = #2
        AchRecordProcessor().parse_iat_addenda_712(  #3
           ach_records_type_7_id, line  #3
        ) 
        AchIat712AddendaSql()#3
           .insert_record(ach_iat_addenda_record) 
```

#1 可以传递给我们的预期记录类型

#2 解析实际上是由 AchRecordProcessor 完成的。在这个时候，我们也可以考虑将一些解析例程移动到新的类中。

#3 使用解析后的记录，我们需要将其插入到数据库中。

接下来，我们创建`parse_iat_addenda_712`方法。这项任务也可能涉及它自己的单独单元测试，因为该类的目的是双重的。首先，它将实际的解析逻辑集中到一个中央位置，这减少了我们的 ACH 文件处理器中的代码，并允许更清晰地理解 ACH 处理流程。其次，它允许我们在不需要在数据库中进行大量设置的情况下单独测试解析逻辑。

然而，解析并不过于复杂，并且将通过此整体过程进行测试，所以目前我们不会担心它需要自己的单元测试。以下列表显示了解析 IAT 附加记录的代码。

##### 列表 11.7 解析 IAT 附加记录

```py
def parse_iat_addenda_712(
        self, ach_records_type_7_id, line
    ) -> AchIat712AddendaSchema:
   regex = r"([^*]+)\*([^\\]+)\\" #1
   match = re.match(regex, line[3:38]) #2
   if not match:  #2
      raise ValueError("Error parsing originator  #2
➥ city and state")  #2
   originator_city, originator_state = match.groups()  #2

   match = re.match(regex, line[38:73]) #3
   if not match:  #3
      raise ValueError("Error parsing originator country   #3
                                and postal code")  #3
   originator_country, originator_postal_code = match.groups()  #3

   return AchIat712AddendaSchema( #4
      ach_records_type_7_id=ach_records_type_7_id,  #4
      record_type_code=line[0],  #4
      addenda_type_code=line[1:3],  #4
      originator_city=originator_city.strip(),  #4
      originator_state=originator_state.strip(),  #4
      originator_country=originator_country.strip(),  #4
      originator_postal_code=originator_postal_code.strip(),  #4
      entry_detail_sequence_number=line[87:94],  #4
   )  #4
```

#1 创建一个正则表达式来解析记录中的分隔字段

#2 确保我们有匹配项并从中提取城市和州

#3 确保我们有匹配项并从中提取国家和邮政编码

#4 解析记录并将其作为我们模式的一部分返回

接下来，我们需要定义用于这些记录的模式。如列表 11.8 所示，我们提供了一个最小布局，以匹配我们期望插入到数据库中的内容。

##### 列表 11.8 IAT 附加 712 模式

```py
class AchIat712AddendaSchema(BaseModel):
    ach_records_type_7_id: UUID
    record_type_code: str = Literal["7"]
    addenda_type_code: int = Literal[12]
    originator_city: str = Field(..., max_length=35)
    originator_state: str = Field(..., max_length=35)
    originator_country: str = Field(..., max_length=35)
    originator_postal_code: str = Field(..., max_length=35)
    entry_detail_sequence_number: int = Field(..., ge=0)
```

最后一步是创建一个类来处理插入和检索记录的 SQL 逻辑。

##### 列表 11.9 IAT 附加类型 12 SQL

```py
class AchIat712AddendaSql: #1
 #2
    def insert_record(self,  #2
                      ach_iat_addenda: AchIat712AddendaSchema): 
        with get_db_connection() as conn: #2
            conn.execute(                 #3
                """  #3
                   INSERT INTO ach_iat_addenda_712_records (  #3
                            ach_records_type_7_id, addenda_type_code,  #3
                            originator_city, originator_state,  #3
                            originator_country, originator_postal_code,  #3
                            entry_detail_sequence_number )  #3
                        VALUES ( %(ach_records_type_7_id)s,  #3
      %(addenda_type_code)s, %(originator_city)s,  #3
      %(originator_state)s,  #3
      %(originator_country)s, %(originator_postal_code)s,   #3
      %(entry_detail_sequence_number)s)  #3
                """,  #3
                ach_iat_addenda.model_dump(), #4
            )
… #5
```

#1 定义一个类和方法来插入我们的记录模式

#2 获取数据库连接

#3 SQL 用于插入模型字段和值

#4 为 INSERT 语句创建一个模式字典

#5 其余的方法，特别是通过 UUID 获取记录

这样，我们应该能够回去确保所有导入语句都已就位，然后成功运行我们的单元测试。假设我们正确解析了记录（对于预期和实际记录），我们应该有一个通过的单位测试。

##### 单元测试 IAT 挑战

这种模式需要为所有新的数据库记录重复，现在是休息一下并编写一些代码的好时机。我们有样本文件。现在的问题仅仅是处理剩余记录的解析。如果之前的代码示例仍然有点令人畏惧，我们也可以从小规模开始，为`AchRecordProcessor`创建单元测试，它只负责解析 ACH 记录，没有数据库交互。测试`Ach-RecordParser`的简单要求意味着设置我们的单元测试需要的工作量更少。一旦在该级别验证了解析，我们可以退后一步，看看更大的图景，并开始编写涉及数据库的单元测试，如本节所述。

虽然我们应该在这个阶段验证所有的解析，但我们仍然需要更新我们的`ach_file_processor`以处理 IAT 批次并利用我们刚刚编写的所有这些美好代码。

## 11.5 IAT 文件处理

到目前为止，我们已经构建了我们 IAT 处理的部分。我们应该相当自信，我们可以处理单个记录并对它们进行解析。现在，我们需要能够在文件中结合实际的 IAT 批次的解析。我们需要记住，我们在前面的章节中测试的是单个部分。例如，我们知道当我们调用`_parse_iat_batch_header`方法并传递一个 IAT 批次头记录时，它将被解析并存储在数据库中。然而，该方法在当前通过`POST`调用加载 ACH 文件的过程中没有被调用。当我们向解析器添加功能时，我们应该牢记一些目标和要求：

+   解析 PPD 批次仍然按预期工作。

+   补充记录全部存在。

+   补充记录的顺序是正确的。
