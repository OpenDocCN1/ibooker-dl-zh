
# 解析 Nacha 批次头记录

```py
Feature: Parsing a Nacha Batch Header

  Scenario: We have a record type of 5
    When We parse the batch header 
➥"5200Company name    DiscretionaryData   Company
 IDARCComp desc 0216232302160471061000010000001"
    ➥ Then The record type should be 5
```

我们定义了功能，然后是一个简单的场景：“我们有一个记录类型 5。”场景可以命名为任何名称，但显然，我们希望传达这个测试将要做什么。`when`和`then`语句是实际工作发生的地方——在我们的例子中，*当*我们在解析头记录时，*然后*我们想要确保我们有一个类型为 5 的记录。

一旦我们有了这些信息，下一步就是将功能文件与所谓的*步骤定义*连接起来。这就是我们将可读文本转换为可以用`pytest`执行的方法。为了简洁，我们将展示“then”步骤定义的一个示例。其余的代码在`test_batch_header.py`中可供您浏览。

##### 列表 2.10 Python 中的行为驱动开发测试

```py
@then(parsers.parse("the record type should be {type_5}"))
def then_record_type_code_should_be(record, type_5):
    assert (
        record["record_type_code"] == type_5
    ), f"Expected {type_5}, but got {record['record_type_code']}"
```

我们使用`@then`装饰器来装饰函数并解析文本字符串。在这个字符串中，我们有`{type_5}`。这是一个动态值，将从特征文件中提取。因此，虽然我们使用了“记录类型应该是 5”，但 5 成为函数的参数，我们可以轻松地创建其他场景来测试其他值。然后我们定义函数，传递给它我们解析的值、一个记录（这是一个 Python fixture——稍后会更详细地介绍）以及一个我们之前见过的`assert`语句。然后这个测试可以与其他任何测试一起运行，当这个测试通过时，我们就完成了编码。当然，我们之前已经编写了这段代码，但在后面的章节中，我们将努力在处理功能之前定义这些场景。我们可能为每个字段提出额外的场景，或者简单地扩展测试中的`then`部分，为每个字段包含多个`then`语句。

## 条目详细记录（类型 6）

条目详细记录包含单个交易数据，包括账户号码和需要贷记或借记账户的金额。请注意，根据正在处理的记录类型，记录的解析可能会有所不同。批次标题记录包含批次中存在的条目记录类型，这被称为标准条目类（SEC）代码。

解析条目详细记录对 ChatGPT 来说更具挑战性，并且在我们为其制定的提示中也是如此。最初，我们尝试了扩展提示

**![image](img/Prompt-Icon.png)** 完全定义`parse_entry_detail`方法并提供一个`pytest`来验证它。

虽然 ChatpGPT 提供了方法和`pytest`，但它未能有意义地解析记录。我们再次尝试

**![image](img/Prompt-Icon.png)** 请使用 NACHA 类型 6 记录的完整布局。

然后，ChatGPT 开始将类型 6 记录称为附加记录，并使用与该记录相关的字段来解析它。我们意识到存在不同类型的条目详细记录（CCD、CTX、PPD 等），因此我们尝试通过

**![image](img/Prompt-Icon.png)** 请使用 NACHA 类型 6 CCD 记录的完整布局。

虽然系统正确地将类型识别为现金集中支付，但它仍然将条目称为附加记录。

使用 Copilot 产生了更好的结果，我们在键入时填充了正确的字段名称，并允许我们快速创建一个只需要稍作更新即可通过测试的模板。然而，到目前为止，我们看到的潜在主题是，虽然这两个工具都很强大，但它们需要我们验证结果，而不是盲目地插入。因此，我们可以看到，如果我们想能够验证我们的结果，了解领域知识是很重要的。

### 附加记录（类型 7）

附加记录包含适用于条目详细记录的额外付款相关信息。每个条目详细记录可能有多个附加记录。

当 Copilot 被提示时

**![图片](img/Prompt-Icon.png)** 定义一个函数来解析 Nacha 附加记录。

它产生了几个代码建议。下面的列表显示了最接近的有效建议。它使用了所有必需的字段，但偏移量并不完全正确。

##### 列表 2.11 附加记录解析函数

```py
def parseAddendaRecord(record):
    # Create a dictionary to hold the parsed record
    parsedRecord = {
        'recordType': record[0:1],
        'typeCode': record[1:2],
        'paymentRelatedInformation': record[2:80],
        'addendaSequenceNumber': record[80:87],
        'entryDetailSequenceNumber': record[87:94]
    }

    return parsedRecord
```

除了一般的解析不正确外，我们更愿意使用蛇形命名法作为字典键。我们可以通过定义一个函数来再次细化提示，该函数使用字典和蛇形命名法解析 Nacha 附加记录。这为我们提供了我们偏好的键格式，但我们仍然需要更新偏移量。

##### 列表 2.12 使用蛇形命名法的附加记录解析函数

```py
def parseAddendaRecordSnake(record):
    # Create a dictionary to hold the parsed record
    parsedRecord = {
        'record_type': record[0:1],
        'type_code': record[1:2],
        'payment_related_information': record[2:80],
        'addenda_sequence_number': record[80:87],
        'entry_detail_sequence_number': record[87:94]
    }

    return parsedRecord
```

为了得到正确的解析，我们需要调整`type_code`、`payment_related_information`和`addenda_sequence_number`。下面的列表显示了可以使用的更新后的返回语句。

##### 列表 2.13 更新后的返回语句，包含已修正的字段

```py
…    
return {
   "record_type_code": record[0],
   "addenda_type_code": record[1:3],
   "payment_related_information": record[3:83],
   "addenda_sequence_number": record[83:87],
   "entry_detail_sequence_number": record[87:94],
}
…
```

### 批次控制记录（类型 8）

批次控制记录是每个批次的尾记录，是一个必需的记录。我们使用记录中的字段，如总借记/贷记金额和记录计数，来验证我们是否收到了正确的批次内容。下面的列表显示了 Copilot 在解析此记录时采取了不同的方法。

##### 列表 2.14 Copilot 解析批次控制记录

```py
def parse_ach_batch_control_record(record):
# Define a dictionary to hold the parsed data
    parsed_record = {}

    # Define the fields in the record
    fields = [
        "record_type",
        "service_class_code",
        "entry_count",
        "entry_hash",
        "total_debit_entry_dollar_amount",
        "total_credit_entry_dollar_amount",
        "company_identification",
        "message_authentication_code",
        "reserved",
        "originating_dfi_identification",
        "batch_number"
    ]

    # Define the field lengths
    field_lengths = [
        1, # record_type
        3, # service_class_code
        6, # entry_count
        10, # entry_hash
        12, # total_debit_entry_dollar_amount
        12, # total_credit_entry_dollar_amount
        10, # company_identification
        19, # message_authentication_code
        6, # reserved
        8, # originating_dfi_identification
        7 # batch_number
    ]

    # Define the field start positions
    field_start_positions = [0]
    for field_length in field_lengths:
        field_start_positions.append(field_start_positions[-1] + field_length)

    # Parse the fields
    for i in range(len(fields)):
        field = fields[i]
        start_position = field_start_positions[i]
        end_position = field_start_positions[i + 1]
        parsed_record[field] = record[start_position:end_position]

    # Return the parsed record
    return parsed_record
```

这段代码看起来有点过度设计，因为这种广泛使用的固定位置文件格式的本质意味着字段位置不会发生变化。就个人喜好而言，我们更喜欢看到实际使用的偏移量，如列表 2.12 所示。偏移量也会使我们在处理解析记录中的错误时工作更轻松。将字段定义为`'entry_detail_sequence_number': record[87:94]`意味着我们知道字段`entry_detail_sequence_number`的开始和结束位置。我们也愿意让这些魔法数字存在于代码中，因为它们仅限于这个特定区域，而不是散布在整个代码中。当然，我们也可以创建名为`BEGIN_ENTRY_DETAIL_SEQUENCE_NUMBER_POS`和`END_ENTRY_DETAIL_SEQUENCE_NUMBER_POS`的变量，并在找到令人信服的理由时使用它们。现在，让我们来看看解析文件尾记录。

### 文件尾记录（类型 9）

在 ACH 文件中的最后一个记录是文件尾记录（也称为文件控制记录）。文件尾记录提供了诸如批次计数和条目/附加条目计数等字段，我们使用这些字段来验证文件是否正确接收。请注意，所需的格式要求记录数必须是 10 的倍数。因此，你可能会发现文件或软件会使用全部为 9 的记录来填充创建的 ACH 文件。然而，大多数软件现在不再需要这样做。

在下面的列表中，我们又回到了 Copilot 按照我们预期的样子解析记录。所有字段都被转换为整数。

##### 列表 2.15 解析文件尾记录

```py
def parse_ach_trailer_record(record):
    return {
        "record_type": record[0],
        "batch_count": int(record[1:7]),
        "block_count": int(record[7:13]),
        "entry_count": int(record[13:21]),
        "entry_hash": int(record[21:31]),
        "total_debit_amount": int(record[31:43]),
        "total_credit_amount": int(record[43:55]),
    }
```

虽然这段代码正确地解析了记录，但我们缺少最后一个字段，该字段被标记为保留。因此，尽管现在可能并不必要，但我们仍然可能想要考虑为了完整性而包括它。

我们还可能想要创建一个示例 BDD 测试，因为，理论上，主题专家（SMEs）可能会为我们提供解析此记录（或任何这些记录）的特定用例。BDD 风格的测试可能看起来像以下列表。

##### 列表 2.16 类型 9 的 BDD 风格测试——文件尾记录

```py
Feature: Parsing a Nacha File Trailer

  Scenario: We have a record type of 9
     Given the File Trailer Record "9000010000010000000740198019800000000007825000114611480"
     When We parse the file trailer
     Then the record type should be 9
     And the batch count should be 10
     And the block count should be 1
     And the entry count should be 74
     And the entry hash should be 0198019800
     And the total debit amount should be 78.25
     And the total credit amount should be 1146114.80
```

### 通过了！

哇，我们做到了！那是一阵编码和测试的狂潮。即使有生成式 AI 的帮助，这也需要很多吸收。现在让我们回顾一下我们刚刚完成的事情。我们开始使用小开发周期来构建单元测试，以验证各种 ACH 记录类型的解析。重要的是要理解，我们是将生成式 AI 创建的程序分解成函数开始的（在我们开始之前，我们要求 ChatGPT 为我们做这件事）。让生成式 AI 创建函数使我们能够更容易地为每种记录类型创建单元测试。否则，我们就必须想出一种方法来确定记录是否被正确解析，并在文件加载后进行检查。

每次我们都是从创建一个应该失败的单元测试开始，然后编写足够的代码使其通过。在这个时候，我们应该相对熟悉这个过程，并准备好将其应用到项目的其他部分。

## 不太愉快的路径

恭喜！我们刚刚解析了我们很可能的第一个 ACH 文件。我们专注于有时被称为“愉快的路径”的部分。这是所有事情都按预期进行，没有抛出任何错误的地方。我们在整个解析过程中使用了一个格式良好的单个 ACH 文件来展示这个过程。

我们现在也应该考虑不太愉快的路径，这可能是我们在日常编码中更有可能遇到的。我们的不太愉快的路径将涵盖在加载文件时可能发生的一些问题。存在文件或批次可能被拒绝或条目可能触发拒绝的场景。我们将在第九章中进一步探讨异常处理和从 ACH 文件问题中恢复。目前，我们只想简要提及一些可能性。

我们将确定一些可能的拒绝场景，并提供用于编码这些场景的示例 ACH 文件和`pytests`。当一个文件被拒绝时，我们通常必须回到原始发送者那里请求一个新的文件。如果有任何交易已经过账，它们可能需要在拒绝过程中进行撤销。目前，我们更专注于识别坏文件，而不是从它们中恢复。当然，如果您想跳过，我们会提供一个解决`pytests`的完整示例。显然，我们会鼓励您通过这些场景，因为我们选择的特定场景不需要对 ACH 处理有广泛的知识。

随着项目的扩展，我们还将添加进一步的验证。目前，我们只是在扩展我们的概念验证。

### 文件拒绝

当格式不正确时，ACH 文件可以被拒绝。记住记录的顺序很重要。一个 ACH 文件由批次组成，每个批次包含条目和附加记录。一个批次和整个文件都有尾记录。所有记录都应该是 94 个字符长。因此，对于第一个场景，我们希望处理可能不是正确长度的记录的文件。

为什么会产生记录长度小于 94 字节的文件？在 SFTP（安全文件传输协议）变得普遍之前，当 ftp 设置为截断尾部空格时，我们会遇到这种情况。虽然 ftp 不像以前那样普遍，但它仍然可以用于内部文件传输，因此原始用例可能是有效的。此外，文件可能被路由和重新传输，甚至可能在某人的计算机上创建/更新。随着银行的合并和收购，这些问题可能仍然会出现。由于 Nacha 规范要求 94 字节的记录，我们将寻求强制执行这一点。我们已经包含了包含空单元测试的`ACH_Parser_v3`，以帮助我们验证我们的更改。
